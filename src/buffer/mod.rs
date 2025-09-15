//! # Buffer Module - Core Ring Buffer Implementation
//! 
//! This module provides the foundation for Perdix's high-performance ring buffer,
//! implementing a lock-free SPSC (Single Producer Single Consumer) design with
//! GPU-CPU shared memory via CUDA unified memory.
//! 
//! ## Architecture
//! 
//! The buffer consists of:
//! - **Header**: 256-byte cache-aligned control structure with atomic counters
//! - **Slots**: Fixed-size 256-byte entries containing message payloads
//! - **Memory**: CUDA unified memory for zero-copy GPU-CPU access
//! 
//! ## Memory Layout
//! 
//! ```text
//! ┌──────────────────────────────────────────┐
//! │ Header (256 bytes, 4 cache lines)        │
//! ├──────────────────────────────────────────┤
//! │ Slot 0 (256 bytes)                       │
//! │   ├─ sequence (8 bytes)                  │
//! │   ├─ length (4 bytes)                    │
//! │   ├─ agent_type (4 bytes)                │
//! │   └─ payload (192 bytes)                 │
//! ├──────────────────────────────────────────┤
//! │ Slot 1 (256 bytes)                       │
//! ├──────────────────────────────────────────┤
//! │ ...                                      │
//! └──────────────────────────────────────────┘
//! ```
//! 
//! ## Submodules
//! 
//! - `device`: Device memory buffer management
//! - `ffi`: Foreign function interface for CUDA interop
//! - `header`: Ring buffer header structure
//! - `pinned`: RAII wrapper for CUDA unified memory
//! - `slot`: Individual message slot structure
//! - `spsc`: Producer and consumer implementations
//! - `gpu_arena`: GPU-optimized text arena allocator (CUDA only)

mod device;
pub mod ffi;
pub mod header;
pub mod pinned;
pub mod slot;
pub mod spsc;
#[cfg(feature = "cuda")]
pub mod gpu_arena;

pub use device::DeviceBuffer;
pub use ffi::*;
pub use header::Header;
pub use slot::Slot;
pub use spsc::{Consumer, Producer, Message};
#[cfg(feature = "cuda")]
pub use gpu_arena::{GpuTextArena, PackedStreamContext};

pub use self::pinned::Pinned;
use std::ptr;

/// The primary owner of the shared GPU-CPU ring buffer.
///
/// This struct manages the underlying pinned memory allocation and provides
/// a safe mechanism to split access into a single `Producer` and a single
/// `Consumer`.
/// 
/// # Examples
/// 
/// ```rust,no_run
/// use perdix::{Buffer, AgentType};
/// 
/// // Create buffer with 1024 slots (must be power of 2)
/// let mut buffer = Buffer::new(1024)?;
/// 
/// // Split into producer/consumer for same-thread usage
/// let (mut producer, mut consumer) = buffer.split_mut();
/// 
/// // Write from producer
/// producer.try_produce(b"Hello", AgentType::System);
/// 
/// // Read from consumer
/// if let Some(msg) = consumer.try_consume() {
///     println!("Got: {}", msg.as_str());
/// }
/// # Ok::<(), String>(())
/// ```
/// 
/// # Performance
/// 
/// The buffer is optimized for:
/// - **Cache Efficiency**: 256-byte aligned slots fit cache lines
/// - **False Sharing Prevention**: Producer/consumer indices separated
/// - **Memory Ordering**: Acquire-release semantics for cross-device visibility
/// - **Batch Processing**: Multiple messages can be written/read per operation
pub struct Buffer {
    pinned: Pinned,
    stream: u64, // We can store the CUDA stream here
}

impl Buffer {
    /// Creates a new ring buffer with the specified number of slots.
    /// 
    /// # Arguments
    /// 
    /// * `n_slots` - Number of slots in the ring buffer (must be power of 2)
    /// 
    /// # Returns
    /// 
    /// Returns `Ok(Buffer)` on success, or an error string describing the failure.
    /// 
    /// # Errors
    /// 
    /// - `n_slots` is not a power of 2
    /// - GPU initialization fails (falls back to CPU mode)
    /// - Memory allocation fails
    /// 
    /// # Backend Selection
    /// 
    /// The function attempts backends in order:
    /// 1. CUDA (if feature enabled and hardware available)
    /// 2. WebGPU (if feature enabled and available)
    /// 3. CPU fallback (demo/testing only, warns user)
    /// 
    /// # Examples
    /// 
    /// ```rust,no_run
    /// use perdix::Buffer;
    /// 
    /// // Create buffer with 4096 slots
    /// let buffer = Buffer::new(4096)?;
    /// 
    /// // Invalid: not power of 2
    /// assert!(Buffer::new(1000).is_err());
    /// # Ok::<(), String>(())
    /// ```
    pub fn new(n_slots: usize) -> Result<Self, String> {
        // Power-of-two check
        if !n_slots.is_power_of_two() {
            return Err("n_slots must be a power of 2".to_string());
        }

        // Try CUDA first if available
        #[cfg(feature = "cuda")]
        {
            if unsafe { ffi::cuda_init_device(0) } >= 0 {
                println!("✓ Using NVIDIA CUDA acceleration");
                return Self::new_cuda(n_slots);
            }
            println!("CUDA not available, trying WebGPU...");
        }
        
        // Try WebGPU as fallback
        #[cfg(feature = "webgpu")]
        {
            if let Ok(buffer) = Self::new_webgpu(n_slots) {
                println!("✓ Using WebGPU acceleration");
                return Ok(buffer);
            }
            println!("WebGPU not available, falling back to CPU...");
        }
        
        // Final fallback to CPU (demo mode)
        println!("⚠️  WARNING: Running in CPU-only mode (demo/testing only)");
        println!("   GPU acceleration is required for production use!");
        Self::new_cpu_fallback(n_slots)
    }
    
    #[cfg(feature = "cuda")]
    fn new_cuda(n_slots: usize) -> Result<Self, String> {
        // Original CUDA initialization code
        if unsafe { ffi::cuda_init_device(0) } < 0 {
            return Err("Failed to initialize CUDA device".to_string());
        }

        // Allocate the unified buffer using our FFI call
        let mut slots_ptr: *mut Slot = ptr::null_mut();
        let mut header_ptr: *mut Header = ptr::null_mut();

        let result = unsafe {
            ffi::init_unified_buffer(
                &mut slots_ptr,
                &mut header_ptr,
                n_slots as std::os::raw::c_int,
            )
        };

        if result != 0 || header_ptr.is_null() || slots_ptr.is_null() {
            return Err("Failed to initialize unified buffer".to_string());
        }

        Ok(Self {
            pinned: Pinned {
                header: header_ptr,
                slots: slots_ptr,
                n_slots,
            },
            stream: 0, // Default stream
        })
    }

    /// Splits the buffer into a producer and a consumer handle.
    ///
    /// This method consumes the `Buffer` to enforce that it can only be split once,
    /// guaranteeing the Single-Producer, Single-Consumer (SPSC) contract.
    ///
    /// The returned `Producer` and `Consumer` are tied to the lifetime of the
    /// `Buffer`'s memory, which is now managed internally and will be dropped
    /// correctly when both handles go out of scope (due to Arc).
    ///
    /// This is an advanced use case if you need to move the handles across threads
    /// while the Buffer is owned elsewhere. A simpler split is `split_mut`.
    /// 
    /// # Thread Safety
    /// 
    /// The returned handles can be safely moved to different threads:
    /// - `Producer` can be moved to a GPU kernel thread or CPU thread
    /// - `Consumer` should remain on CPU for reading messages
    /// 
    /// # Examples
    /// 
    /// ```rust,no_run
    /// use perdix::Buffer;
    /// use std::thread;
    /// 
    /// let buffer = Buffer::new(1024)?;
    /// let (producer, consumer) = buffer.split();
    /// 
    /// // Move producer to another thread
    /// let producer_thread = thread::spawn(move || {
    ///     // Producer can write from this thread
    /// });
    /// 
    /// // Consumer reads in main thread
    /// // consumer.try_consume();
    /// # Ok::<(), String>(())
    /// ```
    pub fn split(self) -> (Producer<'static>, Consumer<'static>) {
        // This is a bit tricky. To make the handles independent and movable,
        // we need to put the Pinned buffer into an Arc. This allows the producer
        // and consumer to share ownership of the allocation.
        let arc_pinned = std::sync::Arc::new(self.pinned);
        let producer = Producer::new(arc_pinned.clone());
        let consumer = Consumer::new(arc_pinned);

        (producer, consumer)
    }

    /// Splits the buffer into mutable producer and consumer handles with lifetimes.
    ///
    /// This is a simpler and more common use case where the buffer outlives the
    /// handles within the same scope.
    pub fn split_mut<'a>(&'a mut self) -> (Producer<'a>, Consumer<'a>) {
        // We can create two handles that borrow the same Pinned buffer.
        // Rust's borrow checker would normally prevent this (one mutable borrow
        // at a time), but since our Producer and Consumer operate on raw pointers
        // internally and we guarantee they don't conflict, this is a safe
        // abstraction. We can achieve this by using raw pointers.

        let pinned_ptr = &mut self.pinned as *mut Pinned;

        // SAFETY: We are creating two structs that hold references derived from
        // the same mutable borrow of `self`. This is safe because the Producer
        // and Consumer are designed to never access the same memory locations
        // concurrently in a conflicting manner. They operate on different indices
        // of the ring buffer. The lifetime 'a ensures they cannot outlive the Buffer.
        unsafe {
            let producer = Producer::new_from_ref(&*pinned_ptr);
            let consumer = Consumer::new_from_ref(&*pinned_ptr);
            (producer, consumer)
        }
    }

    /// Provides raw pointers to the underlying memory for FFI (e.g., launching a CUDA kernel).
    /// 
    /// # Safety
    /// 
    /// The returned pointers are valid for the lifetime of the Buffer.
    /// Callers must ensure:
    /// - No concurrent modifications through these pointers
    /// - Proper synchronization when used with GPU kernels
    /// - Memory fence operations for cross-device visibility
    /// 
    /// # Returns
    /// 
    /// Tuple of `(*mut Header, *mut Slot)` pointers to the buffer's memory.
    /// 
    /// # Examples
    /// 
    /// ```rust,no_run
    /// use perdix::Buffer;
    /// 
    /// let buffer = Buffer::new(1024)?;
    /// let (header_ptr, slots_ptr) = buffer.as_raw_parts();
    /// 
    /// // Pass to CUDA kernel launch
    /// unsafe {
    ///     // launch_cuda_kernel(header_ptr, slots_ptr, 1024);
    /// }
    /// # Ok::<(), String>(())
    /// ```
    pub fn as_raw_parts(&self) -> (*mut Header, *mut Slot) {
        (self.pinned.as_header(), self.pinned.as_slots())
    }
}

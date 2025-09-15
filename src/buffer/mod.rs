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
pub struct Buffer {
    pinned: Pinned,
    stream: u64, // We can store the CUDA stream here
}

impl Buffer {
    pub fn new(n_slots: usize) -> Result<Self, String> {
        // Power-of-two check
        if !n_slots.is_power_of_two() {
            return Err("n_slots must be a power of 2".to_string());
        }

        // Initialize CUDA device
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
    pub fn as_raw_parts(&self) -> (*mut Header, *mut Slot) {
        (self.pinned.as_header(), self.pinned.as_slots())
    }
}

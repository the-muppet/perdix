//! # FFI Module - CUDA Foreign Function Interface
//! 
//! This module provides the bridge between Rust and CUDA C++ code, defining
//! the foreign function interface for GPU operations. All FFI boundaries are
//! carefully designed to ensure memory safety and proper error handling.
//! 
//! ## Safety Requirements
//! 
//! All FFI functions in this module are `unsafe` and require:
//! 
//! 1. **Valid Pointers**: All pointer arguments must be valid and aligned
//! 2. **CUDA Initialization**: Device must be initialized before kernel launches
//! 3. **Memory Lifetime**: Buffers must outlive kernel execution
//! 4. **Synchronization**: Proper stream synchronization for async operations
//! 
//! ## Error Codes
//! 
//! FFI functions return negative values on error:
//! - `-1`: CUDA initialization or device error
//! - `-2`: Memory allocation failure
//! - `-3`: Invalid parameters
//! - `-4`: Kernel launch failure
//! - `-5`: Stream synchronization error

use crate::buffer::header::Header;
use crate::buffer::slot::Slot;
#[cfg(feature = "cuda")]
use crate::buffer::gpu_arena::PackedStreamContext;
use std::os::raw::c_int;

/// Agent types for categorizing message sources.
/// 
/// These types match the CUDA kernel definitions and are used to
/// categorize messages for routing and formatting.
/// 
/// # Memory Layout
/// 
/// Uses `#[repr(u8)]` to ensure single-byte representation matching
/// the CUDA kernel's enum definition.
#[repr(u8)]
#[derive(Debug, Clone, Copy)]
pub enum AgentType {
    System = 0,
    User = 1,
    Assistant = 2,
    Error = 3,
    Warning = 4,
    Info = 5,
    Debug = 6,
    Trace = 7,
}

/// Stream context for AI agent responses.
/// 
/// This structure contains all metadata needed to process a message
/// through the GPU pipeline. It's designed to be POD (Plain Old Data)
/// for safe FFI transmission to CUDA kernels.
/// 
/// # Memory Layout
/// 
/// Uses `#[repr(C)]` to ensure C-compatible memory layout for FFI.
/// The structure is carefully padded to avoid alignment issues.
/// 
/// # Safety
/// 
/// The `text` pointer must remain valid for the duration of kernel execution.
/// Typically, this is ensured by keeping the source data alive in the caller.
#[repr(C)]
#[derive(Debug, Clone)]
pub struct StreamContext {
    pub text: *const u8,
    pub text_len: u32,
    pub agent_type: AgentType,
    pub stream_id: u32,
    pub timestamp: u64,
    pub is_continuation: bool,
    pub enable_ansi: bool,
    _pad: [u8; 2],
}

impl StreamContext {
    pub fn new(text: &[u8], agent_type: AgentType) -> Self {
        Self {
            text: text.as_ptr(),
            text_len: text.len() as u32,
            agent_type,
            stream_id: 0,
            timestamp: 0,
            is_continuation: false,
            enable_ansi: true,
            _pad: [0; 2],
        }
    }
}

extern "C" {
    /// Initialize CUDA device for ring buffer operations.
    /// 
    /// # Safety
    /// 
    /// This function is unsafe because it:
    /// - Modifies global CUDA state
    /// - Must be called before any other CUDA operations
    /// - Can fail if device doesn't exist or is unavailable
    /// 
    /// # Arguments
    /// 
    /// * `device_id` - CUDA device index (typically 0)
    /// 
    /// # Returns
    /// 
    /// - `0`: Success
    /// - `-1`: Device initialization failed
    /// - `-2`: Invalid device ID
    pub fn cuda_init_device(device_id: c_int) -> c_int;

    /// Allocate and initialize CUDA unified memory buffer.
    /// 
    /// # Safety
    /// 
    /// This function is unsafe because it:
    /// - Allocates GPU memory that must be freed with `cleanup_unified_buffer`
    /// - Writes to provided pointers
    /// - Requires CUDA device to be initialized
    /// 
    /// # Arguments
    /// 
    /// * `slots` - Output pointer for slot array
    /// * `hdr` - Output pointer for header structure
    /// * `n_slots` - Number of slots (must be power of 2)
    /// 
    /// # Returns
    /// 
    /// - `0`: Success
    /// - `-1`: CUDA allocation failed
    /// - `-2`: Invalid parameters
    /// - `-3`: Not a power of 2
    pub fn init_unified_buffer(
        slots: *mut *mut Slot,
        hdr: *mut *mut Header,
        n_slots: c_int,
    ) -> c_int;

    /// Launch GPU kernel to process messages.
    /// 
    /// # Safety
    /// 
    /// This function is unsafe because it:
    /// - Launches asynchronous GPU operations
    /// - Requires valid buffer pointers from `init_unified_buffer`
    /// - Contexts must remain valid until kernel completes
    /// - Stream must be a valid CUDA stream or null for default
    /// 
    /// # Arguments
    /// 
    /// * `slots` - Ring buffer slot array
    /// * `hdr` - Ring buffer header
    /// * `contexts` - Array of message contexts
    /// * `n_messages` - Number of messages to process
    /// * `enable_metrics` - Whether to collect performance metrics
    /// * `stream` - CUDA stream for async execution (null for default)
    /// 
    /// # Returns
    /// 
    /// - `0`: Success
    /// - `-4`: Kernel launch failed
    /// - `-5`: Invalid parameters
    pub fn launch_unified_kernel(
        slots: *mut Slot,
        hdr: *mut Header,
        contexts: *const StreamContext,
        n_messages: u32,
        enable_metrics: i32,
        stream: *mut std::ffi::c_void,  // cudaStream_t is a pointer type
    ) -> c_int;

    pub fn launch_simple_test(slots: *mut Slot, hdr: *mut Header, n_msgs: c_int) -> c_int;
    
    pub fn launch_unified_kernel_async(
        slots: *mut Slot,
        hdr: *mut Header,
        packed_contexts: *const PackedStreamContext,
        text_arena: *const u8,
        n_messages: u32,
        enable_metrics: i32,
        stream: *mut std::ffi::c_void,
    ) -> c_int;

    /// Free CUDA unified memory buffer.
    /// 
    /// # Safety
    /// 
    /// This function is unsafe because it:
    /// - Frees GPU memory allocated by `init_unified_buffer`
    /// - Must only be called once per allocation
    /// - Pointers become invalid after this call
    /// 
    /// # Arguments
    /// 
    /// * `slots` - Slot array to free
    /// * `hdr` - Header structure to free
    /// 
    /// # Returns
    /// 
    /// - `0`: Success
    /// - `-1`: CUDA free failed
    pub fn cleanup_unified_buffer(slots: *mut Slot, hdr: *mut Header) -> c_int;
}

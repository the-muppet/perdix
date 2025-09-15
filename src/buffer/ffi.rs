use crate::buffer::header::Header;
use crate::buffer::slot::Slot;
use crate::buffer::gpu_arena::PackedStreamContext;
use std::os::raw::c_int;

// Agent types matching CUDA kernel
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

// Stream context for AI agent responses
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
    pub fn cuda_init_device(device_id: c_int) -> c_int;

    pub fn init_unified_buffer(
        slots: *mut *mut Slot,
        hdr: *mut *mut Header,
        n_slots: c_int,
    ) -> c_int;

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

    pub fn cleanup_unified_buffer(slots: *mut Slot, hdr: *mut Header) -> c_int;
}

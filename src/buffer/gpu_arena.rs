use std::os::raw::c_void;
use crate::buffer::ffi::{AgentType, StreamContext};

/// GPU-friendly packed context using offsets instead of pointers
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PackedStreamContext {
    pub text_offset: u32,
    pub text_len: u32,
    pub agent_type: AgentType,
    pub stream_id: u32,
    pub timestamp: u64,
    pub is_continuation: bool,
    pub enable_ansi: bool,
    _pad: [u8; 2],
}

/// Arena allocator for GPU text data
pub struct GpuTextArena {
    /// Host-side staging buffer for text
    host_text: Vec<u8>,
    /// Host-side packed contexts
    host_contexts: Vec<PackedStreamContext>,
    /// Device text buffer
    device_text: *mut u8,
    /// Device contexts buffer
    device_contexts: *mut PackedStreamContext,
    /// Current offset in text buffer
    current_offset: usize,
    /// Maximum arena size
    max_size: usize,
    /// CUDA stream for async operations
    stream: *mut c_void,
}

impl GpuTextArena {
    pub fn new(max_size: usize) -> Result<Self, String> {
        // Create CUDA stream for async operations
        let stream = unsafe {
            let mut stream: *mut c_void = std::ptr::null_mut();
            let result = cuda_create_stream(&mut stream);
            if result != 0 {
                return Err(format!("Failed to create CUDA stream: {}", result));
            }
            stream
        };
        
        Ok(Self {
            host_text: Vec::with_capacity(max_size),
            host_contexts: Vec::with_capacity(1024),
            device_text: std::ptr::null_mut(),
            device_contexts: std::ptr::null_mut(),
            current_offset: 0,
            max_size,
            stream,
        })
    }
    
    /// Pack messages into arena for GPU processing
    pub fn pack_messages(&mut self, contexts: &[StreamContext]) -> Result<(), String> {
        // Clear previous data
        self.host_text.clear();
        self.host_contexts.clear();
        self.current_offset = 0;
        
        // Pack each message
        for ctx in contexts {
            // Check if we have space
            let text_slice = unsafe {
                std::slice::from_raw_parts(ctx.text, ctx.text_len as usize)
            };
            
            if self.current_offset + text_slice.len() > self.max_size {
                return Err("Text arena overflow".to_string());
            }
            
            // Create packed context with offset
            let packed = PackedStreamContext {
                text_offset: self.current_offset as u32,
                text_len: ctx.text_len,
                agent_type: ctx.agent_type,
                stream_id: ctx.stream_id,
                timestamp: ctx.timestamp,
                is_continuation: ctx.is_continuation,
                enable_ansi: ctx.enable_ansi,
                _pad: [0; 2],
            };
            
            // Copy text to arena
            self.host_text.extend_from_slice(text_slice);
            self.host_contexts.push(packed);
            self.current_offset += text_slice.len();
        }
        
        Ok(())
    }
    
    /// Allocate device memory and copy data asynchronously
    pub fn upload_to_device_async(&mut self) -> Result<(), String> {
        unsafe {
            // Allocate device memory for text
            if !self.device_text.is_null() {
                cuda_free_async(self.device_text as *mut c_void, self.stream);
            }
            
            let text_size = self.host_text.len();
            if text_size > 0 {
                let result = cuda_malloc_async(
                    &mut self.device_text as *mut *mut u8 as *mut *mut c_void,
                    text_size,
                    self.stream
                );
                if result != 0 {
                    return Err(format!("Failed to allocate device text: {}", result));
                }
                
                // Copy text to device asynchronously
                let result = cuda_memcpy_async(
                    self.device_text as *mut c_void,
                    self.host_text.as_ptr() as *const c_void,
                    text_size,
                    CudaMemcpyKind::HostToDevice,
                    self.stream
                );
                if result != 0 {
                    return Err(format!("Failed to copy text to device: {}", result));
                }
            }
            
            // Allocate device memory for contexts
            if !self.device_contexts.is_null() {
                cuda_free_async(self.device_contexts as *mut c_void, self.stream);
            }
            
            let contexts_size = self.host_contexts.len() * std::mem::size_of::<PackedStreamContext>();
            if contexts_size > 0 {
                let result = cuda_malloc_async(
                    &mut self.device_contexts as *mut *mut PackedStreamContext as *mut *mut c_void,
                    contexts_size,
                    self.stream
                );
                if result != 0 {
                    return Err(format!("Failed to allocate device contexts: {}", result));
                }
                
                // Copy contexts to device asynchronously
                let result = cuda_memcpy_async(
                    self.device_contexts as *mut c_void,
                    self.host_contexts.as_ptr() as *const c_void,
                    contexts_size,
                    CudaMemcpyKind::HostToDevice,
                    self.stream
                );
                if result != 0 {
                    return Err(format!("Failed to copy contexts to device: {}", result));
                }
            }
        }
        
        Ok(())
    }
    
    /// Launch kernel asynchronously (no synchronization!)
    pub fn launch_kernel_async(
        &self,
        slots: *mut crate::buffer::Slot,
        header: *mut crate::buffer::Header,
        enable_metrics: bool,
    ) -> Result<(), String> {
        if self.host_contexts.is_empty() {
            return Ok(());
        }
        
        unsafe {
            let result = launch_unified_kernel_async(
                slots,
                header,
                self.device_contexts,
                self.device_text,
                self.host_contexts.len() as u32,
                enable_metrics as i32,
                self.stream,
            );
            
            if result != 0 {
                return Err(format!("Kernel launch failed: {}", result));
            }
        }
        
        Ok(())
    }
    
    /// Check if kernel is done (non-blocking)
    pub fn is_kernel_done(&self) -> bool {
        unsafe {
            cuda_stream_query(self.stream) == 0
        }
    }
    
    /// Get the CUDA stream handle
    pub fn stream(&self) -> *mut c_void {
        self.stream
    }
    
    /// Add a single text message to the arena
    pub fn add_text(&mut self, text: &[u8], agent_type: AgentType) -> Result<(), String> {
        if self.current_offset + text.len() > self.capacity {
            return Err("Arena capacity exceeded".to_string());
        }
        
        let packed = PackedStreamContext {
            text_offset: self.current_offset as u32,
            text_len: text.len() as u32,
            agent_type,
            stream_id: 0,
            timestamp: 0,
            is_continuation: false,
            enable_ansi: true,
            _pad: [0; 2],
        };
        
        self.host_text.extend_from_slice(text);
        self.host_contexts.push(packed);
        self.current_offset += text.len();
        
        Ok(())
    }
    
    /// Get the packed contexts and text data
    pub fn pack(&self) -> (&[PackedStreamContext], &[u8]) {
        (&self.host_contexts, &self.host_text)
    }
    
    /// Upload packed data to device
    pub fn upload_to_device(
        &mut self,
        contexts: &[PackedStreamContext],
        text_data: &[u8],
    ) -> Result<(), String> {
        // Use the async version internally
        self.host_contexts = contexts.to_vec();
        self.host_text = text_data.to_vec();
        self.upload_to_device_async()
    }
    
    /// Get device pointers for kernel launch
    pub fn get_device_pointers(&self) -> (*const PackedStreamContext, *const u8, *mut c_void) {
        (self.device_contexts, self.device_text, self.stream)
    }
}

impl Drop for GpuTextArena {
    fn drop(&mut self) {
        unsafe {
            // Clean up device memory
            if !self.device_text.is_null() {
                cuda_free_async(self.device_text as *mut c_void, self.stream);
            }
            if !self.device_contexts.is_null() {
                cuda_free_async(self.device_contexts as *mut c_void, self.stream);
            }
            
            // Synchronize and destroy stream
            cuda_stream_synchronize(self.stream);
            cuda_stream_destroy(self.stream);
        }
    }
}

// CUDA FFI functions
#[repr(C)]
enum CudaMemcpyKind {
    HostToHost = 0,
    HostToDevice = 1,
    DeviceToHost = 2,
    DeviceToDevice = 3,
}

extern "C" {
    fn cuda_create_stream(stream: *mut *mut c_void) -> i32;
    fn cuda_stream_destroy(stream: *mut c_void) -> i32;
    fn cuda_stream_synchronize(stream: *mut c_void) -> i32;
    fn cuda_stream_query(stream: *mut c_void) -> i32;
    fn cuda_malloc_async(ptr: *mut *mut c_void, size: usize, stream: *mut c_void) -> i32;
    fn cuda_free_async(ptr: *mut c_void, stream: *mut c_void) -> i32;
    fn cuda_memcpy_async(
        dst: *mut c_void,
        src: *const c_void,
        size: usize,
        kind: CudaMemcpyKind,
        stream: *mut c_void,
    ) -> i32;
    
    fn launch_unified_kernel_async(
        slots: *mut crate::buffer::Slot,
        header: *mut crate::buffer::Header,
        packed_contexts: *const PackedStreamContext,
        text_arena: *const u8,
        n_messages: u32,
        enable_metrics: i32,
        stream: *mut c_void,
    ) -> i32;
}
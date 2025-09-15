/// CUDA Driver API parameter passing implementation
/// Based on official CUDA documentation
/// 
/// The cuLaunchKernel function expects kernelParams to be an array of pointers
/// where each pointer points to the actual parameter value.
use std::os::raw::c_void;

pub struct CudaParams {
    // Store actual parameter values directly
    device_ptrs: Vec<usize>,  // Device pointer values (as usize)
    i32_values: Vec<i32>,     // i32 values
    
    // Store pointers to the values above
    param_ptrs: Vec<*mut c_void>,
}

impl CudaParams {
    pub fn new() -> Self {
        Self {
            device_ptrs: Vec::new(),
            i32_values: Vec::new(),
            param_ptrs: Vec::new(),
        }
    }
    
    /// Add a device pointer parameter
    /// For device pointers (CUdeviceptr), we pass the pointer value itself
    pub fn add_device_ptr(&mut self, device_ptr: *mut c_void) {
        // Store the pointer value
        self.device_ptrs.push(device_ptr as usize);
        
        // Get a stable pointer to the stored value
        let ptr = self.device_ptrs.last().unwrap() as *const usize as *mut c_void;
        self.param_ptrs.push(ptr);
    }
    
    /// Add an i32 parameter
    pub fn add_i32(&mut self, value: i32) {
        // Store the value
        self.i32_values.push(value);
        
        // Get a stable pointer to the stored value
        let ptr = self.i32_values.last().unwrap() as *const i32 as *mut c_void;
        self.param_ptrs.push(ptr);
    }
    
    /// Get the parameter array for cuLaunchKernel
    /// IMPORTANT: This struct must remain alive during kernel execution
    pub fn as_kernel_params(&self) -> &[*mut c_void] {
        &self.param_ptrs
    }
}
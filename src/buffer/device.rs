use std::os::raw::c_void;
use crate::runtime::CompilerError;

/// Device memory buffer
pub struct DeviceBuffer {
    ptr: u64,
    size: usize,
}

impl DeviceBuffer {
    pub fn new(size: usize) -> Result<Self, CompilerError> {
        unsafe {
            let mut ptr = 0u64;
            let result = crate::runtime::runtime_compiler::cuMemAlloc(&mut ptr, size);
            if result != crate::runtime::runtime_compiler::CUDA_SUCCESS {
                return Err(CompilerError::CudaError(format!(
                    "Failed to allocate device memory: {}",
                    result
                )));
            }

            Ok(Self { ptr, size })
        }
    }

    pub fn copy_from_host(&mut self, data: &[u8]) -> Result<(), CompilerError> {
        if data.len() > self.size {
            return Err(CompilerError::CudaError("Data too large for buffer".into()));
        }

        unsafe {
            let result = crate::runtime::runtime_compiler::cuMemcpyHtoD(self.ptr, data.as_ptr() as *const c_void, data.len());
            if result != crate::runtime::runtime_compiler::CUDA_SUCCESS {
                return Err(CompilerError::CudaError(format!(
                    "Failed to copy to device: {}",
                    result
                )));
            }
        }

        Ok(())
    }

    pub fn copy_to_host(&self, data: &mut [u8]) -> Result<(), CompilerError> {
        if data.len() > self.size {
            return Err(CompilerError::CudaError("Buffer too small".into()));
        }

        unsafe {
            let result = crate::runtime::runtime_compiler::cuMemcpyDtoH(data.as_mut_ptr() as *mut c_void, self.ptr, data.len());
            if result != crate::runtime::runtime_compiler::CUDA_SUCCESS {
                return Err(CompilerError::CudaError(format!(
                    "Failed to copy from device: {}",
                    result
                )));
            }
        }

        Ok(())
    }

    pub fn as_ptr(&self) -> u64 {
        self.ptr
    }
}

impl Drop for DeviceBuffer {
    fn drop(&mut self) {
        unsafe {
            crate::runtime::runtime_compiler::cuMemFree(self.ptr);
        }
    }
}

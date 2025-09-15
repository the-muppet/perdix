use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int, c_void};
use std::ptr;

// NVRTC FFI bindings
#[cfg(feature = "cuda")]
#[link(name = "nvrtc")]
extern "C" {
    fn nvrtcCreateProgram(
        prog: *mut *mut c_void,
        src: *const c_char,
        name: *const c_char,
        num_headers: c_int,
        headers: *const *const c_char,
        include_names: *const *const c_char,
    ) -> c_int;

    fn nvrtcCompileProgram(
        prog: *mut c_void,
        num_options: c_int,
        options: *const *const c_char,
    ) -> c_int;

    fn nvrtcGetPTXSize(prog: *mut c_void, ptx_size: *mut usize) -> c_int;
    fn nvrtcGetPTX(prog: *mut c_void, ptx: *mut c_char) -> c_int;
    fn nvrtcDestroyProgram(prog: *mut *mut c_void) -> c_int;
    fn nvrtcGetErrorString(result: c_int) -> *const c_char;
    fn nvrtcGetProgramLog(prog: *mut c_void, log: *mut c_char) -> c_int;
    fn nvrtcGetProgramLogSize(prog: *mut c_void, log_size: *mut usize) -> c_int;
}

// CUDA Driver API bindings for PTX loading
#[cfg(feature = "cuda")]
#[link(name = "cuda")]
extern "C" {
    fn cuInit(flags: c_int) -> c_int;
    fn cuDeviceGet(device: *mut c_int, ordinal: c_int) -> c_int;
    fn cuCtxCreate(ctx: *mut *mut c_void, flags: c_int, device: c_int) -> c_int;
    fn cuModuleLoadData(module: *mut *mut c_void, image: *const c_void) -> c_int;
    fn cuModuleGetFunction(
        func: *mut *mut c_void,
        module: *mut c_void,
        name: *const c_char,
    ) -> c_int;
    fn cuLaunchKernel(
        f: *mut c_void,
        grid_dim_x: c_int,
        grid_dim_y: c_int,
        grid_dim_z: c_int,
        block_dim_x: c_int,
        block_dim_y: c_int,
        block_dim_z: c_int,
        shared_mem_bytes: c_int,
        stream: *mut c_void,
        kernel_params: *mut *mut c_void,
        extra: *mut *mut c_void,
    ) -> c_int;
    fn cuCtxSynchronize() -> c_int;
    pub fn cuMemAlloc(dptr: *mut u64, bytesize: usize) -> c_int;
    pub fn cuMemFree(dptr: u64) -> c_int;
    pub fn cuMemcpyHtoD(dst: u64, src: *const c_void, bytesize: usize) -> c_int;
    pub fn cuMemcpyDtoH(dst: *mut c_void, src: u64, bytesize: usize) -> c_int;
    fn cuMemAllocHost(pp: *mut *mut c_void, bytesize: usize) -> c_int;
    fn cuMemFreeHost(p: *mut c_void) -> c_int;
}

// Stub implementations when CUDA is not available
#[cfg(not(feature = "cuda"))]
mod stubs {
    use super::*;
    
    pub unsafe fn nvrtcCreateProgram(_: *mut *mut c_void, _: *const c_char, _: *const c_char, _: c_int, _: *const *const c_char, _: *const *const c_char) -> c_int { -1 }
    pub unsafe fn nvrtcCompileProgram(_: *mut c_void, _: c_int, _: *const *const c_char) -> c_int { -1 }
    pub unsafe fn nvrtcGetPTXSize(_: *mut c_void, _: *mut usize) -> c_int { -1 }
    pub unsafe fn nvrtcGetPTX(_: *mut c_void, _: *mut c_char) -> c_int { -1 }
    pub unsafe fn nvrtcDestroyProgram(_: *mut *mut c_void) -> c_int { -1 }
    pub unsafe fn nvrtcGetErrorString(_: c_int) -> *const c_char { b"CUDA not available\0".as_ptr() as *const c_char }
    pub unsafe fn nvrtcGetProgramLog(_: *mut c_void, _: *mut c_char) -> c_int { -1 }
    pub unsafe fn nvrtcGetProgramLogSize(_: *mut c_void, _: *mut usize) -> c_int { -1 }
    
    pub unsafe fn cuInit(_: c_int) -> c_int { -1 }
    pub unsafe fn cuDeviceGet(_: *mut c_int, _: c_int) -> c_int { -1 }
    pub unsafe fn cuCtxCreate(_: *mut *mut c_void, _: c_int, _: c_int) -> c_int { -1 }
    pub unsafe fn cuModuleLoadData(_: *mut *mut c_void, _: *const c_void) -> c_int { -1 }
    pub unsafe fn cuModuleGetFunction(_: *mut *mut c_void, _: *mut c_void, _: *const c_char) -> c_int { -1 }
    pub unsafe fn cuLaunchKernel(_: *mut c_void, _: c_int, _: c_int, _: c_int, _: c_int, _: c_int, _: c_int, _: c_int, _: *mut c_void, _: *mut *mut c_void, _: *mut *mut c_void) -> c_int { -1 }
    pub unsafe fn cuCtxSynchronize() -> c_int { -1 }
    pub unsafe fn cuMemAlloc(_: *mut u64, _: usize) -> c_int { -1 }
    pub unsafe fn cuMemFree(_: u64) -> c_int { -1 }
    pub unsafe fn cuMemcpyHtoD(_: u64, _: *const c_void, _: usize) -> c_int { -1 }
    pub unsafe fn cuMemcpyDtoH(_: *mut c_void, _: u64, _: usize) -> c_int { -1 }
    pub unsafe fn cuMemAllocHost(_: *mut *mut c_void, _: usize) -> c_int { -1 }
    pub unsafe fn cuMemFreeHost(_: *mut c_void) -> c_int { -1 }
}

#[cfg(not(feature = "cuda"))]
pub use stubs::*;

// Error codes
pub const NVRTC_SUCCESS: c_int = 0;
pub const CUDA_SUCCESS: c_int = 0;

#[derive(Debug)]
pub enum CompilerError {
    NvrtcError(String),
    CudaError(String),
    InvalidKernel,
    CompilationFailed(String),
}

impl std::fmt::Display for CompilerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CompilerError::NvrtcError(msg) => write!(f, "NVRTC Error: {}", msg),
            CompilerError::CudaError(msg) => write!(f, "CUDA Error: {}", msg),
            CompilerError::InvalidKernel => write!(f, "Invalid kernel"),
            CompilerError::CompilationFailed(log) => write!(f, "Compilation failed:\n{}", log),
        }
    }
}

impl std::error::Error for CompilerError {}

type Result<T> = std::result::Result<T, CompilerError>;

/// Runtime CUDA compiler using NVRTC
pub struct CudaRuntimeCompiler {
    pub context: *mut c_void,
    pub device: c_int,
}

impl CudaRuntimeCompiler {
    pub fn new(device_id: i32) -> Result<Self> {
        unsafe {
            // Initialize CUDA driver
            let result = cuInit(0);
            if result != CUDA_SUCCESS {
                return Err(CompilerError::CudaError(format!(
                    "Failed to initialize CUDA driver: {}",
                    result
                )));
            }

            // Get device
            let mut device = 0;
            let result = cuDeviceGet(&mut device, device_id);
            if result != CUDA_SUCCESS {
                return Err(CompilerError::CudaError(format!(
                    "Failed to get device {}: {}",
                    device_id, result
                )));
            }

            // Create context
            let mut context = ptr::null_mut();
            let result = cuCtxCreate(&mut context, 0, device);
            if result != CUDA_SUCCESS {
                return Err(CompilerError::CudaError(format!(
                    "Failed to create context: {}",
                    result
                )));
            }

            Ok(Self { context, device })
        }
    }

    /// Compile CUDA source code to PTX
    pub fn compile_to_ptx(
        &self,
        source: &str,
        kernel_name: &str,
        compute_capability: &str,
    ) -> Result<String> {
        unsafe {
            let src_cstring = CString::new(source).unwrap();
            let name_cstring = CString::new(kernel_name).unwrap();

            // Create program
            let mut prog = ptr::null_mut();
            let result = nvrtcCreateProgram(
                &mut prog,
                src_cstring.as_ptr(),
                name_cstring.as_ptr(),
                0,
                ptr::null(),
                ptr::null(),
            );

            if result != NVRTC_SUCCESS {
                let error_str = CStr::from_ptr(nvrtcGetErrorString(result))
                    .to_string_lossy()
                    .to_string();
                return Err(CompilerError::NvrtcError(error_str));
            }

            // Compile options
            let arch_option = format!("--gpu-architecture={}", compute_capability);
            let options = vec![
                CString::new(arch_option).unwrap(),
                CString::new("--relocatable-device-code=false").unwrap(),
                CString::new("--use_fast_math").unwrap(),
            ];
            let option_ptrs: Vec<*const c_char> =
                options.iter().map(|s| s.as_ptr()).collect();

            // Compile
            let result = nvrtcCompileProgram(prog, option_ptrs.len() as c_int, option_ptrs.as_ptr());

            // Get compilation log
            let mut log_size = 0;
            nvrtcGetProgramLogSize(prog, &mut log_size);
            let mut log = vec![0u8; log_size];
            nvrtcGetProgramLog(prog, log.as_mut_ptr() as *mut c_char);
            let log_str = String::from_utf8_lossy(&log).to_string();

            if result != NVRTC_SUCCESS {
                nvrtcDestroyProgram(&mut prog);
                return Err(CompilerError::CompilationFailed(log_str));
            }

            // Get PTX
            let mut ptx_size = 0;
            nvrtcGetPTXSize(prog, &mut ptx_size);
            let mut ptx = vec![0u8; ptx_size];
            nvrtcGetPTX(prog, ptx.as_mut_ptr() as *mut c_char);

            // Cleanup
            nvrtcDestroyProgram(&mut prog);

            // Remove null terminator if present
            if ptx_size > 0 && ptx[ptx_size - 1] == 0 {
                ptx.truncate(ptx_size - 1);
            }
            
            Ok(String::from_utf8_lossy(&ptx).to_string())
        }
    }

    /// Load PTX and get kernel function
    pub fn load_ptx_module(&self, ptx: &str) -> Result<CudaModule> {
        unsafe {
            let ptx_cstring = CString::new(ptx).unwrap();
            let mut module = ptr::null_mut();

            let result = cuModuleLoadData(&mut module, ptx_cstring.as_ptr() as *const c_void);
            if result != CUDA_SUCCESS {
                return Err(CompilerError::CudaError(format!(
                    "Failed to load PTX module: {}",
                    result
                )));
            }

            Ok(CudaModule { module })
        }
    }
}

impl Drop for CudaRuntimeCompiler {
    fn drop(&mut self) {
        // Context is automatically destroyed when the process exits
        // For production, you might want to explicitly destroy it
    }
}

/// CUDA module loaded from PTX
pub struct CudaModule {
    pub module: *mut c_void,
}

impl CudaModule {
    /// Get a kernel function from the module
    pub fn get_function(&self, name: &str) -> Result<CudaFunction> {
        unsafe {
            let name_cstring = CString::new(name).unwrap();
            let mut func = ptr::null_mut();

            let result = cuModuleGetFunction(&mut func, self.module, name_cstring.as_ptr());
            if result != CUDA_SUCCESS {
                return Err(CompilerError::CudaError(format!(
                    "Failed to get function '{}': {}",
                    name, result
                )));
            }

            Ok(CudaFunction { func })
        }
    }
}

/// CUDA kernel function
pub struct CudaFunction {
    pub func: *mut c_void,
}

impl CudaFunction {
    /// Launch the kernel
    pub fn launch(
        &self,
        grid_dim: (u32, u32, u32),
        block_dim: (u32, u32, u32),
        shared_mem: u32,
        params: &[*mut c_void],
    ) -> Result<()> {
        unsafe {
            // cuLaunchKernel expects params as void** (array of pointers)
            let result = cuLaunchKernel(
                self.func,
                grid_dim.0 as c_int,
                grid_dim.1 as c_int,
                grid_dim.2 as c_int,
                block_dim.0 as c_int,
                block_dim.1 as c_int,
                block_dim.2 as c_int,
                shared_mem as c_int,
                ptr::null_mut(), // default stream
                params.as_ptr() as *mut *mut c_void,
                ptr::null_mut(),
            );

            if result != CUDA_SUCCESS {
                return Err(CompilerError::CudaError(format!(
                    "Failed to launch kernel: {}",
                    result
                )));
            }

            // Synchronize
            let result = cuCtxSynchronize();
            if result != CUDA_SUCCESS {
                return Err(CompilerError::CudaError(format!(
                    "Failed to synchronize: {}",
                    result
                )));
            }

            Ok(())
        }
    }
}


#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::*;
    use crate::buffer::DeviceBuffer;

    const SIMPLE_KERNEL: &str = r#"
extern "C" __global__ void test_kernel(float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = idx * 2.0f;
    }
}
"#;

    #[test]
    fn test_compile_and_run() {
        let compiler = CudaRuntimeCompiler::new(0).expect("Failed to create compiler");
        
        // Compile to PTX
        let ptx = compiler
            .compile_to_ptx(SIMPLE_KERNEL, "test_kernel", "compute_89")
            .expect("Failed to compile");
        
        println!("Generated PTX:\n{}", ptx);
        
        // Load module
        let module = compiler.load_ptx_module(&ptx).expect("Failed to load PTX");
        
        // Get function
        let func = module.get_function("test_kernel").expect("Failed to get function");
        
        // Allocate memory
        let n = 256;
        let mut buffer = DeviceBuffer::new(n * std::mem::size_of::<f32>())
            .expect("Failed to allocate device memory");
        
        // Launch kernel
        let n_i32 = n as i32;
        let params = vec![
            buffer.as_ptr() as *mut c_void,
            &n_i32 as *const _ as *mut c_void,
        ];
        
        func.launch((1, 1, 1), (256, 1, 1), 0, &params)
            .expect("Failed to launch kernel");
        
        // Read results
        let mut results = vec![0f32; n];
        buffer
            .copy_to_host(unsafe {
                std::slice::from_raw_parts_mut(
                    results.as_mut_ptr() as *mut u8,
                    n * std::mem::size_of::<f32>(),
                )
            })
            .expect("Failed to copy results");
        
        // Verify
        for i in 0..n {
            assert_eq!(results[i], (i * 2) as f32);
        }
        
        println!("Test passed!");
    }
}
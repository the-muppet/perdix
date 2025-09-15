/// Test CUDA Driver API parameter passing
use perdix::cuda_runtime_compiler::{CudaRuntimeCompiler, PinnedBuffer};
use std::ffi::c_void;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Testing CUDA Driver API parameter passing...\n");

    // Simple kernel source
    let kernel_source = r#"
extern "C" __global__ void simple_test(int* data, int value) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0) {
        data[0] = value;
    }
}
"#;

    // Initialize compiler
    let compiler = CudaRuntimeCompiler::new(0)?;

    // Compile to PTX
    println!("Compiling kernel...");
    let ptx = compiler.compile_to_ptx(kernel_source, "test", "compute_89")?;

    // Load module
    let module = compiler.load_ptx_module(&ptx)?;
    let kernel = module.get_function("simple_test")?;

    // Allocate pinned memory
    let mut buffer = PinnedBuffer::new(4)?; // 4 bytes for int

    // Initialize to 0
    unsafe {
        *(buffer.as_mut_ptr() as *mut i32) = 0;
    }

    // Test value to write
    let test_value: i32 = 42;

    // Prepare parameters - this is the critical part
    let data_ptr = buffer.as_mut_ptr();

    // Method 1: Direct pointers (what we're currently doing)
    println!("\nMethod 1: Direct pointers");
    let params1 = vec![
        &data_ptr as *const _ as *mut c_void,
        &test_value as *const _ as *mut c_void,
    ];

    // Launch kernel
    match kernel.launch((1, 1, 1), (1, 1, 1), 0, &params1) {
        Ok(_) => {
            let result = unsafe { *(buffer.as_ptr() as *const i32) };
            println!("Success! Result: {}", result);
        }
        Err(e) => {
            println!("Failed: {:?}", e);

            // Try Method 2: Using raw pointers directly
            println!("\nMethod 2: Raw value for pointer, address for value");

            // Reset buffer
            unsafe {
                *(buffer.as_mut_ptr() as *mut i32) = 0;
            }

            // For Driver API, pointer params are passed by value,
            // scalar params are passed by address
            let mut params2: Vec<*mut c_void> = vec![
                data_ptr as *mut c_void,                // Pass pointer by value
                &test_value as *const _ as *mut c_void, // Pass scalar by address
            ];

            // Try with raw array
            unsafe {
                let result = perdix::cuda_runtime_compiler::cuLaunchKernel(
                    kernel.func,
                    1,
                    1,
                    1, // grid
                    1,
                    1,
                    1,                    // block
                    0,                    // shared mem
                    std::ptr::null_mut(), // stream
                    params2.as_mut_ptr(),
                    std::ptr::null_mut(),
                );

                if result == 0 {
                    perdix::cuda_runtime_compiler::cuCtxSynchronize();
                    let value = *(buffer.as_ptr() as *const i32);
                    println!("Method 2 Success! Result: {}", value);
                } else {
                    println!("Method 2 Failed: error {}", result);
                }
            }
        }
    }

    Ok(())
}

// Re-export for test
pub use perdix::cuda_runtime_compiler;

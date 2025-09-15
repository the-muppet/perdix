use perdixlib::runtime::{CudaFunction, CudaModule, CudaRuntimeCompiler};

fn main() {
    println!("Testing PTX Runtime Compilation");
    println!("================================");

    // Simple CUDA kernel source
    let kernel_source = r#"
extern "C" __global__ void test_kernel(int* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = idx * 2;
    }
}
"#;

    // Try to create runtime compiler
    match CudaRuntimeCompiler::new(0) {
        Ok(compiler) => {
            println!("✓ CUDA runtime compiler initialized");

            // Try to compile to PTX
            match compiler.compile_to_ptx(kernel_source, "test_kernel", "compute_89") {
                Ok(ptx) => {
                    println!("✓ Kernel compiled to PTX successfully");
                    println!("PTX size: {} bytes", ptx.len());

                    // Try to load the PTX module
                    match compiler.load_ptx_module(&ptx) {
                        Ok(module) => {
                            println!("✓ PTX module loaded successfully");

                            // Try to get the kernel function
                            match module.get_function("test_kernel") {
                                Ok(func) => {
                                    println!("✓ Kernel function retrieved successfully");
                                    println!("\n=== PTX Runtime Compilation SUCCESSFUL ===");
                                    println!("Your CUDA installation is working correctly!");
                                }
                                Err(e) => {
                                    println!("✗ Failed to get kernel function: {}", e);
                                }
                            }
                        }
                        Err(e) => {
                            println!("✗ Failed to load PTX module: {}", e);
                        }
                    }
                }
                Err(e) => {
                    println!("✗ Failed to compile to PTX: {}", e);
                    println!("\nThis might mean:");
                    println!("- NVRTC library is not properly linked");
                    println!("- CUDA runtime is not available");
                }
            }
        }
        Err(e) => {
            println!("✗ Failed to initialize CUDA runtime compiler: {}", e);
            println!("\nThis might mean:");
            println!("- CUDA driver is not installed");
            println!("- No NVIDIA GPU is available");
            println!("- CUDA libraries are not properly linked");
        }
    }
}

use std::env;
use std::path::PathBuf;

fn main() {
    // Only build CUDA if the feature is enabled
    if env::var("CARGO_FEATURE_CUDA").is_ok() {
        println!("cargo:rerun-if-changed=cuda/");

        // Get CUDA path
        let cuda_path = env::var("CUDA_PATH")
            .or_else(|_| env::var("CUDA_HOME"))
            .unwrap_or_else(|_| String::from("/usr/local/cuda"));

        println!("cargo:info=Using CUDA at: {}", cuda_path);

        // Build CUDA kernels
        let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

        // Compile text processing kernel
        cc::Build::new()
            .cuda(true)
            .flag("-arch=sm_70")
            .flag("-std=c++14")
            .flag("--use_fast_math")
            .flag("-O3")
            .flag("--ptxas-options=-v")
            .include(format!("{}/include", cuda_path))
            .file("cuda/text_kernel.cu")
            .file("cuda/persistent.cu")
            .compile("terminal_cuda");

        // Link CUDA libraries
        println!("cargo:rustc-link-search=native={}/lib64", cuda_path);
        println!("cargo:rustc-link-search=native={}/lib/x64", cuda_path);
        println!("cargo:rustc-link-lib=cudart");
        println!("cargo:rustc-link-lib=cuda");
    }
}
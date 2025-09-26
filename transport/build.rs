use std::env;
use std::path::PathBuf;
use std::process::Command;
use std::fs;

fn create_stub_library(out_dir: &str, silent: bool) {
    // Create stub C file with all required functions
    let stub_content = r#"
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#ifdef _WIN32
#include <malloc.h>
#endif

// Stub implementations for CUDA functions
int cuda_init_device(int device_id) {
    return -1; // CUDA not available
}

int init_unified_buffer(void** slots, void** hdr, int n_slots) {
    // Allocate on heap instead of CUDA unified memory
    size_t header_size = 256;  // Header is 256 bytes
    size_t slot_size = 256;    // Each slot is 256 bytes
    size_t slots_size = n_slots * slot_size;

    // Use aligned_alloc for proper cache-line alignment (64 bytes)
    #ifdef _WIN32
        *hdr = _aligned_malloc(header_size, 64);
        *slots = _aligned_malloc(slots_size, 64);
    #else
        *hdr = aligned_alloc(64, header_size);
        *slots = aligned_alloc(64, slots_size);
    #endif

    if (!*hdr || !*slots) {
        return -1;
    }

    // Zero initialize
    memset(*hdr, 0, header_size);
    memset(*slots, 0, slots_size);

    // Initialize header with basic values
    uint64_t* n_slots_ptr = (uint64_t*)((char*)*hdr + 128); // config.n_slots offset
    *n_slots_ptr = n_slots;

    uint64_t* slot_mask = (uint64_t*)((char*)*hdr + 136); // config.slot_mask offset
    *slot_mask = n_slots - 1;

    return 0;
}

int launch_transport_kernel(void* slots, void* hdr, const void* contexts,
                           uint32_t n_messages, int enable_metrics, void* stream) {
    return -1; // CUDA not available
}

int launch_packed_kernel(void* slots, void* hdr, const void* packed_contexts,
                        const void* text_arena, uint32_t n_messages,
                        int enable_metrics, void* stream) {
    return -1; // CUDA not available
}

// Legacy compatibility names
int launch_unified_kernel(void* slots, void* hdr, const void* contexts,
                         uint32_t n_messages, int enable_metrics, void* stream) {
    return -1; // CUDA not available
}

int launch_unified_kernel_async(void* slots, void* hdr, const void* packed_contexts,
                               const void* text_arena, uint32_t n_messages,
                               int enable_metrics, void* stream) {
    return -1; // CUDA not available
}

int launch_simple_test(void* slots, void* hdr, int n_msgs) {
    return -1; // CUDA not available
}

int cleanup_unified_buffer(void* slots, void* hdr) {
    #ifdef _WIN32
        _aligned_free(slots);
        _aligned_free(hdr);
    #else
        free(slots);
        free(hdr);
    #endif
    return 0;
}

int cuda_reset_device() {
    return 0; // No-op when CUDA not available
}
"#;

    let stub_file = format!("{}/stub.c", out_dir);
    fs::write(&stub_file, stub_content).expect("Failed to write stub file");

    // Compile stub to static library
    cc::Build::new()
        .file(&stub_file)
        .compile("perdix");

    if !silent {
        println!("cargo:warning=CUDA compilation skipped, using stub implementation");
    }
}

fn compile_cuda_kernels(cuda_path: &str, out_dir: &str) -> Result<(), String> {
    // List of kernel files to compile
    let kernel_files = vec![
        "cuda/init_kernel.cu",
        "cuda/transport_kernel.cu",
        "cuda/test_kernel.cu",
        "cuda/packed_kernel.cu",
    ];

    let mut obj_files = Vec::new();

    // Compile each kernel to object file
    for kernel_file in &kernel_files {
        println!("cargo:rerun-if-changed={}", kernel_file);

        let kernel_name = kernel_file.split('/').last().unwrap()
            .split('.').next().unwrap();
        let obj_path = PathBuf::from(&out_dir).join(format!("{}.obj", kernel_name));

        println!("Compiling CUDA kernel: {}", kernel_file);

        let mut nvcc = Command::new(format!("{}/bin/nvcc", cuda_path));
        nvcc.arg("-c")
            .arg(kernel_file)
            .arg("-o")
            .arg(&obj_path)
            .arg("-I")
            .arg("cuda"); // Include directory for common.cuh

        nvcc.arg("--compiler-options").arg("/MD");

        nvcc.arg("-arch=sm_75")  // Minimum Turing architecture (RTX 20 series)
            .arg("-O3")
            .arg("-use_fast_math")
            .arg("-lineinfo");

        nvcc.arg("-Xcompiler").arg("/wd4819");  // Suppress warning about non-ASCII characters

        let output = nvcc.output()
            .map_err(|e| format!("Failed to execute nvcc for {}: {}", kernel_file, e))?;

        if !output.status.success() {
            println!("cargo:warning=nvcc stdout: {}", String::from_utf8_lossy(&output.stdout));
            println!("cargo:warning=nvcc stderr: {}", String::from_utf8_lossy(&output.stderr));
            return Err(format!("CUDA compilation failed for {}", kernel_file));
        }

        obj_files.push(obj_path);
    }

    // Create static library from all object files
    let lib_path = PathBuf::from(&out_dir).join("perdix.lib");

    // Build command with all object files
    let mut lib_cmd = Command::new("lib.exe");
    lib_cmd.arg(format!("/OUT:{}", lib_path.to_str().unwrap()));
    for obj_file in &obj_files {
        lib_cmd.arg(obj_file.to_str().unwrap());
    }

    let lib_result = lib_cmd.output();

    if lib_result.is_err() {
        // Try ar as fallback
        println!("cargo:warning=lib.exe not found, trying ar");
        let mut ar_cmd = Command::new("ar");
        ar_cmd.arg("rcs").arg(&lib_path);
        for obj_file in &obj_files {
            ar_cmd.arg(obj_file);
        }

        ar_cmd.output()
            .map_err(|e| format!("Failed to create static library: {}", e))?;
    }

    Ok(())
}

fn main() {
    // Rerun if any kernel files change
    println!("cargo:rerun-if-changed=cuda/common.cuh");
    println!("cargo:rerun-if-changed=cuda/init_kernel.cu");
    println!("cargo:rerun-if-changed=cuda/transport_kernel.cu");
    println!("cargo:rerun-if-changed=cuda/test_kernel.cu");
    println!("cargo:rerun-if-changed=cuda/packed_kernel.cu");
    println!("cargo:rerun-if-changed=build.rs");

    let out_dir = env::var("OUT_DIR").unwrap();

    // Check if CUDA feature is enabled
    if env::var("CARGO_FEATURE_CUDA").is_ok() {
        // CUDA feature enabled - provide library paths
        let cuda_path = env::var("CUDA_PATH")
            .unwrap_or_else(|_| {
                "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9".to_string()
            });

        let cuda_lib = format!("{}/lib/x64", cuda_path);
        println!("cargo:rustc-link-search=native={}", cuda_lib);
        println!("cargo:rustc-link-lib=nvrtc");
        println!("cargo:rustc-link-lib=cuda");
        println!("cargo:rustc-link-lib=cudart");
    }

    // Skip CUDA compilation if not building with cuda feature
    if env::var("CARGO_FEATURE_CUDA").is_err() {
        // Silently create stub library when CUDA feature is not enabled
        create_stub_library(&out_dir, true);
        return;
    }

    let cuda_path = env::var("CUDA_PATH")
        .unwrap_or_else(|_| {
            "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9".to_string()
        });

    // Set up Visual Studio environment for NVCC
    let vs_paths = vec![
        "C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.44.35207/bin/Hostx64/x64",
        "C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.44.35207/bin/Hostx86/x64",
    ];

    let mut found_vs = false;
    for path in vs_paths {
        if std::path::Path::new(path).exists() {
            let current_path = env::var("PATH").unwrap_or_default();
            env::set_var("PATH", format!("{};{}", path, current_path));
            found_vs = true;
            println!("cargo:warning=Found Visual Studio at: {}", path);
            break;
        }
    }

    if !found_vs {
        println!("cargo:warning=Visual Studio not found, attempting compilation anyway");
    }

    // Compile all CUDA kernels
    match compile_cuda_kernels(&cuda_path, &out_dir) {
        Ok(()) => {
            // Link the static library
            println!("cargo:rustc-link-search=native={}", out_dir);
            println!("cargo:rustc-link-lib=static=perdix");
            println!("CUDA kernel compilation successful!");
        }
        Err(e) => {
            println!("cargo:warning=CUDA compilation failed: {}", e);
            println!("cargo:warning=Using stub implementation");
            create_stub_library(&out_dir, false);
        }
    }
}
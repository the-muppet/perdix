use std::env;
use std::path::PathBuf;
use std::process::Command;

// Create a stub library with minimal implementations
fn create_stub_library(out_dir: &str) {
    use std::fs::File;
    use std::io::Write;
    
    let stub_c = PathBuf::from(out_dir).join("stub.c");
    {
        let mut file = File::create(&stub_c).unwrap();
        
        writeln!(file, r#"
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#ifdef _WIN32
#include <malloc.h>
#endif

// Stub implementations for CUDA functions
int cuda_init_device(int device_id) {{
    printf("WARNING: Using stub CUDA implementation (no GPU support)\n");
    return 0;
}}

int init_unified_buffer(void** slots, void** hdr, int n_slots) {{
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
    
    if (!*hdr || !*slots) {{
        return -1;
    }}
    
    // Zero initialize
    memset(*hdr, 0, header_size);
    memset(*slots, 0, slots_size);
    
    // Initialize header with basic values
    uint64_t* wrap_mask = (uint64_t*)((char*)*hdr + 128); // config.wrap_mask offset
    *wrap_mask = n_slots - 1;
    
    uint32_t* slot_count = (uint32_t*)((char*)*hdr + 136); // config.slot_count offset
    *slot_count = n_slots;
    
    return 0;
}}

int launch_unified_kernel(void* slots, void* hdr, const void* contexts, 
                         uint32_t n_messages, int enable_metrics, uint64_t stream) {{
    printf("WARNING: CUDA kernel launch attempted but not available (stub)\n");
    return -1;
}}

int launch_simple_test(void* slots, void* hdr, int n_msgs) {{
    printf("WARNING: CUDA test kernel not available (stub)\n");
    return -1;
}}

int cleanup_unified_buffer(void* slots, void* hdr) {{
    #ifdef _WIN32
        _aligned_free(slots);
        _aligned_free(hdr);
    #else
        free(slots);
        free(hdr);
    #endif
    return 0;
}}
"#).unwrap();
    } // File is closed here
    
    // Compile stub to object file
    let _obj_path = if cfg!(target_os = "windows") {
        PathBuf::from(out_dir).join("stub.obj")
    } else {
        PathBuf::from(out_dir).join("stub.o")
    };
    
    let mut cc = cc::Build::new();
    cc.file(&stub_c)
      .compile("perdix");
    
    println!("cargo:rustc-link-search=native={}", out_dir);
}

fn main() {
    println!("cargo:rerun-if-changed=cuda/perdix_kernel.cu");
    println!("cargo:rerun-if-changed=build.rs");
    
    // Skip CUDA compilation if not building with cuda feature
    if env::var("CARGO_FEATURE_CUDA").is_err() {
    println!("cargo:warning=CARGO_FEATURE_CUDA env var: {:?}", env::var("CARGO_FEATURE_CUDA"));
        println!("Skipping CUDA compilation (cuda feature not enabled)");
        println!("Creating stub implementation...");
        let out_dir = env::var("OUT_DIR").unwrap();
        create_stub_library(&out_dir);
        return;
    }
    
    let out_dir = env::var("OUT_DIR").unwrap();
    let cuda_path = env::var("CUDA_PATH")
        .unwrap_or_else(|_| {
            if cfg!(target_os = "windows") {
                "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9".to_string()
            } else {
                "/usr/local/cuda".to_string()
            }
        });
    
    // Try to find Visual Studio on Windows
    if cfg!(target_os = "windows") {
        // Add Visual Studio paths to environment
        let vs_paths = vec![
            "C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.44.35207/bin/Hostx64/x64",
            "C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.39.33519/bin/Hostx64/x64",
            "C:/Program Files/Microsoft Visual Studio/2022/Professional/VC/Tools/MSVC/14.39.33519/bin/Hostx64/x64",
            "C:/Program Files/Microsoft Visual Studio/2022/Enterprise/VC/Tools/MSVC/14.39.33519/bin/Hostx64/x64",
            "C:/Program Files (x86)/Microsoft Visual Studio/2019/Community/VC/Tools/MSVC/14.29.30133/bin/Hostx64/x64",
        ];
        
        let mut found_vs = false;
        for path in vs_paths {
            if std::path::Path::new(path).exists() {
                let current_path = env::var("PATH").unwrap_or_default();
                env::set_var("PATH", format!("{};{}", path, current_path));
                found_vs = true;
                println!("Found Visual Studio at: {}", path);
                break;
            }
        }
        
        if !found_vs {
            println!("cargo:warning=Visual Studio not found, attempting compilation anyway");
        }
    }
    
    // Compile CUDA kernel to object file
    let kernel_path = "cuda/perdix_kernel.cu";
    let obj_path = if cfg!(target_os = "windows") {
        PathBuf::from(&out_dir).join("perdix_kernel.obj")
    } else {
        PathBuf::from(&out_dir).join("perdix_kernel.o")
    };
    
    println!("Compiling CUDA kernel: {}", kernel_path);
    
    let mut nvcc = Command::new(format!("{}/bin/nvcc", cuda_path));
    nvcc.arg("-c")
        .arg(kernel_path)
        .arg("-o")
        .arg(&obj_path);
    
    if cfg!(target_os = "windows") {
        nvcc.arg("--compiler-options").arg("/MD");
        // Explicitly tell NVCC where to find cl.exe
        nvcc.arg("-ccbin").arg("C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.44.35207/bin/Hostx64/x64");
    }
    
    nvcc.arg("-arch=sm_89")  // Minimum Volta architecture
        .arg("-O3")
        .arg("-use_fast_math")
        .arg("-lineinfo");
    
    if cfg!(target_os = "windows") {
        nvcc.arg("-Xcompiler").arg("/wd4819");  // Suppress warning about non-ASCII characters
    }
    
    let output = nvcc.output();
    
    match output {
        Ok(output) => {
            if !output.status.success() {
                eprintln!("nvcc stdout: {}", String::from_utf8_lossy(&output.stdout));
                eprintln!("nvcc stderr: {}", String::from_utf8_lossy(&output.stderr));
                
                // Fall back to stub implementation
                println!("cargo:warning=CUDA compilation failed, using stub implementation");
                create_stub_library(&out_dir);
                return;
            }
        }
        Err(e) => {
            println!("cargo:warning=Failed to execute nvcc: {}", e);
            println!("cargo:warning=Using stub implementation");
            create_stub_library(&out_dir);
            return;
        }
    }
    
    // Create static library from object file
    let lib_path = if cfg!(target_os = "windows") {
        PathBuf::from(&out_dir).join("perdix.lib")
    } else {
        PathBuf::from(&out_dir).join("libperdix.a")
    };
    
    if cfg!(target_os = "windows") {
        // Try to use lib.exe on Windows
        let lib_result = Command::new("lib.exe")
            .arg(format!("/OUT:{}", lib_path.to_str().unwrap()))
            .arg(obj_path.to_str().unwrap())
            .output();
        
        if lib_result.is_err() {
            // Fall back to using ar if lib.exe not found
            println!("cargo:warning=lib.exe not found, trying ar");
            let ar_result = Command::new("ar")
                .arg("rcs")
                .arg(&lib_path)
                .arg(&obj_path)
                .output();
            
            if ar_result.is_err() {
                println!("cargo:warning=Failed to create static library, using stub");
                create_stub_library(&out_dir);
                return;
            }
        }
    } else {
        // Use ar on Unix
        let output = Command::new("ar")
            .arg("rcs")
            .arg(&lib_path)
            .arg(&obj_path)
            .output()
            .expect("Failed to execute ar");
        
        if !output.status.success() {
            eprintln!("ar stderr: {}", String::from_utf8_lossy(&output.stderr));
            panic!("Failed to create static library");
        }
    }
    
    // Link the static library
    println!("cargo:rustc-link-search=native={}", out_dir);
    println!("cargo:rustc-link-lib=static=perdix");
    
    // Link CUDA libraries
    if cfg!(target_os = "windows") {
        let cuda_lib = format!("{}/lib/x64", cuda_path);
        println!("cargo:rustc-link-search=native={}", cuda_lib);
        println!("cargo:rustc-link-lib=cudart");
        println!("cargo:rustc-link-lib=cuda");
    } else {
        println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
        println!("cargo:rustc-link-lib=cudart");
        println!("cargo:rustc-link-lib=cuda");
    }
    
    println!("CUDA kernel compilation successful!");
}
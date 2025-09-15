//! # Runtime Module - Dynamic CUDA Kernel Compilation
//! 
//! This module provides runtime compilation of CUDA kernels using NVRTC (NVIDIA Runtime Compilation).
//! This approach bypasses traditional nvcc/MSVC toolchain conflicts and enables dynamic kernel
//! optimization based on runtime parameters.
//! 
//! ## Key Components
//! 
//! - **NVRTC Compilation**: Compiles CUDA C++ source to PTX at runtime
//! - **Dynamic Loading**: Loads compiled PTX modules into CUDA context
//! - **Parameter Binding**: Safe parameter passing to GPU kernels
//! - **Error Handling**: Comprehensive error reporting with line numbers
//! 
//! ## Architecture
//! 
//! ```text
//! ┌─────────────────┐
//! │  CUDA C++ Source│
//! └────────┬────────┘
//!          │
//!          ▼
//! ┌─────────────────┐
//! │  NVRTC Compiler │  Runtime compilation
//! └────────┬────────┘
//!          │
//!          ▼
//! ┌─────────────────┐
//! │   PTX Assembly  │
//! └────────┬────────┘
//!          │
//!          ▼
//! ┌─────────────────┐
//! │  CUDA Module    │  Loaded into GPU
//! └────────┬────────┘
//!          │
//!          ▼
//! ┌─────────────────┐
//! │ Kernel Function │  Ready for launch
//! └─────────────────┘
//! ```
//! 
//! ## Benefits
//! 
//! 1. **Toolchain Independence**: No dependency on nvcc or Visual Studio
//! 2. **Runtime Optimization**: Kernels optimized for specific GPU architecture
//! 3. **Dynamic Specialization**: Template parameters resolved at runtime
//! 4. **Hot Reload**: Kernels can be recompiled without restarting
//! 
//! ## Example
//! 
//! ```rust,no_run
//! use perdix::runtime::{CudaRuntimeCompiler, get_kernel_source};
//! 
//! // Get kernel source with configuration
//! let kernel_info = get_kernel_source(256, 32, true);
//! 
//! // Compile to PTX
//! let mut compiler = CudaRuntimeCompiler::new();
//! let ptx = compiler.compile(&kernel_info.source, &kernel_info.name)?;
//! 
//! // Load module and get function
//! let module = compiler.load_ptx(&ptx)?;
//! let function = module.get_function("produce_messages")?;
//! 
//! // Launch kernel
//! function.launch(grid_size, block_size, shared_mem, stream)?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

pub mod cuda_params;
pub mod jit;
pub mod kernel_source;
pub mod runtime_compiler;

pub use cuda_params::CudaParams;
pub use jit::JITRuntime;
pub use kernel_source::{get_kernel_source, KernelInfo};
pub use runtime_compiler::{CompilerError, CudaFunction, CudaModule, CudaRuntimeCompiler};

/// High-level runtime system for Perdix GPU operations.
/// 
/// Combines all runtime components into a unified interface for kernel
/// compilation, loading, and execution.
/// 
/// # Components
/// 
/// - `params`: CUDA kernel parameter management
/// - `runtime`: JIT compilation runtime
/// - `compiler`: NVRTC compiler interface
/// - `kernel_info`: Compiled kernel metadata
/// - `error`: Last compilation error (if any)
/// 
/// # Usage
/// 
/// This struct is typically used internally by the GPU producer.
/// Direct usage is for advanced scenarios requiring custom kernel compilation.
pub struct PerdixRuntime {
    pub params: CudaParams,
    pub runtime: JITRuntime,
    pub compiler: CudaRuntimeCompiler,
    pub kernel_info: KernelInfo,
    pub error: CompilerError,
}

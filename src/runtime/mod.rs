pub mod cuda_params;
pub mod jit;
pub mod kernel_source;
pub mod runtime_compiler;

pub use cuda_params::CudaParams;
pub use jit::JITRuntime;
pub use kernel_source::{get_kernel_source, KernelInfo};
pub use runtime_compiler::{CompilerError, CudaFunction, CudaModule, CudaRuntimeCompiler};

pub struct PerdixRuntime {
    pub params: CudaParams,
    pub runtime: JITRuntime,
    pub compiler: CudaRuntimeCompiler,
    pub kernel_info: KernelInfo,
    pub error: CompilerError,
}

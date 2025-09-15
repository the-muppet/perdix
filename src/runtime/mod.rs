pub mod runtime_compiler;
pub mod kernel_source;
pub mod jit;
pub mod cuda_params;

pub use runtime_compiler::{CompilerError, CudaModule, CudaFunction, CudaRuntimeCompiler};
pub use kernel_source::{KernelInfo, get_kernel_source};
pub use cuda_params::CudaParams;
pub use jit::JITRuntime;

pub struct PerdixRuntime {
    pub params: CudaParams,
    pub runtime: JITRuntime,
    pub compiler: CudaRuntimeCompiler,
    pub kernel_info: KernelInfo,
    pub error: CompilerError
}
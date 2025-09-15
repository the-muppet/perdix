pub mod buffer;
pub mod runtime;
pub mod pty;

#[cfg(feature = "cuda")]
pub mod gpu;

#[cfg(not(feature = "cuda"))]
pub mod cpu;

// Re-export key types
pub use buffer::ffi::{AgentType, StreamContext};
pub use buffer::{Buffer, Consumer, Header, Producer, Slot};

#[cfg(feature = "cuda")]
pub use gpu::GpuProducer;

pub use runtime::{CudaFunction, CudaModule, CudaRuntimeCompiler, PerdixRuntime};

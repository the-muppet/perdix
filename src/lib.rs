pub mod buffer;
pub mod runtime;

#[cfg(feature = "pty")]
pub mod pty;

#[cfg(feature = "cuda")]
pub mod gpu;

#[cfg(not(feature = "cuda"))]
pub mod cpu;

// Re-export key types
pub use buffer::{Buffer, Consumer, Producer, Header, Slot};
pub use buffer::ffi::{AgentType, StreamContext};

#[cfg(feature = "cuda")]
pub use gpu::GpuProducer;

pub use runtime::{CudaRuntimeCompiler, CudaModule, CudaFunction, PerdixRuntime};
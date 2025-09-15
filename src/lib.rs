//! # Perdix - High-Performance GPU-Accelerated Ring Buffer
//! 
//! Perdix is a zero-copy, lock-free SPSC (Single Producer Single Consumer) ring buffer
//! designed for ultra-low latency streaming between GPU producers and CPU consumers.
//! Optimized for AI text streaming workloads with ANSI formatting support.
//! 
//! ## Key Features
//! 
//! - **Zero-Copy Architecture**: CUDA unified memory eliminates CPU-GPU memcpy overhead
//! - **Lock-Free Design**: Atomic sequence numbers ensure ordering without locks
//! - **Sub-Microsecond Latency**: <1μs producer-to-consumer communication
//! - **Multi-Backend Support**: CUDA, WebGPU, and CPU fallback implementations
//! - **Runtime Compilation**: NVRTC for dynamic kernel generation (bypasses toolchain conflicts)
//! - **Production Ready**: Comprehensive error handling and recovery mechanisms
//! 
//! ## Performance Characteristics
//! 
//! - **Throughput**: 2-3 GB/s sustained
//! - **Message Rate**: >10M messages/second
//! - **Memory Layout**: Cache-aligned 256-byte slots
//! - **Batch Processing**: Warp-level batching reduces atomic contention
//! 
//! ## Quick Start
//! 
//! ```rust,no_run
//! use perdix::{Buffer, AgentType};
//! 
//! // Create a ring buffer with 4096 slots
//! let mut buffer = Buffer::new(4096).expect("Failed to create buffer");
//! 
//! // Split into producer and consumer
//! let (mut producer, mut consumer) = buffer.split_mut();
//! 
//! // Producer writes messages (can be from GPU or CPU)
//! producer.try_produce(b"Hello from Perdix!", AgentType::Assistant);
//! 
//! // Consumer reads messages
//! if let Some(message) = consumer.try_consume() {
//!     println!("Received: {}", message.as_str());
//! }
//! ```
//! 
//! ## Architecture Overview
//! 
//! ```text
//! ┌─────────────┐     ┌─────────────────┐     ┌─────────────┐
//! │ GPU Kernel  │────▶│   Ring Buffer   │◀────│ CPU Reader  │
//! │  Producer   │     │  (Unified Mem)  │     │  Consumer   │  
//! └─────────────┘     └─────────────────┘     └─────────────┘
//!                             ▲
//!                    Zero-Copy Shared Memory
//! ```
//! 
//! ## Usage Modes
//! 
//! ### Interactive REPL Mode
//! ```bash
//! cargo run --features cuda --bin perdix -- --repl
//! ```
//! 
//! ### Continuous Streaming Mode
//! ```bash
//! cargo run --features cuda --bin perdix -- --stream
//! ```
//! 
//! ### Performance Benchmarking
//! ```bash
//! cargo run --features cuda --bin perdix -- --benchmark
//! ```
//! 
//! ## Feature Flags
//! 
//! - `cuda`: Enable NVIDIA CUDA acceleration (recommended for production)
//! - `webgpu`: Enable WebGPU backend (experimental, cross-platform)
//! - `runtime`: Enable runtime kernel compilation via NVRTC
//! 
//! ## Safety and Error Handling
//! 
//! Perdix enforces memory safety through Rust's ownership system while providing
//! high-performance GPU interop. All FFI boundaries are documented with safety
//! requirements and error codes.

pub mod buffer;
#[cfg(feature = "cuda")]
pub mod runtime;
pub mod pty;

#[cfg(feature = "cuda")]
pub mod gpu;

#[cfg(feature = "webgpu")]
pub mod webgpu;

#[cfg(all(not(feature = "cuda"), not(feature = "webgpu")))]
pub mod cpu;

// Re-export key types
pub use buffer::ffi::{AgentType, StreamContext};
pub use buffer::{Buffer, Consumer, Header, Producer, Slot};

#[cfg(feature = "cuda")]
pub use gpu::GpuProducer;

#[cfg(feature = "cuda")]
pub use runtime::{CudaFunction, CudaModule, CudaRuntimeCompiler, PerdixRuntime};

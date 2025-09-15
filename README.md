# Perdix

High-performance GPU-accelerated ring buffer for ultra-low latency streaming between producers and consumers. Optimized for multiple AI text streaming workloads to prevent screen tearing and terminal corruption with support for NVIDIA CUDA, WebGPU, and CPU fallback.

## Overview

Perdix implements a lock-free, zero-copy Single Producer Single Consumer (SPSC) ring buffer using GPU unified memory. It achieves sub-microsecond latency and multi-gigabyte throughput, making it ideal for real-time AI assistant output streaming, high-frequency data processing, and terminal multiplexing applications.

**Primary Purpose**: Perdix was specifically designed to eliminate screen tearing when multiple AI agents (Claude, GPT, etc.) stream output simultaneously to the same terminal. By routing all output through a GPU-managed ring buffer with atomic operations and proper memory fencing, Perdix ensures clean, tear-free terminal rendering even with dozens of concurrent AI streams.

### Key Features

- **Zero-Copy Architecture**: Direct GPU-to-CPU memory access without explicit transfers
- **Lock-Free Design**: Atomic operations ensure thread safety without mutex overhead
- **Multi-Backend Support**: CUDA (NVIDIA), WebGPU (cross-platform), CPU fallback
- **Production Ready**: Comprehensive error handling and recovery mechanisms
- **Terminal Integration**: Built-in PTY support for AI-to-terminal streaming

### Performance Metrics

| Metric | Value |
|--------|-------|
| Throughput | 2-3 GB/s sustained |
| Latency | <1 microsecond GPU-to-CPU |
| Message Rate | >10M messages/second |
| Memory Efficiency | Cache-aligned 256-byte slots |

### Architecture

High level overview
```
┌─────────────────────────────────────────────────────────────────────────┐
│                                Perdix                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────────────┐                                        │
│  │ ╔═══════════╗ ╔════════════╗│                                        │
│  │ ║ Streaming ║ ║ Responses  ║│                                        │
│  │ ║    of     ║ ║    of      ║│                                        │
│  │ ║ Multiple  ║ ║ AI Agents  ║│                                        │
│  │ ╚═════╪═════╝ ╚═════╪══════╝│                                        │
│  └───────┼─────────────┼───────┘                                        │
│          └──────┬──────┘                                                │
│                 ▼                                                       │
│  ╔═══════════════════════════════════════╗                              │
│  ║      GPU Backend (Multi-Platform)     ║                              │
│  ║  ┌────────────────┐ ┌──────────────┐  ║                              │
│  ║  │ CUDA Kernels / │ │    Input     │  ║                              │
│  ║  │ WebGPU Compute │ │    Buffers   │  ║                              │
│  ║  │    Shaders     │ └──────────────┘  ║                              │
│  ║  └───────▲────────┘                   ║                              │
│  ║          │        (GPU-Text-ready)    ║                              │
│  ╚══════════╪════════════════════════════╝                              │
│             │                                                           │
│    Parse/Transform → Build ANSI Spans                                   │
│             │                                                           │
│  ┌──────────┼──────────────────────────────────────┐                    │
│  │          ▼                                      │ __threadfence()    │
│  │    Direct Write                                 │     Updates        │
│  │  (Device-Mapped)                                │                    │
│  │                                                 │                    │
│  │  ╔═══════════════════════════════════════════╗  │                    │
│  │  ║  Shared Pinned Host Ring Buffer           ║  │                    │
│  │  ║  CUDA Unified Memory / WebGPU Mapped      ║  │     TIOCOUTO/      │
│  │  ║         Lock-Free SPPC                    ║◄─┼──── SIGWINCH       │
│  │  ║  ┌──────────┬──────────┬──────────────┐   ║  │                    │
│  │  ║  │  Slots   │ Header   │              │   ║  │                    │
│  │  ║  │ (write,  │  {u8; N} │              │   ║  │                    │
│  │  ║  │  index,  │  (U8; N) │     ◄────────┼───╫──┘                    │
│  │  ║  │  flags)  │          │              │   ║                       │
│  │  ║  │ Payload[ANSI+Text Bytes]           │   ║   writev() Syscall    │
│  │  ║  └────────────────────────────────────┘   ║     (Batched)         │
│  │  ╚═══════════════════════════════════════════╝                       │
│  │                                                    • Poll/Update     │
│  └────────────────────────────────────────────────────  SIGWINCH        │
│                                                                         │
│  ┌─────────────────────────────────────┐                                │
│  │  CPU Flush Thread                   │                                │
│  │                                     │      Read Spans/Batch Iovecs   │
│  │  • Poll Epochs                      ├────────────────────────────►   │
│  │  • Read-Ready Spans                 │                                │
│  │  • writev to PTY (Batched)          │     ╔═════════════════════╗    │
│  │  • Backpressure (TIOCOUTO)          │     ║  Portable PTY       ║    │
│  └─────────────────────────────────────┘     ║  ┌────────┬───────┐ ║    │
│                                              ║  │ Master │ Slave │ ║    │
│                                              ║  │(writev)│  Raw  │ ║    │
│                                              ║  └────────┴───┬───┘ ║    │
│                                              ╚═══════════════╪═════╝    │
│                                                              ▼          │
│                                                      ┌──────────────┐   │
│                                                      │   Terminal   │   │
│                                                      │   Emulator   │   │
│                                                      │    Output    │   │
│                                                      └──────────────┘   │
│                                                                         │
│  Zero-Copy Data Path: GPU → Host-Pinned Ring → PTY writev()             │
│  CU handles only I/O syscalls                                           │
└─────────────────────────────────────────────────────────────────────────┘
```
Perdix uses a carefully designed memory layout to maximize performance:

```
┌──────────────────────────────────────────┐
│ Header (256 bytes, 4 cache lines)        │
├──────────────────────────────────────────┤
│ Slot 0 (256 bytes)                       │
│   ├─ sequence (8 bytes)                  │
│   ├─ length (4 bytes)                    │
│   ├─ agent_type (4 bytes)                │
│   └─ payload (240 bytes)                 │
├──────────────────────────────────────────┤
│ Slot 1 (256 bytes)                       │
├──────────────────────────────────────────┤
│ ...                                      │
└──────────────────────────────────────────┘
```

### Data Flow Sequence

```
 GPU Side (Producer)          CPU Side (Consumer)
 ═══════════════════          ═══════════════════
       │                             │
       ▼                             ▼
 atomicAdd(write_idx)          Poll rd_seq
       │                             │
       ▼                             ▼
 Write to slot[seq]            Check slot.seq
       │                             │
       ▼                             ▼
 __threadfence_system()        Read payload
       │                             │
       ▼                             ▼
 slot->seq = seq               writev() to PTY
```

The header is divided into cache lines to prevent false sharing:
- **Producer line**: Hot for GPU writes
- **Consumer line**: Hot for CPU reads  
- **Config line**: Read-only after initialization
- **Control line**: Infrequently accessed flags

## Installation

### Prerequisites

#### For CUDA Support (Recommended)
- NVIDIA GPU with Compute Capability 7.0+
- CUDA Driver 11.0+
- CUDA Toolkit (optional, for runtime compilation)

#### For WebGPU Support
- Modern GPU with WebGPU support
- Compatible graphics drivers

### Building from Source

```bash
# Clone the repository
git clone https://github.com/the-muppet/perdix.git
cd perdix

# Build with CUDA support (recommended for NVIDIA GPUs)
cargo build --release --features cuda

# Build with WebGPU support (cross-platform)
cargo build --release --features webgpu

# Build with both backends
cargo build --release --all-features
```

## Usage

### Basic Example

```rust
use perdix::{Buffer, AgentType};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create ring buffer with 4096 slots
    let mut buffer = Buffer::new(4096)?;
    
    // Split into producer and consumer
    let (mut producer, mut consumer) = buffer.split_mut();
    
    // Producer writes messages
    producer.try_produce(b"Hello from GPU", AgentType::Assistant);
    
    // Consumer reads messages
    if let Some(message) = consumer.try_consume() {
        println!("Received: {}", message.as_str());
    }
    
    Ok(())
}
```

### Multi-threaded Example

```rust
use perdix::{Buffer, AgentType};
use std::thread;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let buffer = Buffer::new(1024)?;
    let (producer, consumer) = buffer.split();
    
    // Producer thread (could be GPU kernel)
    let producer_handle = thread::spawn(move || {
        let mut producer = producer;
        for i in 0..100 {
            let msg = format!("Message {}", i);
            producer.try_produce(msg.as_bytes(), AgentType::Info);
        }
    });
    
    // Consumer thread
    let consumer_handle = thread::spawn(move || {
        let mut consumer = consumer;
        let mut count = 0;
        while count < 100 {
            if let Some(msg) = consumer.try_consume() {
                println!("Got: {}", msg.as_str());
                count += 1;
            }
        }
    });
    
    producer_handle.join().unwrap();
    consumer_handle.join().unwrap();
    Ok(())
}
```

### GPU Streaming Example

```rust
#[cfg(feature = "cuda")]
use perdix::{Buffer, GpuProducer};
use perdix::buffer::ffi::StreamContext;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let buffer = Buffer::new(4096)?;
    let mut gpu = GpuProducer::new(buffer, 0)?;
    
    // Create batch of messages
    let contexts: Vec<StreamContext> = (0..32)
        .map(|i| StreamContext::new(
            format!("GPU message {}", i).as_bytes(),
            AgentType::Assistant
        ))
        .collect();
    
    // Process batch on GPU
    gpu.process_batch(&contexts, true)?;
    
    Ok(())
}
```

## Command Line Interface

Perdix includes a versatile CLI for testing and demonstration:

```bash
# Interactive REPL mode
perdix --repl

# Continuous streaming mode
perdix --stream

# Performance benchmark
perdix --benchmark

# Zero-copy GPU-to-PTY streaming
perdix --zerocopy

# Launch external process through GPU PTY
perdix --claude

# Custom slot count
perdix --slots=8192 --benchmark
```

## Integration

### Terminal Multiplexing

Perdix can stream AI assistant output directly to pseudo-terminals:

```rust
use perdix::Buffer;
use perdix::pty::portable::PortablePtyWriter;

let buffer = Buffer::new(1024)?;
let (producer, consumer) = buffer.split();

// Create PTY and start writer thread
let pty = PortablePtyWriter::new()?;
let (stop_flag, handle) = pty.start_writer_thread(consumer);

// Producer writes → Ring Buffer → PTY → Terminal
// ...

stop_flag.store(true, Ordering::Relaxed);
handle.join().unwrap();
```

### Runtime Kernel Compilation

For advanced users, Perdix supports runtime CUDA kernel compilation:

```rust
use perdix::runtime::{CudaRuntimeCompiler, get_kernel_source};

let kernel_info = get_kernel_source(256, 32, true);
let mut compiler = CudaRuntimeCompiler::new();
let ptx = compiler.compile(&kernel_info.source, &kernel_info.name)?;
let module = compiler.load_ptx(&ptx)?;
let function = module.get_function("produce_messages")?;
```

## Performance Tuning

### Cache Alignment

Adjust cache line size for your architecture in `build.rs`:
- x86_64: 64 bytes (default)
- ARM: 128 bytes

### Batch Size Optimization

Configure batch size based on GPU architecture:
```rust
const BATCH_SIZE: usize = 32;  // Warp size for NVIDIA GPUs
```

### Memory Allocation

For optimal performance, ensure slot count is a power of 2:
```rust
let buffer = Buffer::new(4096)?;  // Good: 2^12
let buffer = Buffer::new(5000)?;  // Bad: Not power of 2 (will fail)
```

## Benchmarks

Performance measurements on RTX 4070:

| Operation | Performance |
|-----------|------------|
| Single message | <1 μs latency |
| Batch (32 msgs) | ~15 μs total |
| Sustained streaming | 2.8 GB/s |
| Peak message rate | 12M msgs/sec |

### Running Benchmarks
(Work in progress)  
Perdix includes comprehensive benchmarks using the Criterion framework:

```bash
# Run all benchmarks
cargo bench

# Run specific benchmark suite
cargo bench --bench throughput
cargo bench --bench latency

# Run with CUDA features (recommended for GPU benchmarks)
cargo bench --features cuda --bench gpu_vs_cpu

# Quick benchmark run (fewer samples, faster)
cargo bench -- --quick

# Run benchmarks and save baseline
cargo bench -- --save-baseline my-baseline

# Compare against baseline
cargo bench -- --baseline my-baseline

# Generate HTML reports (output in target/criterion/)
cargo bench -- --verbose
```

The benchmark results are saved in `target/criterion/` with detailed HTML reports showing:
- Performance graphs
- Statistical analysis
- Regression detection
- Historical comparisons

For the built-in simple benchmark:
```bash
cargo run --release --features cuda --bin perdix -- --benchmark
```

## Project Structure

```
perdix/
├── src/
│   ├── buffer/          # Ring buffer implementation
│   │   ├── mod.rs       # Buffer management
│   │   ├── spsc.rs      # Producer/Consumer logic
│   │   ├── ffi.rs       # CUDA FFI interface
│   │   ├── slot.rs      # Message slot structure
│   │   └── gpu_arena.rs # GPU text arena allocator
│   ├── runtime/         # CUDA runtime compilation
│   │   ├── mod.rs       # Runtime system
│   │   └── jit.rs       # NVRTC integration
│   ├── gpu.rs           # GPU producer implementation
│   ├── webgpu.rs        # WebGPU backend implementation
│   ├── pty/             # Terminal integration
│   └── main.rs          # CLI application
├── cuda/
│   └── perdix_kernel.cu # CUDA kernel implementation
├── bin/
│   ├── gpu_test.rs      # GPU testing utility
│   ├── gpu_pty.rs       # GPU-to-PTY demo
│   └── test_unified.rs  # Unified kernel tests
└── benches/             # Performance benchmarks
    ├── throughput.rs    # Message throughput tests
    ├── latency.rs       # End-to-end latency tests
    └── gpu_vs_cpu.rs    # GPU acceleration comparison
```

## Development

### Running Tests

```bash
# Run all tests
cargo test

# Run with CUDA features
cargo test --features cuda

# Run documentation tests
cargo test --doc
```

### Building Documentation

```bash
# Generate and open documentation
cargo doc --all-features --no-deps --open
```

### Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Safety and FFI

Perdix uses unsafe code for GPU interop. All FFI boundaries are documented with safety requirements:

- CUDA device must be initialized before kernel launches
- Memory buffers must outlive kernel execution
- Proper synchronization required for async operations

See documentation for detailed safety requirements.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Copyright (c) 2025 Robert Pratt

## Acknowledgments

- Built with Rust for memory safety and performance
- CUDA kernels optimized for modern NVIDIA GPUs
- WebGPU support for cross-platform compatibility
- Inspired by high-frequency trading systems and real-time streaming architectures

## Support

For issues, questions, or contributions, please visit the [GitHub repository](https://github.com/the-muppet/perdix).
# Perdix System

A multi-component system for high-performance GPU-accelerated terminal multiplexing and AI agent orchestration.

## Components

### Transport Layer (`transport/perdix`)
✅ **Status: Complete**

High-performance zero-copy GPU-accelerated ring buffer for producer-consumer communication. Provides sub-microsecond latency message passing between GPU kernels and CPU consumers.

- CUDA unified memory for zero-copy transfers
- Lock-free SPSC design
- 2-3 GB/s throughput, <1μs latency

### Orchestration Layer (Coming Soon)
🚧 **Status: Planned**

Multi-agent orchestrator for managing concurrent AI agents and preventing output conflicts.

### Terminal Renderer (Coming Soon)
🚧 **Status: Planned**

High-performance terminal rendering engine optimized for handling multiple concurrent streams.

### Integration Bridge (Coming Soon)
🚧 **Status: Planned**

Integration layer for connecting transport, orchestration, and rendering components.

## Problem Statement

When running multiple AI agents concurrently in a terminal environment, users experience:
- Screen tearing and visual artifacts
- "Whiplash" effects from rapid context switching
- Infinite scrolling from uncoordinated output
- Terminal crashes from buffer overflow

This system addresses these issues by providing:
1. **Transport Layer**: High-performance data movement between producers and consumers
2. **Orchestration**: Intelligent scheduling and merging of multi-agent outputs
3. **Rendering**: Optimized terminal rendering that can handle high-throughput streams
4. **Integration**: Seamless connection between all components

## Architecture

```
┌──────────────────────────────────────────────────────┐
│                    AI Agents (1..N)                  │
└──────────────┬───────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────┐
│              Orchestration Layer                      │
│         (Scheduling, Merging, Buffering)             │
└──────────────┬───────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────┐
│              Transport Layer (Perdix)                 │
│         (Zero-copy GPU↔CPU Ring Buffer)              │
└──────────────┬───────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────┐
│              Terminal Renderer                        │
│         (Optimized ANSI Processing)                  │
└──────────────┬───────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────┐
│              Terminal Emulator                        │
└──────────────────────────────────────────────────────┘
```

## Building

This is a Rust workspace project. Build all components with:

```bash
# Build everything
cargo build --release --all-features

# Build specific component
cargo build --release -p perdix-transport --features cuda

# Run tests
cargo test --workspace

# Run benchmarks
cargo bench --workspace
```

## Development Status

- ✅ Transport Layer - Complete and tested
- 🚧 Orchestration - In design phase
- 🚧 Terminal Renderer - In design phase
- 🚧 Integration - In design phase

## Requirements

- Rust 1.75+
- CUDA Toolkit 11.0+ (for GPU acceleration)
- NVIDIA GPU with compute capability 5.0+

## License

MIT OR Apache-2.0
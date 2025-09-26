# Perdix Transport Layer

A high-performance, zero-copy GPU-accelerated ring buffer transport layer for producer-consumer communication.

## Overview

Perdix Transport provides a lock-free SPSC (Single Producer Single Consumer) ring buffer implementation that leverages CUDA unified memory for zero-copy data transfer between GPU and CPU. This transport layer is designed for ultra-low latency streaming with sub-microsecond round-trip times.

## Features

- **Zero-Copy Architecture**: CUDA unified memory eliminates CPU-GPU memcpy overhead
- **Lock-Free Design**: Atomic sequence numbers ensure ordering without locks
- **Cache-Aligned**: 256-byte slots optimized for cache line efficiency
- **Backpressure Handling**: Automatic flow control prevents buffer overflow
- **Sub-Microsecond Latency**: <1μs producer-to-consumer communication
- **High Throughput**: 2-3 GB/s sustained data transfer rates

## Architecture

```
┌─────────────┐     ┌─────────────────┐     ┌─────────────┐
│  Producer   │────▶│   Ring Buffer   │◀────│  Consumer   │
│  (GPU/CPU)  │     │  (Unified Mem)  │     │    (CPU)    │
└─────────────┘     └─────────────────┘     └─────────────┘
                            ▲
                   Zero-Copy Shared Memory
```

## Usage

```rust
use perdix_transport::Buffer;

// Create a ring buffer with 4096 slots
let mut buffer = Buffer::new(4096)?;

// Split into producer and consumer
let (mut producer, mut consumer) = buffer.split_mut();

// Producer writes messages
producer.try_produce(b"Hello from GPU!")?;

// Consumer reads messages
if let Some(message) = consumer.try_consume() {
    println!("Received: {:?}", message);
}
```

## Performance

- **Throughput**: 2-3 GB/s sustained
- **Message Rate**: >10M messages/second
- **Latency**: <1μs round-trip time
- **Memory**: Zero-copy between GPU and CPU

## Requirements

- NVIDIA GPU with CUDA support (compute capability 5.0+)
- CUDA Toolkit 11.0 or later
- Rust 1.75 or later

## Building

```bash
# With CUDA support
cargo build --release --features cuda

# Run tests
cargo run --release --features cuda --bin transport_test

# Run benchmarks
cargo run --release --features cuda --bin benchmark
```

## Testing

The transport layer includes comprehensive tests:

- **Basic functionality**: Single message round-trip verification
- **Stress testing**: High-throughput sustained load testing
- **Latency measurement**: Round-trip time profiling
- **Backpressure**: Buffer overflow handling
- **Wraparound**: Ring buffer wraparound correctness

Run tests with:
```bash
cargo run --release --features cuda --bin transport_test
```

## License

MIT OR Apache-2.0
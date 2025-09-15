# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Enhanced crate metadata for better discoverability on crates.io
- Comprehensive README with badges, examples, and performance metrics
- CHANGELOG.md for version tracking
- Examples directory with practical usage demonstrations

## [0.1.1] - 2025-01-15

### Added
- Initial public release
- Core ring buffer implementation with GPU acceleration
- CUDA backend support for NVIDIA GPUs
- WebGPU backend for cross-platform compatibility
- CPU fallback implementation
- Lock-free SPSC (Single Producer Single Consumer) design
- Zero-copy architecture using unified memory
- PTY integration for terminal multiplexing
- CLI tool with REPL, streaming, and benchmark modes
- Runtime CUDA kernel compilation support
- Comprehensive benchmarks (throughput, latency, GPU vs CPU)

### Performance
- Sub-microsecond latency (<1μs)
- 2-3 GB/s sustained throughput
- >10M messages/second processing rate

[Unreleased]: https://github.com/the-muppet/perdix/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/the-muppet/perdix/releases/tag/v0.1.1
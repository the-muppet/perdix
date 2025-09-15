//! # PTY Module - Pseudo-Terminal Integration
//! 
//! This module provides cross-platform pseudo-terminal (PTY) functionality for
//! streaming GPU-accelerated output to terminal emulators. It enables Perdix to
//! act as a transparent layer between AI assistants and terminal applications.
//! 
//! ## Purpose
//! 
//! PTYs allow Perdix to:
//! - Route AI output through GPU acceleration to terminals
//! - Preserve ANSI formatting and control sequences
//! - Support interactive terminal applications
//! - Provide cross-platform terminal emulation
//! 
//! ## Architecture
//! 
//! ```text
//! ┌────────────────┐
//! │  AI Assistants │
//! └────────┬───────┘
//!          │
//!          ▼ GPU writes
//! ┌────────────────┐
//! │  Ring Buffer   │  Zero-copy GPU→CPU
//! └────────┬───────┘
//!          │
//!          ▼ Consumer reads
//! ┌────────────────┐
//! │  PTY Writer    │  This module
//! └────────┬───────┘
//!          │
//!          ▼ Writes to PTY
//! ┌────────────────┐
//! │    Terminal    │
//! └────────────────┘
//! ```
//! 
//! ## Features
//! 
//! - **Cross-Platform**: Works on Windows, Linux, macOS
//! - **ANSI Support**: Preserves colors and formatting
//! - **Async Operation**: Non-blocking PTY I/O
//! - **Process Management**: Spawns and manages shell processes
//! 
//! ## Usage
//! 
//! ```rust,no_run
//! use perdix::pty::portable::PortablePtyWriter;
//! use perdix::Buffer;
//! 
//! // Create buffer and get consumer
//! let buffer = Buffer::new(1024)?;
//! let (_, consumer) = buffer.split();
//! 
//! // Create PTY and start writer thread
//! let pty = PortablePtyWriter::new()?;
//! let (stop_flag, handle) = pty.start_writer_thread(consumer);
//! 
//! // ... GPU produces messages ...
//! 
//! // Stop PTY writer
//! stop_flag.store(true, std::sync::atomic::Ordering::Relaxed);
//! handle.join().unwrap();
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

pub mod portable;

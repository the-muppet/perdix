#[cfg(unix)]
use std::os::unix::io::{AsRawFd, RawFd};
use std::io::{self, Write};
use std::ptr;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

#[cfg(feature = "pty")]
use nix;
#[cfg(feature = "pty")]
use smallvec::{SmallVec, smallvec};

use crate::buffer::Consumer;

#[cfg(feature = "pty")]
use nix::pty::{self, PtyMaster};
#[cfg(feature = "pty")]
use nix::unistd::{write, close};
#[cfg(feature = "pty")]
use nix::sys::signal::{self, Signal};
#[cfg(feature = "pty")]
use nix::sys::wait::waitpid;

/// PTY output manager with zero-copy batching
pub struct PtyWriter {
    #[cfg(feature = "pty")]
    master_fd: RawFd,
    #[cfg(not(feature = "pty"))]
    stdout: std::io::Stdout,
    buffer: Vec<u8>,
    batch_size: usize,
    stop_flag: Arc<AtomicBool>,
}

impl PtyWriter {
    #[cfg(feature = "pty")]
    pub fn new() -> io::Result<Self> {
        let pty_result = pty::posix_openpt(pty::OFlag::O_RDWR)?;
        let master_fd = pty_result.as_raw_fd();
        
        // Grant access and unlock the slave
        pty::grantpt(&pty_result)?;
        pty::unlockpt(&pty_result)?;
        
        // Get slave name for connection
        let slave_name = pty::ptsname_r(&pty_result)?;
        println!("PTY slave: {}", slave_name);
        
        Ok(Self {
            master_fd,
            buffer: Vec::with_capacity(8192),
            batch_size: 4096,
            stop_flag: Arc::new(AtomicBool::new(false)),
        })
    }
    
    #[cfg(not(feature = "pty"))]
    pub fn new() -> io::Result<Self> {
        Ok(Self {
            stdout: std::io::stdout(),
            buffer: Vec::with_capacity(8192),
            batch_size: 4096,
            stop_flag: Arc::new(AtomicBool::new(false)),
        })
    }
    
    /// Start PTY flush thread for continuous output
    pub fn start_flush_thread(&self, buffer: &PinnedBuffer) -> thread::JoinHandle<()> {
        let stop_flag = Arc::clone(&self.stop_flag);
        
        #[cfg(feature = "pty")]
        let master_fd = self.master_fd;
        
        // Consumer for reading from ring buffer
        let consumer_buffer = unsafe {
            // Create a new reference to the buffer for the consumer
            let header_ptr = buffer.as_header();
            let slots_ptr = buffer.as_slots();
            std::mem::transmute::<&PinnedBuffer, &'static PinnedBuffer>(buffer)
        };
        
        thread::spawn(move || {
            let mut consumer = Consumer::new(consumer_buffer);
            let mut io_vectors = Vec::with_capacity(64);  // For writev batching
            let mut total_bytes = 0usize;
            
            while !stop_flag.load(Ordering::Acquire) {
                // Collect messages in batches for efficient writev()
                io_vectors.clear();
                total_bytes = 0;
                
                // Try to collect up to 64 messages or 64KB for batching
                for _ in 0..64 {
                    if let Some((seq, data)) = consumer.try_consume() {
                        // Add to writev batch
                        let iovec = libc::iovec {
                            iov_base: data.as_ptr() as *mut libc::c_void,
                            iov_len: data.len(),
                        };
                        io_vectors.push((iovec, data)); // Keep data alive
                        total_bytes += data.len();
                        
                        // If we hit our batch size limit, flush immediately
                        if total_bytes >= 65536 {
                            break;
                        }
                    } else {
                        break; // No more messages available
                    }
                }
                
                // Flush batch if we have data
                if !io_vectors.is_empty() {
                    Self::flush_batch(&io_vectors, total_bytes);
                } else {
                    // No data available, brief sleep to avoid spinning
                    thread::sleep(Duration::from_micros(100));
                }
            }
            
            println!("PTY flush thread stopped");
        })
    }
    
    #[cfg(feature = "pty")]
    fn flush_batch(io_vectors: &[(libc::iovec, Vec<u8>)], total_bytes: usize) {
        let iovecs: Vec<libc::iovec> = io_vectors.iter().map(|(iov, _)| *iov).collect();
        
        unsafe {
            let result = libc::writev(
                self.master_fd,
                iovecs.as_ptr(),
                iovecs.len() as libc::c_int,
            );
            
            if result < 0 {
                eprintln!("PTY writev failed: {}", io::Error::last_os_error());
            } else if result as usize != total_bytes {
                eprintln!("PTY partial write: {} of {} bytes", result, total_bytes);
            }
        }
    }
    
    #[cfg(not(feature = "pty"))]
    fn flush_batch(io_vectors: &[(libc::iovec, Vec<u8>)], _total_bytes: usize) {
        // Fallback to stdout for testing without PTY
        let mut stdout = std::io::stdout();
        for (_, data) in io_vectors {
            if stdout.write_all(data).is_err() {
                eprintln!("Failed to write to stdout");
                break;
            }
        }
        stdout.flush().ok();
    }
    
    /// Signal stop to flush thread
    pub fn stop(&self) {
        self.stop_flag.store(true, Ordering::Release);
    }
    
    /// Write single message immediately (for testing)
    pub fn write_immediate(&mut self, data: &[u8]) -> io::Result<()> {
        #[cfg(feature = "pty")]
        {
            use nix::unistd::write;
            match write(self.master_fd, data) {
                Ok(bytes_written) => {
                    if bytes_written != data.len() {
                        eprintln!("PTY partial write: {} of {} bytes", bytes_written, data.len());
                    }
                    Ok(())
                }
                Err(e) => Err(io::Error::from(e)),
            }
        }
        
        #[cfg(not(feature = "pty"))]
        {
            self.stdout.write_all(data)?;
            self.stdout.flush()?;
            Ok(())
        }
    }
}

impl Drop for PtyWriter {
    fn drop(&mut self) {
        self.stop();
        
        #[cfg(feature = "pty")]
        {
            if self.master_fd >= 0 {
                close(self.master_fd).ok();
            }
        }
    }
}

/// Signal handler for terminal size changes and cleanup
pub struct SignalHandler {
    stop_flag: Arc<AtomicBool>,
    sigwinch_flag: Arc<AtomicBool>,
}

impl SignalHandler {
    pub fn new() -> Self {
        Self {
            stop_flag: Arc::new(AtomicBool::new(false)),
            sigwinch_flag: Arc::new(AtomicBool::new(false)),
        }
    }
    
    /// Install signal handlers for graceful shutdown and SIGWINCH
    #[cfg(feature = "pty")]
    pub fn install_handlers(&self) -> io::Result<()> {
        use nix::sys::signal::{SigHandler, SigAction, SigSet};
        
        let stop_flag = Arc::clone(&self.stop_flag);
        let sigwinch_flag = Arc::clone(&self.sigwinch_flag);
        
        // SIGINT/SIGTERM handler for graceful shutdown
        let stop_handler = SigHandler::Handler(Self::signal_stop);
        let stop_action = SigAction::new(stop_handler, SigSet::empty(), signal::SaFlags::empty());
        
        unsafe {
            signal::sigaction(Signal::SIGINT, &stop_action)?;
            signal::sigaction(Signal::SIGTERM, &stop_action)?;
        }
        
        // SIGWINCH handler for terminal resize
        let winch_handler = SigHandler::Handler(Self::signal_winch);
        let winch_action = SigAction::new(winch_handler, SigSet::empty(), signal::SaFlags::empty());
        
        unsafe {
            signal::sigaction(Signal::SIGWINCH, &winch_action)?;
        }
        
        println!("Signal handlers installed");
        Ok(())
    }
    
    #[cfg(not(feature = "pty"))]
    pub fn install_handlers(&self) -> io::Result<()> {
        // Simplified signal handling without nix
        println!("Signal handling not available without PTY feature");
        Ok(())
    }
    
    extern "C" fn signal_stop(_: libc::c_int) {
        // Signal handlers must be async-signal-safe
        // We just set an atomic flag here
        unsafe {
            // This is a simplification - in practice you'd need a global reference
            // For a complete implementation, use a static AtomicBool
            static mut STOP_REQUESTED: bool = false;
            STOP_REQUESTED = true;
        }
    }
    
    extern "C" fn signal_winch(_: libc::c_int) {
        // Handle terminal size change
        unsafe {
            static mut WINCH_RECEIVED: bool = false;
            WINCH_RECEIVED = true;
        }
    }
    
    /// Check if stop was requested
    pub fn should_stop(&self) -> bool {
        self.stop_flag.load(Ordering::Acquire)
    }
    
    /// Check if window size changed
    pub fn winch_received(&self) -> bool {
        self.sigwinch_flag.swap(false, Ordering::AcqRel)
    }
    
    /// Request stop
    pub fn request_stop(&self) {
        self.stop_flag.store(true, Ordering::Release);
    }
}

/// Performance metrics for PTY output
#[derive(Debug, Default)]
pub struct PtyMetrics {
    pub messages_written: u64,
    pub bytes_written: u64,
    pub writev_calls: u64,
    pub average_batch_size: f64,
    pub write_errors: u64,
}

impl PtyMetrics {
    pub fn update_batch(&mut self, batch_size: usize, bytes: usize) {
        self.messages_written += batch_size as u64;
        self.bytes_written += bytes as u64;
        self.writev_calls += 1;
        
        // Update running average
        let total_messages = self.messages_written as f64;
        let total_calls = self.writev_calls as f64;
        self.average_batch_size = total_messages / total_calls;
    }
    
    pub fn report(&self) {
        println!("\n=== PTY Output Metrics ===");
        println!("Messages written: {}", self.messages_written);
        println!("Bytes written: {}", self.bytes_written);
        println!("writev() calls: {}", self.writev_calls);
        println!("Average batch size: {:.2}", self.average_batch_size);
        println!("Write errors: {}", self.write_errors);
        
        if self.writev_calls > 0 {
            let avg_bytes_per_call = self.bytes_written as f64 / self.writev_calls as f64;
            println!("Average bytes per writev(): {:.2}", avg_bytes_per_call);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_pty_creation() {
        let pty = PtyOutput::new();
        assert!(pty.is_ok());
    }
    
    #[test]
    fn test_signal_handler() {
        let handler = SignalHandler::new();
        assert!(!handler.should_stop());
        
        handler.request_stop();
        assert!(handler.should_stop());
    }
    
    #[test]
    fn test_metrics() {
        let mut metrics = PtyMetrics::default();
        
        metrics.update_batch(10, 1024);
        assert_eq!(metrics.messages_written, 10);
        assert_eq!(metrics.bytes_written, 1024);
        assert_eq!(metrics.writev_calls, 1);
        assert_eq!(metrics.average_batch_size, 10.0);
        
        metrics.update_batch(20, 2048);
        assert_eq!(metrics.messages_written, 30);
        assert_eq!(metrics.bytes_written, 3072);
        assert_eq!(metrics.writev_calls, 2);
        assert_eq!(metrics.average_batch_size, 15.0);
    }
}
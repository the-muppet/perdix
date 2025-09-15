use portable_pty::{native_pty_system, CommandBuilder, PtySize, MasterPty, Child};
use std::io::{Write, Read};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::Duration;
use crate::buffer::Consumer;

/// Cross-platform PTY writer using portable-pty
pub struct PortablePtyWriter {
    master: Box<dyn MasterPty + Send>,
    child: Box<dyn Child + Send + Sync>,
    stop_flag: Arc<AtomicBool>,
}

impl PortablePtyWriter {
    /// Create a new PTY with a shell
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        // Create a new pty
        let pty_system = native_pty_system();
        
        // Set size
        let pair = pty_system.openpty(PtySize {
            rows: 24,
            cols: 80,
            pixel_width: 0,
            pixel_height: 0,
        })?;
        
        // Spawn a shell
        let mut cmd = CommandBuilder::new(if cfg!(windows) { "cmd.exe" } else { "bash" });
        let child = pair.slave.spawn_command(cmd)?;
        
        println!("PTY created with shell PID: {:?}", child.process_id());
        
        Ok(Self {
            master: pair.master,
            child,
            stop_flag: Arc::new(AtomicBool::new(false)),
        })
    }
    
    /// Start the PTY writer thread that reads from ring buffer and writes to PTY
    pub fn start_writer_thread(
        mut self,
        mut consumer: Consumer<'static>,
    ) -> (Arc<AtomicBool>, thread::JoinHandle<u64>) {
        let stop_flag = Arc::clone(&self.stop_flag);
        let stop_flag_ret = Arc::clone(&self.stop_flag);
        
        let handle = thread::spawn(move || {
            println!("[PTY Writer] Thread started");
            let mut total = 0u64;
            let mut writer = self.master.take_writer().unwrap();
            
            while !self.stop_flag.load(Ordering::Relaxed) {
                // Try to consume messages from ring buffer
                if let Some(msg) = consumer.try_consume() {
                    // Write to PTY
                    if let Err(e) = writer.write_all(&msg.payload[..msg.len as usize]) {
                        println!("[PTY Writer] Write error: {}", e);
                        break;
                    }
                    
                    total += 1;
                    
                    // Flush periodically
                    if total % 10 == 0 {
                        let _ = writer.flush();
                    }
                } else {
                    // No data, yield
                    thread::yield_now();
                }
                
                // Check if child process is still alive
                if let Ok(Some(_exit_status)) = self.child.try_wait() {
                    println!("[PTY Writer] Child process exited");
                    break;
                }
            }
            
            println!("[PTY Writer] Thread stopped. Total messages: {}", total);
            total
        });
        
        (stop_flag_ret, handle)
    }
    
    /// Start a reader thread that echoes PTY output to console
    pub fn start_reader_thread(mut self) -> thread::JoinHandle<()> {
        thread::spawn(move || {
            let mut reader = self.master.try_clone_reader().unwrap();
            let mut buffer = [0u8; 4096];
            
            println!("[PTY Reader] Thread started - echoing PTY output");
            
            loop {
                match reader.read(&mut buffer) {
                    Ok(0) => {
                        println!("[PTY Reader] EOF");
                        break;
                    }
                    Ok(n) => {
                        // Echo to console
                        print!("{}", String::from_utf8_lossy(&buffer[..n]));
                        std::io::stdout().flush().unwrap();
                    }
                    Err(e) => {
                        if e.kind() != std::io::ErrorKind::WouldBlock {
                            println!("[PTY Reader] Error: {}", e);
                            break;
                        }
                        thread::sleep(Duration::from_millis(10));
                    }
                }
            }
            
            println!("[PTY Reader] Thread stopped");
        })
    }
}

/// Zero-copy PTY writer using vectored I/O (where available)
pub struct ZeroCopyPortablePty {
    master: Box<dyn MasterPty + Send>,
    child: Box<dyn Child + Send + Sync>,
}

impl ZeroCopyPortablePty {
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let pty_system = native_pty_system();
        let pair = pty_system.openpty(PtySize {
            rows: 40,
            cols: 120,
            pixel_width: 0,
            pixel_height: 0,
        })?;
        
        // Spawn shell
        let mut cmd = CommandBuilder::new(if cfg!(windows) { "powershell.exe" } else { "bash" });
        let child = pair.slave.spawn_command(cmd)?;
        
        Ok(Self {
            master: pair.master,
            child,
        })
    }
    
    /// Stream GPU messages directly to PTY
    pub fn stream_from_buffer(
        mut self,
        mut consumer: Consumer<'static>,
        stop_flag: Arc<AtomicBool>,
    ) -> u64 {
        let mut writer = self.master.take_writer().unwrap();
        let mut total = 0u64;
        let mut batch = Vec::with_capacity(64);
        
        println!("[PTY Stream] Starting GPU→PTY streaming...");
        
        while !stop_flag.load(Ordering::Relaxed) {
            batch.clear();
            
            // Batch consume for efficiency
            for _ in 0..64 {
                if let Some(msg) = consumer.try_consume() {
                    batch.push(msg);
                } else {
                    break;
                }
            }
            
            if batch.is_empty() {
                thread::yield_now();
                continue;
            }
            
            // Write batch to PTY
            for msg in &batch {
                if let Err(e) = writer.write_all(&msg.payload[..msg.len as usize]) {
                    println!("[PTY Stream] Write error: {}", e);
                    return total;
                }
            }
            
            total += batch.len() as u64;
            
            // Flush after batch
            let _ = writer.flush();
            
            // Check child status
            if let Ok(Some(_)) = self.child.try_wait() {
                println!("[PTY Stream] Child exited");
                break;
            }
        }
        
        println!("[PTY Stream] Streamed {} messages", total);
        total
    }
}
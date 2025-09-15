#[cfg(feature = "pty")]
use nix::pty::{openpty, OpenptyResult};
#[cfg(feature = "pty")]
use nix::sys::termios::{self, SetArg, Termios};
#[cfg(feature = "pty")]
use nix::sys::uio::IoVec;
#[cfg(feature = "pty")]
use std::os::unix::io::RawFd;

use crate::buffer::{Consumer, Slot, Header};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::Duration;
use libc::{iovec, c_void};

/// Zero-copy PTY writer that uses writev directly on ring buffer slots
pub struct ZeroCopyPtyWriter<'a> {
    #[cfg(feature = "pty")]
    master_fd: RawFd,
    #[cfg(feature = "pty")]
    slave_fd: RawFd,
    consumer: Consumer<'a>,
    stop_flag: Arc<AtomicBool>,
}

impl<'a> ZeroCopyPtyWriter<'a> {
    #[cfg(feature = "pty")]
    pub fn new(consumer: Consumer<'a>) -> std::io::Result<Self> {
        // Open PTY pair
        let OpenptyResult { master, slave } = openpty(None, None)?;
        
        // Configure slave terminal for raw mode
        let mut termios = termios::tcgetattr(slave)?;
        
        // Raw mode: no echo, no canonical processing, no signals
        termios.local_flags &= !(termios::LocalFlags::ECHO 
                              | termios::LocalFlags::ICANON 
                              | termios::LocalFlags::ISIG);
        termios.input_flags &= !(termios::InputFlags::ICRNL);
        termios.output_flags &= !(termios::OutputFlags::OPOST);
        
        // Apply settings
        termios::tcsetattr(slave, SetArg::TCSANOW, &termios)?;
        
        println!("PTY opened - master: {}, slave: {}", master, slave);
        println!("Connect to slave PTY: /dev/pts/{}", slave);
        
        Ok(Self {
            master_fd: master,
            slave_fd: slave,
            consumer,
            stop_flag: Arc::new(AtomicBool::new(false)),
        })
    }
    
    #[cfg(not(feature = "pty"))]
    pub fn new(consumer: Consumer<'a>) -> std::io::Result<Self> {
        Ok(Self {
            consumer,
            stop_flag: Arc::new(AtomicBool::new(false)),
        })
    }
    
    /// Start the zero-copy flush thread
    pub fn start_flush_thread(mut self) -> (Arc<AtomicBool>, thread::JoinHandle<()>) {
        let stop_flag = Arc::clone(&self.stop_flag);
        let stop_flag_ret = Arc::clone(&self.stop_flag);
        
        let handle = thread::spawn(move || {
            println!("[PTY Flush] Thread started with zero-copy I/O");
            
            // Pre-allocate iovec array for batching
            let mut iovecs: Vec<iovec> = Vec::with_capacity(64);
            let mut slot_refs: Vec<*const Slot> = Vec::with_capacity(64);
            let mut total_consumed = 0u64;
            
            while !self.stop_flag.load(Ordering::Acquire) {
                iovecs.clear();
                slot_refs.clear();
                let mut batch_bytes = 0usize;
                
                // Collect up to 64 messages for batched writev
                for _ in 0..64 {
                    if let Some(slot) = self.consumer.consume_raw() {
                        // Safety: slot is valid and pinned in memory
                        unsafe {
                            let payload_ptr = (*slot).payload.as_ptr();
                            let payload_len = (*slot).len as usize;
                            
                            // Only add non-empty payloads
                            if payload_len > 0 {
                                iovecs.push(iovec {
                                    iov_base: payload_ptr as *mut c_void,
                                    iov_len: payload_len,
                                });
                                
                                // Keep reference to prevent reuse
                                slot_refs.push(slot);
                                batch_bytes += payload_len;
                            }
                        }
                        
                        // Stop batching if we have enough data
                        if batch_bytes >= 65536 {
                            break;
                        }
                    } else {
                        // No more messages available
                        break;
                    }
                }
                
                // Flush batch if we have data
                if !iovecs.is_empty() {
                    #[cfg(feature = "pty")]
                    {
                        unsafe {
                            let result = libc::writev(
                                self.master_fd,
                                iovecs.as_ptr(),
                                iovecs.len() as libc::c_int
                            );
                            
                            if result < 0 {
                                let err = std::io::Error::last_os_error();
                                eprintln!("[PTY Flush] writev failed: {}", err);
                            } else if result as usize != batch_bytes {
                                eprintln!("[PTY Flush] Partial write: {} of {} bytes", 
                                         result, batch_bytes);
                            }
                        }
                    }
                    
                    #[cfg(not(feature = "pty"))]
                    {
                        // Fallback: print to stdout for debugging
                        for slot_ptr in &slot_refs {
                            unsafe {
                                let slot = &**slot_ptr;
                                let payload = std::slice::from_raw_parts(
                                    slot.payload.as_ptr(),
                                    slot.len as usize
                                );
                                if let Ok(text) = std::str::from_utf8(payload) {
                                    print!("{}", text);
                                }
                            }
                        }
                        let _ = std::io::Write::flush(&mut std::io::stdout());
                    }
                    
                    total_consumed += iovecs.len() as u64;
                    
                    // Periodically report stats
                    if total_consumed % 10000 == 0 && total_consumed > 0 {
                        println!("[PTY Flush] Consumed {} messages, last batch: {} msgs, {} bytes",
                                total_consumed, iovecs.len(), batch_bytes);
                    }
                } else {
                    // No data available, brief sleep
                    thread::sleep(Duration::from_micros(100));
                }
            }
            
            println!("[PTY Flush] Thread stopping. Total consumed: {}", total_consumed);
            
            #[cfg(feature = "pty")]
            {
                // Clean up PTY
                unsafe {
                    libc::close(self.master_fd);
                    libc::close(self.slave_fd);
                }
            }
        });
        
        (stop_flag_ret, handle)
    }
    
    pub fn stop(&self) {
        self.stop_flag.store(true, Ordering::Release);
    }
}

/// Alternative implementation using raw pointers for maximum performance
pub struct UnsafePtyFlusher {
    #[cfg(feature = "pty")]
    master_fd: RawFd,
    header: *const Header,
    slots: *const Slot,
    n_slots: usize,
}

impl UnsafePtyFlusher {
    pub unsafe fn new(header: *const Header, slots: *const Slot, n_slots: usize) -> std::io::Result<Self> {
        #[cfg(feature = "pty")]
        {
            let OpenptyResult { master, slave } = openpty(None, None)?;
            
            // Configure for raw mode
            let mut termios = termios::tcgetattr(slave)?;
            termios.local_flags &= !(termios::LocalFlags::ECHO 
                                  | termios::LocalFlags::ICANON 
                                  | termios::LocalFlags::ISIG);
            termios::tcsetattr(slave, SetArg::TCSANOW, &termios)?;
            
            Ok(Self {
                master_fd: master,
                header,
                slots,
                n_slots,
            })
        }
        
        #[cfg(not(feature = "pty"))]
        Ok(Self {
            header,
            slots, 
            n_slots,
        })
    }
    
    /// Ultra low-latency flush loop with no allocations
    pub unsafe fn flush_loop(&self, stop_flag: Arc<AtomicBool>) {
        let mut read_idx = 0u64;
        let wrap_mask = self.n_slots as u64 - 1;
        
        // Stack-allocated iovec array
        let mut iovecs: [iovec; 64] = [iovec {
            iov_base: std::ptr::null_mut(),
            iov_len: 0,
        }; 64];
        
        while !stop_flag.load(Ordering::Acquire) {
            let write_idx = (*self.header).producer.write_idx;
            let mut batch_size = 0;
            let mut batch_bytes = 0;
            
            // Collect available messages
            while read_idx < write_idx && batch_size < 64 {
                let slot_idx = (read_idx & wrap_mask) as usize;
                let slot = &*self.slots.add(slot_idx);
                
                // Wait for sequence to be published
                while slot.seq != read_idx {
                    std::hint::spin_loop();
                    if stop_flag.load(Ordering::Acquire) {
                        return;
                    }
                }
                
                // Add to batch
                if slot.len > 0 {
                    iovecs[batch_size] = iovec {
                        iov_base: slot.payload.as_ptr() as *mut c_void,
                        iov_len: slot.len as usize,
                    };
                    batch_bytes += slot.len as usize;
                    batch_size += 1;
                }
                
                read_idx += 1;
            }
            
            // Flush batch
            if batch_size > 0 {
                #[cfg(feature = "pty")]
                {
                    let result = libc::writev(
                        self.master_fd,
                        iovecs.as_ptr(),
                        batch_size as libc::c_int
                    );
                    
                    if result < 0 {
                        eprintln!("writev error: {}", std::io::Error::last_os_error());
                    }
                }
                
                // Update consumer index
                (*self.header).consumer.read_idx = read_idx;
            } else {
                // No data, yield CPU
                std::thread::yield_now();
            }
        }
    }
}
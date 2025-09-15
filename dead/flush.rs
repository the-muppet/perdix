use crate::buffer::spsc::Consumer;
use std::io::{self, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::Duration;

#[cfg(feature = "pty")]
use nix::pty::{self, PtyMaster};
#[cfg(feature = "pty")]
use smallvec::{smallvec, SmallVec};
#[cfg(unix)]
use std::os::unix::io::AsRawFd;

pub struct PtyFlusher {
    #[cfg(feature = "pty")]
    pty_master: Option<PtyMaster>,
    // The flusher takes ownership of a 'static consumer
    consumer: Consumer<'static>,
    stop_flag: Arc<AtomicBool>,
}

#[derive(Clone)]
pub struct StopHandle(Arc<AtomicBool>);

impl StopHandle {
    pub fn stop(&self) {
        self.0.store(true, Ordering::Release);
    }
}

impl PtyFlusher {
    /// Spawns a dedicated thread to flush the ring buffer to a new PTY.
    ///
    /// This takes ownership of a `Consumer` with a `'static` lifetime,
    /// typically created from an `Arc<Pinned>` buffer.
    pub fn spawn(consumer: Consumer<'static>) -> (JoinHandle<()>, StopHandle) {
        let pty_master = pty::posix_openpt(pty::OFlag::O_RDWR).expect("Failed to open PTY");
        pty::grantpt(&pty_master).unwrap();
        pty::unlockpt(&pty_master).unwrap();

        let slave_name = pty::ptsname_r(&pty_master).unwrap();
        println!(
            "[PTY Flusher] PTY slave device is available at: {}",
            slave_name
        );
        println!("[PTY Flusher] In another terminal, run: cat {}", slave_name);

        let stop_flag = Arc::new(AtomicBool::new(false));
        let stop_handle = StopHandle(stop_flag.clone());

        let mut flusher = Self {
            pty_master,
            consumer,
            stop_flag,
        };

        let handle = thread::spawn(move || flusher.run());

        (handle, stop_handle)
    }

    /// The main run loop for the flush thread.
    fn run(&mut self) {
        // SmallVec avoids heap allocations for small batches
        let mut iovecs: SmallVec<[libc::iovec; 64]> = smallvec![];
        let master_fd = self.pty_master.as_raw_fd();

        while !self.stop_flag.load(Ordering::Acquire) {
            iovecs.clear();

            // Collect a batch of available spans
            while iovecs.len() < iovecs.capacity() {
                if let Some(payload_slice) = self.consumer.consume_slice_and_advance() {
                    // Create an iovec that points directly into the pinned buffer.
                    let iov = libc::iovec {
                        iov_base: payload_slice.as_ptr() as *mut libc::c_void,
                        iov_len: payload_slice.len(),
                    };
                    iovecs.push(iov);
                } else {
                    // No more messages are immediately available.
                    break;
                }
            }

            // Flush the batch with a single `writev` syscall.
            if !iovecs.is_empty() {
                let result = unsafe {
                    libc::writev(master_fd, iovecs.as_ptr(), iovecs.len() as libc::c_int)
                };

                if result < 0 {
                    // Handle error, e.g., if the PTY reader disconnects.
                    eprintln!(
                        "[PTY Flusher] writev error: {}",
                        std::io::Error::last_os_error()
                    );
                    break;
                }

                // After a successful write, commit the read index.
                // This lets the GPU producer know these slots can be reused.
                self.consumer.commit_read();
            } else {
                // No work to do, sleep briefly to avoid busy-waiting.
                thread::sleep(Duration::from_micros(100));
            }
        }
        println!("[PTY Flusher] Flush thread shutting down.");
    }
}

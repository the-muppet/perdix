// spsc.rs
use std::ptr;
use std::sync::atomic::Ordering;
use std::sync::Arc;

use crate::buffer::ffi::{self, StreamContext};
use crate::buffer::{pinned::Pinned, Header, Slot};
#[cfg(feature = "cuda")]
use crate::buffer::GpuTextArena;

/// Producer handle for writing to the ring buffer.
pub struct Producer<'a> {
    // We can use a reference for simpler borrows...
    pinned_ref: Option<&'a Pinned>,
    // ...or an Arc for owned, thread-safe handles.
    pinned_arc: Option<Arc<Pinned>>,

    // Direct pointers for performance in the hot path
    header: *const Header,
    slots: *mut Slot,
    wrap_mask: u64,
}

impl<'a> Producer<'a> {
    /// Creates a new producer from a shared Pinned buffer (for moving across threads).
    pub fn new(arc_pinned: Arc<Pinned>) -> Self {
        let header_ptr = arc_pinned.header() as *const Header;
        let slots_ptr = arc_pinned.slots_ptr();
        let wrap_mask = unsafe { (*header_ptr).config.wrap_mask };

        Self {
            pinned_ref: None,
            pinned_arc: Some(arc_pinned),
            header: header_ptr,
            slots: slots_ptr,
            wrap_mask,
        }
    }

    /// Creates a new producer from a borrowed Pinned buffer.
    pub fn new_from_ref(pinned: &'a Pinned) -> Self {
        let header_ptr = pinned.header() as *const Header;
        let slots_ptr = pinned.slots_ptr();
        let wrap_mask = unsafe { (*header_ptr).config.wrap_mask };

        Self {
            pinned_ref: Some(pinned),
            pinned_arc: None,
            header: header_ptr,
            slots: slots_ptr,
            wrap_mask,
        }
    }

    /// Simple test function
    pub fn run_test(&self, n_messages: u32) -> Result<(), String> {
        let result = unsafe {
            ffi::launch_simple_test(
                self.slots,
                self.header as *mut _, // Cast from const to mut
                n_messages as std::os::raw::c_int,
            )
        };

        if result != 0 {
            return Err(format!("Simple test kernel failed with code: {}", result));
        }
        Ok(())
    }

    /// Process a batch of agent responses using the unified CUDA kernel.
    pub fn process_agent_responses(
        &self,
        contexts: &[StreamContext],
        enable_metrics: bool,
        stream: u64, // Pass the stream explicitly
    ) -> Result<(), String> {
        if contexts.is_empty() {
            return Ok(());
        }

        let result = unsafe {
            ffi::launch_unified_kernel(
                self.slots,
                self.header as *mut _,
                contexts.as_ptr(),
                contexts.len() as u32,
                enable_metrics as i32,
                stream as *mut std::ffi::c_void,  // Convert u64 to pointer for cudaStream_t
            )
        };

        if result != 0 {
            return Err(format!("Unified kernel failed with code: {}", result));
        }
        Ok(())
    }

    /// Async version using GPU text arena for true streaming
    #[cfg(feature = "cuda")]
    pub fn process_agent_responses_async(
        &mut self,
        contexts: &[StreamContext],
        enable_metrics: bool,
        arena: &mut GpuTextArena,
    ) -> Result<(), String> {
        if contexts.is_empty() {
            return Ok(());
        }
        
        // Pack messages into arena
        arena.pack_messages(contexts)?;
        
        // Upload to device asynchronously
        arena.upload_to_device_async()?;
        
        // Launch kernel asynchronously
        arena.launch_kernel_async(self.slots, self.header as *mut _, enable_metrics)?;
        
        // NO SYNCHRONIZATION - return immediately!
        Ok(())
    }
    
    /// CPU based produce with proper backpressure
    pub fn try_produce(&mut self, data: &[u8]) -> Result<u64, &'static str> {
        if data.len() > 240 {
            return Err("Payload too large");
        }

        unsafe {
            let header = &*self.header;
            
            // CRITICAL: Check if we're about to lap the consumer
            let write_idx = header.producer.write_idx.load(Ordering::Acquire);
            let read_idx = header.consumer.read_idx.load(Ordering::Acquire);
            let n_slots = header.config.slot_count as u64;
            
            // Prevent producer from overrunning consumer
            // Leave 64 slots as safety margin
            if (write_idx - read_idx) >= (n_slots - 64) {
                return Err("Buffer full - backpressure");
            }

            // Check for explicit backpressure signal
            if header.control.backpressure.load(Ordering::Acquire) != 0 {
                return Err("Backpressure active");
            }

            // Reserve sequence number
            let seq = header.producer.write_idx.fetch_add(1, Ordering::AcqRel);

            // Get slot
            let idx = (seq & self.wrap_mask) as usize;
            let slot = &mut *self.slots.add(idx);

            // Write payload
            ptr::copy_nonoverlapping(data.as_ptr(), slot.payload.as_mut_ptr(), data.len());
            // Use volatile writes for unified memory
            ptr::write_volatile(&mut slot.len, data.len() as u32);
            ptr::write_volatile(&mut slot.flags, 0);

            // Memory fence for cross-device visibility
            std::sync::atomic::fence(Ordering::Release);

            // Publish sequence
            ptr::write_volatile(&mut slot.seq, seq);
            std::sync::atomic::fence(Ordering::Release);

            // Update stats
            header
                .producer
                .messages_produced
                .fetch_add(1, Ordering::Relaxed);

            Ok(seq)
        }
    }
}

// Mark Producer as Send + Sync - safe because we maintain ownership/borrowing
unsafe impl<'a> Send for Producer<'a> {}
unsafe impl<'a> Sync for Producer<'a> {}

/// Consumer handle for reading from the ring buffer.
pub struct Consumer<'a> {
    pinned_ref: Option<&'a Pinned>,
    pinned_arc: Option<Arc<Pinned>>,

    // Direct pointers for performance
    header: *const Header,
    slots: *const Slot,
    wrap_mask: u64,

    // Consumer state
    read_seq: u64,
}

impl<'a> Consumer<'a> {
    /// Creates a new consumer from a shared Pinned buffer.
    pub fn new(arc_pinned: Arc<Pinned>) -> Self {
        let header_ptr = arc_pinned.header() as *const Header;
        let slots_ptr = arc_pinned.slots_ptr() as *const Slot;
        let wrap_mask = unsafe { (*header_ptr).config.wrap_mask };

        Self {
            pinned_ref: None,
            pinned_arc: Some(arc_pinned),
            header: header_ptr,
            slots: slots_ptr,
            wrap_mask,
            read_seq: 0,
        }
    }

    /// Creates a new consumer from a borrowed Pinned buffer.
    pub fn new_from_ref(pinned: &'a Pinned) -> Self {
        let header_ptr = pinned.header() as *const Header;
        let slots_ptr = pinned.slots_ptr() as *const Slot;
        let wrap_mask = unsafe { (*header_ptr).config.wrap_mask };

        Self {
            pinned_ref: Some(pinned),
            pinned_arc: None,
            header: header_ptr,
            slots: slots_ptr,
            wrap_mask,
            read_seq: 0,
        }
    }

    /// Try to consume one message from the ring buffer.
    pub fn try_consume(&mut self) -> Option<Message> {
        unsafe {
            let idx = (self.read_seq & self.wrap_mask) as usize;
            let slot = &*self.slots.add(idx);

            // Memory fence before reading for cross-device visibility
            std::sync::atomic::fence(Ordering::Acquire);

            // Use volatile read for unified memory
            let slot_seq = ptr::read_volatile(&slot.seq);

            // Check if slot is ready
            if slot_seq != self.read_seq {
                // Debug: print mismatch for first few attempts
                if self.read_seq < 25 {
                    println!("[Consumer] Seq mismatch at idx {}: expected {}, got {} (slot addr={:p})", 
                             idx, self.read_seq, slot_seq, slot as *const _);
                }
                return None;
            }

            // Memory fence after sequence check
            std::sync::atomic::fence(Ordering::Acquire);

            // Read slot data
            let len = ptr::read_volatile(&slot.len) as usize;
            let flags = ptr::read_volatile(&slot.flags);

            // Bounds check to prevent allocation errors
            if len > 240 {
                // Slot appears corrupted - skip it
                println!("[Consumer] WARNING: Corrupted slot at seq={}, len={}", slot_seq, len);
                return None;
            }

            // Copy payload
            let mut payload = vec![0u8; len];
            ptr::copy_nonoverlapping(slot.payload.as_ptr(), payload.as_mut_ptr(), len);

            // Update read index
            self.read_seq += 1;
            (*self.header)
                .consumer
                .read_idx
                .store(self.read_seq, Ordering::Release);
            (*self.header)
                .consumer
                .messages_consumed
                .fetch_add(1, Ordering::Relaxed);

            Some(Message {
                seq: slot_seq,
                payload,
                flags,
            })
        }
    }

    /// Consume up to max_messages from the buffer.
    pub fn consume_batch(&mut self, max_messages: usize) -> Vec<Message> {
        let mut messages = Vec::with_capacity(max_messages);

        for _ in 0..max_messages {
            if let Some(msg) = self.try_consume() {
                messages.push(msg);
            } else {
                break;
            }
        }

        messages
    }

    /// Blocking consume with timeout.
    pub fn consume_blocking(&mut self, timeout_ms: u64) -> Option<Message> {
        let start = std::time::Instant::now();
        let timeout = std::time::Duration::from_millis(timeout_ms);

        while start.elapsed() < timeout {
            if let Some(msg) = self.try_consume() {
                return Some(msg);
            }
            std::thread::yield_now();
        }

        None
    }

    /// Consume all available messages.
    pub fn drain(&mut self) -> Vec<Message> {
        let mut messages = Vec::new();
        while let Some(msg) = self.try_consume() {
            messages.push(msg);
        }
        messages
    }

    /// Check how many messages are available.
    pub fn available(&self) -> usize {
        unsafe {
            let header = &*self.header;
            let write_idx = header.producer.write_idx.load(Ordering::Acquire);
            let read_idx = header.consumer.read_idx.load(Ordering::Acquire);
            (write_idx - read_idx) as usize
        }
    }

    /// A more performant batching consume with careful calculation
    pub fn consume_available(&mut self, limit: Option<usize>) -> Vec<Message> {
        unsafe {
            let header = &*self.header;
            let write_idx = header.producer.write_idx.load(Ordering::Acquire);

            // Calculate how many we can consume
            let available = (write_idx - self.read_seq) as usize;
            let max_to_check = limit.unwrap_or(available).min(available);

            let mut messages = Vec::with_capacity(max_to_check);

            for _ in 0..max_to_check {
                if let Some(msg) = self.try_consume() {
                    messages.push(msg);
                } else {
                    // Slot not ready yet, stop here
                    break;
                }
            }

            messages
        }
    }
}

// Mark Consumer as Send + Sync
unsafe impl<'a> Send for Consumer<'a> {}
unsafe impl<'a> Sync for Consumer<'a> {}

/// A message consumed from the ring buffer.
pub struct Message {
    pub seq: u64,
    pub payload: Vec<u8>,
    pub flags: u32,
}

impl Message {
    /// Get the message as a string (assumes UTF-8).
    pub fn as_str(&self) -> Result<&str, std::str::Utf8Error> {
        std::str::from_utf8(&self.payload)
    }
}

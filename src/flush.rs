// src/flush.rs
use std::sync::Arc;
use crate::buffer::{Buffer, Header, Slot}
use std::sync::atomic::Ordering;
use std::time::Duration;

pub struct Flusher {
    buffer: Arc<Buffer>,
}

impl Flusher {
    pub fn new(buffer: Arc<Buffer>) -> Self {
        Self { buffer }
    }

    // The entire loop from your old flush thread is now a method.
    pub fn run_loop(&self, n_msgs_to_flush: u64) {
        let (header_ptr, slots_ptr) = self.buffer.as_raw_parts();
        let mut next_seq: u64 = 0;

        while next_seq < n_msgs_to_flush {
            let idx = (next_seq & unsafe { (*header_ptr).wrap_mask }) as usize;
            let slot = unsafe { &*slots_ptr.add(idx) };

            let seq_val = unsafe { std::ptr::read_volatile(&slot.seq) };
            if seq_val == next_seq {
                let len = unsafe { std::ptr::read_volatile(&slot.len) } as usize;
                let data = &slot.payload[..len];
                println!("[Flusher] Flushed seq {}: len={}", seq_val, len);
                next_seq += 1;
            } else {
                // Not ready, sleep briefly
                thread::sleep(Duration::from_micros(100));
            }
        }
    }
}
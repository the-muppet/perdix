use std::ptr;
use std::sync::atomic::{AtomicU64, Ordering};

/// Cache-line aligned slot for optimal memory access
/// CRITICAL: Do NOT use Rust atomics - they have different memory layout than C types!
#[repr(C, align(64))]
pub struct Slot {
    // Sequence number for ordering (written last by producer)
    // Using u64 directly - atomics will be handled via unsafe operations
    pub seq: u64,

    // Payload length - using u32 directly
    pub len: u32,

    // Flags for metadata - using u32 directly
    pub flags: u32,

    // Padding to align payload
    _pad1: u32,

    // Actual payload data
    pub payload: [u8; 240], // Fits in 4 cache lines total

    // Padding to ensure alignment
    _pad2: [u8; 8],
}

impl Slot {
    /// Check if slot is ready for reading (using volatile read)
    #[inline]
    pub fn is_ready(&self, expected_seq: u64) -> bool {
        unsafe {
            let seq_ptr = &self.seq as *const u64;
            let seq = ptr::read_volatile(seq_ptr);
            seq == expected_seq
        }
    }

    /// Reset slot for reuse (using volatile write)
    #[inline]
    pub fn reset(&mut self) {
        unsafe {
            ptr::write_volatile(&mut self.seq, u64::MAX);
            ptr::write_volatile(&mut self.len, 0);
            ptr::write_volatile(&mut self.flags, 0);
        }
    }

    /// Atomic read of sequence (for CPU side)
    #[inline]
    pub fn read_seq_atomic(&self) -> u64 {
        unsafe {
            let atomic_ref = &*(&self.seq as *const u64 as *const AtomicU64);
            atomic_ref.load(Ordering::Acquire)
        }
    }

    /// Atomic write of sequence (for CPU side)
    #[inline]
    pub fn write_seq_atomic(&self, seq: u64) {
        unsafe {
            let atomic_ref = &*(&self.seq as *const u64 as *const AtomicU64);
            atomic_ref.store(seq, Ordering::Release);
        }
    }
}

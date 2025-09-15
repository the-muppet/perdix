use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

/// Cache-line aligned header matching CUDA kernel layout
/// Total size: 256 bytes (4 cache lines)
#[repr(C, align(64))]
pub struct Header {
    // Producer cache line (64 bytes) - hot for GPU
    pub producer: ProducerSection,

    // Consumer cache line (64 bytes) - hot for CPU
    pub consumer: ConsumerSection,

    // Configuration cache line (64 bytes) - read-only after init
    pub config: ConfigSection,

    // Control cache line (64 bytes) - infrequent access
    pub control: ControlSection,
}

#[repr(C)]
pub struct ProducerSection {
    pub write_idx: AtomicU64,         // 8 bytes
    pub messages_produced: AtomicU64, // 8 bytes
    _pad: [u8; 48],                   // 48 bytes padding to 64
}

#[repr(C)]
pub struct ConsumerSection {
    pub read_idx: AtomicU64,          // 8 bytes
    pub messages_consumed: AtomicU64, // 8 bytes
    _pad: [u8; 48],                   // 48 bytes padding to 64
}

#[repr(C)]
pub struct ConfigSection {
    pub wrap_mask: u64,    // 8 bytes
    pub slot_count: u32,   // 4 bytes
    pub payload_size: u32, // 4 bytes
    pub batch_size: u32,   // 4 bytes
    _reserved: u32,        // 4 bytes
    _pad: [u8; 40],        // 40 bytes padding to 64
}

#[repr(C)]
pub struct ControlSection {
    pub backpressure: AtomicU32, // 4 bytes
    pub stop: AtomicU32,         // 4 bytes
    pub epoch: AtomicU32,        // 4 bytes
    pub error_count: AtomicU32,  // 4 bytes
    _pad: [u8; 48],              // 48 bytes padding to 64
}

impl Header {
    /// Initialize header with default values
    pub fn new(n_slots: usize) -> Self {
        assert!(n_slots.is_power_of_two(), "n_slots must be power of 2");

        Self {
            producer: ProducerSection {
                write_idx: AtomicU64::new(0),
                messages_produced: AtomicU64::new(0),
                _pad: [0; 48],
            },
            consumer: ConsumerSection {
                read_idx: AtomicU64::new(0),
                messages_consumed: AtomicU64::new(0),
                _pad: [0; 48],
            },
            config: ConfigSection {
                wrap_mask: (n_slots - 1) as u64,
                slot_count: n_slots as u32,
                payload_size: 192,
                batch_size: 32,
                _reserved: 0,
                _pad: [0; 40],
            },
            control: ControlSection {
                backpressure: AtomicU32::new(0),
                stop: AtomicU32::new(0),
                epoch: AtomicU32::new(0),
                error_count: AtomicU32::new(0),
                _pad: [0; 48],
            },
        }
    }

    /// Check if buffer is full (producer perspective)
    #[inline]
    pub fn is_full(&self) -> bool {
        let write = self.producer.write_idx.load(Ordering::Acquire);
        let read = self.consumer.read_idx.load(Ordering::Acquire);
        (write - read) >= self.config.slot_count as u64
    }

    /// Check if buffer is empty (consumer perspective)
    #[inline]
    pub fn is_empty(&self) -> bool {
        let write = self.producer.write_idx.load(Ordering::Acquire);
        let read = self.consumer.read_idx.load(Ordering::Acquire);
        write == read
    }

    /// Get current fill level
    #[inline]
    pub fn fill_level(&self) -> u64 {
        let write = self.producer.write_idx.load(Ordering::Acquire);
        let read = self.consumer.read_idx.load(Ordering::Acquire);
        write.saturating_sub(read)
    }
}

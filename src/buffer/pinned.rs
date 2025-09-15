use crate::buffer::ffi;
use crate::buffer::{Header, Slot};
use std::ptr;

/// A safe RAII wrapper around the CUDA unified memory buffer.
///
/// This struct is responsible for calling the CUDA C API to allocate the
/// memory on creation and free it on drop.
pub struct Pinned {
    // These are the raw pointers returned by the CUDA API.
    pub header: *mut Header,
    pub slots: *mut Slot,
    // We store this to provide it to the consumer later.
    pub n_slots: usize,
}

// SAFETY: The PinnedBuffer struct owns the raw pointers to memory that is
// managed by CUDA. This memory is designed to be accessible from multiple
// threads (both CPU and GPU). We guarantee safe access through our SPSC logic.
// Therefore, it's safe to send this owner object across threads.
unsafe impl Send for Pinned {}
unsafe impl Sync for Pinned {}

impl Pinned {
    /// Allocates a new unified memory ring buffer using the CUDA API.
    pub fn new(n_slots: usize) -> Result<Self, String> {
        // Power-of-two check is essential for the wrap_mask optimization
        if !n_slots.is_power_of_two() {
            return Err("n_slots must be a power of two".to_string());
        }

        // Initialize the CUDA device context first.
        let device_id = unsafe { ffi::cuda_init_device(0) };
        if device_id < 0 {
            return Err("Failed to initialize CUDA device".to_string());
        }

        // Prepare pointers to be filled by the FFI call.
        let mut header_ptr: *mut Header = ptr::null_mut();
        let mut slots_ptr: *mut Slot = ptr::null_mut();

        println!(
            "[Buffer] Allocating unified buffer with {} slots...",
            n_slots
        );

        // Call the C function to perform the allocation.
        let result = unsafe {
            ffi::init_unified_buffer(
                &mut slots_ptr,
                &mut header_ptr,
                n_slots as std::os::raw::c_int,
            )
        };

        // Check for allocation failure.
        if result != 0 || header_ptr.is_null() || slots_ptr.is_null() {
            return Err("CUDA FFI call to init_unified_buffer failed".to_string());
        }

        println!("[Buffer] Unified buffer allocated successfully.");
        // If allocation is successful, create the struct that owns the pointers.
        Ok(Self {
            header: header_ptr,
            slots: slots_ptr,
            n_slots,
        })
    }

    /// Provides access to the raw header pointer for FFI.
    #[inline]
    pub fn as_header(&self) -> *mut Header {
        self.header
    }

    /// Alias for as_header for consistency
    #[inline]
    pub fn header(&self) -> *mut Header {
        self.header
    }

    /// Provides access to the raw slots pointer for FFI.
    #[inline]
    pub fn as_slots(&self) -> *mut Slot {
        self.slots
    }

    /// Alias for as_slots for consistency
    #[inline]
    pub fn slots_ptr(&self) -> *mut Slot {
        self.slots
    }

    #[inline]
    pub fn slot_count(&self) -> usize {
        self.n_slots
    }
}

// The crucial part: The Drop trait ensures cleanup is always called.
impl Drop for Pinned {
    fn drop(&mut self) {
        println!("[Buffer] Cleaning up unified CUDA buffer...");
        // This is guaranteed to be called when Pinned goes out of scope,
        // preventing memory leaks even in the case of panics.
        unsafe {
            ffi::cleanup_unified_buffer(self.slots, self.header);
        }
    }
}

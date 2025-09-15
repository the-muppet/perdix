// Test program to verify GPU-CPU synchronization with unified memory

use std::ptr;
use std::sync::atomic::{AtomicU64, Ordering, fence};
use std::thread;
use std::time::Duration;

// Match the exact CUDA struct layout
#[repr(C, align(64))]
struct TestSlot {
    seq: u64,
    len: u32,
    flags: u32,
    _pad1: u32,
    payload: [u8; 240],
    _pad2: [u8; 8],
}

// Test unified memory visibility
pub fn test_unified_memory_sync() {
    println!("\n=== Testing Unified Memory Synchronization ===\n");
    
    #[cfg(feature = "cuda")]
    {
        extern "C" {
            fn cudaMallocManaged(ptr: *mut *mut u8, size: usize, flags: u32) -> i32;
            fn cudaMemPrefetchAsync(ptr: *const u8, size: usize, device: i32, stream: u64) -> i32;
            fn cudaDeviceSynchronize() -> i32;
            fn cudaFree(ptr: *mut u8) -> i32;
        }
        
        const CUDA_CPU_DEVICE_ID: i32 = -1;
        const N_SLOTS: usize = 32;
        
        // Allocate unified memory
        let mut ptr: *mut u8 = ptr::null_mut();
        let size = N_SLOTS * std::mem::size_of::<TestSlot>();
        
        let result = unsafe { cudaMallocManaged(&mut ptr, size, 1) };
        if result != 0 || ptr.is_null() {
            panic!("Failed to allocate unified memory");
        }
        
        println!("Allocated {} bytes of unified memory at {:p}", size, ptr);
        
        // Initialize slots
        let slots = ptr as *mut TestSlot;
        unsafe {
            for i in 0..N_SLOTS {
                let slot = &mut *slots.add(i);
                ptr::write_bytes(slot, 0, 1);
                slot.seq = u64::MAX;  // Invalid sequence
            }
        }
        
        // Test 1: CPU write, CPU read (baseline)
        println!("\nTest 1: CPU write -> CPU read");
        unsafe {
            let slot = &mut *slots;
            ptr::write_volatile(&mut slot.seq, 42);
            ptr::write_volatile(&mut slot.len, 100);
            fence(Ordering::Release);
            
            let read_seq = ptr::read_volatile(&slot.seq);
            let read_len = ptr::read_volatile(&slot.len);
            println!("  Written: seq=42, len=100");
            println!("  Read: seq={}, len={}", read_seq, read_len);
            assert_eq!(read_seq, 42);
            assert_eq!(read_len, 100);
        }
        
        // Test 2: Prefetch pattern
        println!("\nTest 2: Testing prefetch pattern");
        unsafe {
            // Prefetch to GPU
            cudaMemPrefetchAsync(ptr, size, 0, 0);
            cudaDeviceSynchronize();
            println!("  Prefetched to GPU");
            
            // Simulate GPU write (from CPU for testing)
            let slot = &mut *slots.add(1);
            ptr::write_volatile(&mut slot.seq, 99);
            ptr::write_volatile(&mut slot.len, 200);
            
            // Prefetch back to CPU
            cudaMemPrefetchAsync(ptr, size, CUDA_CPU_DEVICE_ID, 0);
            cudaDeviceSynchronize();
            println!("  Prefetched back to CPU");
            
            let read_seq = ptr::read_volatile(&slot.seq);
            let read_len = ptr::read_volatile(&slot.len);
            println!("  Read: seq={}, len={}", read_seq, read_len);
            assert_eq!(read_seq, 99);
            assert_eq!(read_len, 200);
        }
        
        // Test 3: Multiple slots
        println!("\nTest 3: Multiple slots");
        unsafe {
            for i in 0..5 {
                let slot = &mut *slots.add(i);
                ptr::write_volatile(&mut slot.seq, i as u64);
                ptr::write_volatile(&mut slot.len, (i * 10) as u32);
            }
            
            fence(Ordering::Release);
            
            for i in 0..5 {
                let slot = &*slots.add(i);
                let seq = ptr::read_volatile(&slot.seq);
                let len = ptr::read_volatile(&slot.len);
                println!("  Slot {}: seq={}, len={}", i, seq, len);
                assert_eq!(seq, i as u64);
                assert_eq!(len, (i * 10) as u32);
            }
        }
        
        // Cleanup
        unsafe {
            cudaFree(ptr);
        }
        
        println!("\n=== All synchronization tests passed! ===\n");
    }
    
    #[cfg(not(feature = "cuda"))]
    {
        println!("CUDA not enabled, skipping tests");
    }
}

// Test atomic vs non-atomic access patterns
pub fn test_atomic_patterns() {
    println!("\n=== Testing Atomic Access Patterns ===\n");
    
    // Test that volatile reads/writes work correctly
    let mut value: u64 = 0;
    
    unsafe {
        ptr::write_volatile(&mut value, 42);
        let read = ptr::read_volatile(&value);
        assert_eq!(read, 42);
        println!("Volatile access: OK");
    }
    
    // Test atomic through cast
    unsafe {
        let atomic_ref = &*(&value as *const u64 as *const AtomicU64);
        atomic_ref.store(100, Ordering::Release);
        let read = atomic_ref.load(Ordering::Acquire);
        assert_eq!(read, 100);
        println!("Atomic cast: OK");
    }
    
    println!("\n=== Atomic pattern tests passed! ===\n");
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_sync() {
        test_unified_memory_sync();
        test_atomic_patterns();
    }
}
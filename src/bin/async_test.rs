use perdix::buffer::{Buffer, Producer, Consumer};
use perdix::buffer::ffi::AgentType;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::{Duration, Instant};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔══════════════════════════════════════════════╗");
    println!("║       Perdix Async GPU Streaming Test       ║");
    println!("╚══════════════════════════════════════════════╝");
    
    // Configuration
    let n_slots = 4096;
    let n_gpu_messages = 500;
    
    println!("\nConfiguration:");
    println!("  Buffer slots: {}", n_slots);
    println!("  GPU messages: {}", n_gpu_messages);
    
    // Initialize unified memory buffer
    println!("\nInitializing unified memory buffer...");
    let buffer = Buffer::new(n_slots)?;
    let (mut producer, consumer) = buffer.split();
    
    // Start consumer thread
    let stop_flag = Arc::new(AtomicBool::new(false));
    let stop_clone = stop_flag.clone();
    
    let consumer_handle = thread::spawn(move || {
        let mut consumer = consumer;
        let mut total = 0u64;
        let start = Instant::now();
        
        println!("[Consumer] Thread started - waiting for async GPU messages");
        
        while !stop_clone.load(Ordering::Relaxed) && total < n_gpu_messages as u64 {
            if let Some(msg) = consumer.try_consume() {
                // Print sample messages
                if total < 5 || total % 100 == 0 || total == n_gpu_messages as u64 - 1 {
                    let text = String::from_utf8_lossy(&msg.payload);
                    println!("[Consumer] Message {}: {}", msg.seq, text.trim());
                }
                
                total += 1;
            } else {
                thread::yield_now();
            }
        }
        
        let elapsed = start.elapsed();
        let throughput = total as f64 / elapsed.as_secs_f64();
        println!("\n[Consumer] Received {} messages in {:?}", total, elapsed);
        println!("[Consumer] Throughput: {:.0} messages/second", throughput);
        total
    });
    
    // Run async GPU test
    #[cfg(feature = "cuda")]
    {
        println!("\n🚀 Testing ASYNC GPU streaming (no synchronization)...");
        
        // Use the simple test kernel for now
        let start = Instant::now();
        producer.run_test(n_gpu_messages)?;
        println!("  Kernel launched in {:?}", start.elapsed());
        
        // Note: The kernel is now running asynchronously!
        // Consumer will receive messages as they're produced
    }
    
    #[cfg(not(feature = "cuda"))]
    {
        println!("\n⚠️  CUDA not enabled - using CPU producer");
        for i in 0..n_gpu_messages {
            let msg = format!("CPU Message #{:03}", i);
            producer.try_produce(msg.as_bytes())?;
        }
    }
    
    // Wait for consumer
    let total = consumer_handle.join().unwrap();
    
    println!("\n╔══════════════════════════════════════════════╗");
    println!("║              Test Complete!                 ║");
    println!("╠══════════════════════════════════════════════╣");
    println!("║  Messages sent:     {:24} ║", n_gpu_messages);
    println!("║  Messages received: {:24} ║", total);
    println!("║  Status: {}                           ║", 
             if total == n_gpu_messages as u64 { "SUCCESS ✅" } else { "PARTIAL ⚠️ " });
    println!("╚══════════════════════════════════════════════╝");
    
    Ok(())
}
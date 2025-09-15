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
        let mut last_seq = None;
        let start = Instant::now();
        
        println!("[Consumer] Thread started - waiting for async GPU messages");
        
        while !stop_clone.load(Ordering::Relaxed) {
            if let Some(msg) = consumer.try_consume() {
                // Verify sequence
                if let Some(last) = last_seq {
                    if msg.seq != last + 1 {
                        println!("[Consumer] ERROR: Sequence gap! {} -> {}", last, msg.seq);
                    }
                }
                last_seq = Some(msg.seq);
                
                // Print sample messages
                if total < 5 || total % 100 == 0 || total == n_gpu_messages as u64 - 1 {
                    let text = String::from_utf8_lossy(&msg.payload);
                    println!("[Consumer] Message {}: {}", msg.seq, text.trim());
                }
                
                total += 1;
                
                // Stop after receiving all messages
                if total >= n_gpu_messages as u64 {
                    let elapsed = start.elapsed();
                    let throughput = total as f64 / elapsed.as_secs_f64();
                    println!("\n[Consumer] Received all {} messages in {:?}", total, elapsed);
                    println!("[Consumer] Throughput: {:.0} messages/second", throughput);
                    break;
                }
            } else {
                // Yield to avoid busy waiting
                thread::yield_now();
            }
        }
        
        println!("[Consumer] Thread stopped. Total consumed: {}", total);
        total
    });
    
    // Run async GPU test
    #[cfg(feature = "cuda")]
    {
        run_async_gpu_test(&mut producer, n_gpu_messages)?;
    }
    
    #[cfg(not(feature = "cuda"))]
    {
        println!("\n⚠️  CUDA not enabled - simulating with CPU producer");
        for i in 0..n_gpu_messages {
            let msg = format!("CPU Simulation Message #{:03}", i);
            producer.try_produce(msg.as_bytes())?;
        }
    }
    
    // Wait for consumer to finish
    let timeout = Duration::from_secs(10);
    let start = Instant::now();
    
    while start.elapsed() < timeout {
        if consumer_handle.is_finished() {
            break;
        }
        thread::sleep(Duration::from_millis(100));
    }
    
    // Stop consumer if still running
    stop_flag.store(true, Ordering::Release);
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

#[cfg(feature = "cuda")]
fn run_async_gpu_test(producer: &mut Producer, n_messages: u32) -> Result<(), Box<dyn std::error::Error>> {
    use perdix::buffer::gpu_arena::GpuTextArena;
    
    println!("\n🚀 Launching ASYNC GPU kernel (no blocking)...");
    
    // Create GPU text arena
    let mut arena = GpuTextArena::new(1024 * 1024)?; // 1MB arena
    
    // Generate diverse messages
    for i in 0..n_messages {
        let text = format!("🚀 Async GPU Msg #{:04}", i);
        let agent_type = match i % 7 {
            0 => AgentType::System,
            1 => AgentType::User,
            2 => AgentType::Assistant,
            3 => AgentType::Error,
            4 => AgentType::Warning,
            5 => AgentType::Info,
            _ => AgentType::Debug,
        };
        arena.add_text(text.as_bytes(), agent_type)?;
    }
    
    // Pack messages and get info
    println!("  Packing {} messages into arena...", n_messages);
    let (num_contexts, num_bytes) = {
        let (packed_contexts, text_data) = arena.pack();
        (packed_contexts.len(), text_data.len())
    };
    
    println!("  Uploading to GPU ({} bytes text, {} contexts)...", num_bytes, num_contexts);
    
    // Now we can upload without borrow issues since the refs are dropped
    arena.upload_to_device_async()?;
    
    // Launch async kernel - returns immediately!
    let start = Instant::now();
    producer.launch_async_gpu_kernel(&arena, n_messages)?;
    let launch_time = start.elapsed();
    
    println!("  ✅ Kernel launched in {:?} (returned immediately)", launch_time);
    println!("  📊 GPU is now streaming {} messages asynchronously...", n_messages);
    println!("  ⏳ Consumer will receive messages as GPU produces them");
    
    // The kernel is now running in the background!
    // The consumer thread will receive messages as they become available
    
    Ok(())
}
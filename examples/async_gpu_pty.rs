use perdix::buffer::{Buffer, Producer, Consumer};
use perdix::buffer::gpu_arena::GpuTextArena;
#[cfg(feature = "pty")]
use perdix::pty::zero_copy::ZeroCopyPtyWriter;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::{Duration, Instant};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔══════════════════════════════════════════════╗");
    println!("║    Perdix Async GPU→PTY Streaming Demo      ║");
    println!("╚══════════════════════════════════════════════╝");
    
    // Configuration
    let n_slots = 4096;
    let n_gpu_messages = 1000;
    
    println!("\nConfiguration:");
    println!("  Buffer slots: {}", n_slots);
    println!("  GPU messages: {}", n_gpu_messages);
    
    // Initialize unified memory buffer
    println!("\nInitializing unified memory buffer...");
    let buffer = Buffer::new(n_slots)?;
    let (mut producer, consumer) = buffer.split();
    
    // Create PTY writer if feature enabled
    #[cfg(feature = "pty")]
    {
        println!("\nInitializing PTY writer with zero-copy I/O...");
        let pty_writer = ZeroCopyPtyWriter::new(consumer)?;
        let (stop_flag, pty_handle) = pty_writer.start_flush_thread();
        
        // Run async GPU streaming test
        run_async_gpu_test(&mut producer, n_gpu_messages)?;
        
        // Let it run for a bit
        thread::sleep(Duration::from_secs(2));
        
        // Stop PTY writer
        stop_flag.store(true, Ordering::Release);
        pty_handle.join().unwrap();
        
        println!("\n✅ Async GPU→PTY streaming complete!");
    }
    
    #[cfg(not(feature = "pty"))]
    {
        // Fallback: just consume to stdout
        let stop_flag = Arc::new(AtomicBool::new(false));
        let stop_clone = stop_flag.clone();
        
        let consumer_handle = thread::spawn(move || {
            let mut consumer = consumer;
            let mut total = 0u64;
            
            println!("[Consumer] Thread started (stdout mode)");
            while !stop_clone.load(Ordering::Relaxed) {
                if let Some(msg) = consumer.try_consume() {
                    if total < 5 || total % 100 == 0 {
                        let text = String::from_utf8_lossy(&msg.payload);
                        println!("[Consumer] Message {}: {}", msg.seq, text.trim());
                    }
                    total += 1;
                    
                    if total >= n_gpu_messages as u64 {
                        break;
                    }
                } else {
                    thread::yield_now();
                }
            }
            println!("[Consumer] Total consumed: {}", total);
            total
        });
        
        // Run async GPU test
        run_async_gpu_test(&mut producer, n_gpu_messages)?;
        
        // Wait for consumer
        thread::sleep(Duration::from_secs(2));
        stop_flag.store(true, Ordering::Release);
        let total = consumer_handle.join().unwrap();
        
        println!("\n✅ Async GPU streaming complete! Messages: {}", total);
    }
    
    Ok(())
}

#[cfg(feature = "cuda")]
fn run_async_gpu_test(producer: &mut Producer, n_messages: u32) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🚀 Launching ASYNC GPU kernel (no blocking)...");
    
    // Create GPU text arena with sample messages
    let mut arena = GpuTextArena::new()?;
    
    // Generate messages
    for i in 0..n_messages {
        let text = format!("🚀 GPU Stream Message #{:04} - Async streaming from GPU to PTY!", i);
        let agent_type = match i % 5 {
            0 => perdix::buffer::ffi::AgentType::System,
            1 => perdix::buffer::ffi::AgentType::User,
            2 => perdix::buffer::ffi::AgentType::Assistant,
            3 => perdix::buffer::ffi::AgentType::Info,
            _ => perdix::buffer::ffi::AgentType::Debug,
        };
        arena.add_text(text.as_bytes(), agent_type)?;
    }
    
    // Pack and upload to GPU
    let (packed_contexts, text_data) = arena.pack();
    arena.upload_to_device(&packed_contexts, &text_data)?;
    
    // Launch async kernel - returns immediately!
    let start = Instant::now();
    producer.launch_async_gpu_kernel(&arena, n_messages)?;
    let launch_time = start.elapsed();
    
    println!("  Kernel launched in {:?} (returned immediately)", launch_time);
    println!("  GPU is now streaming {} messages asynchronously...", n_messages);
    
    Ok(())
}

#[cfg(not(feature = "cuda"))]
fn run_async_gpu_test(_producer: &mut Producer, _n_messages: u32) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n⚠️  CUDA not enabled - simulating with CPU producer");
    
    // Simulate async behavior with CPU
    for i in 0..100 {
        let msg = format!("CPU Simulation Message #{:03}", i);
        _producer.try_produce(msg.as_bytes())?;
    }
    
    Ok(())
}
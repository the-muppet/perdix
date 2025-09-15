use perdix::buffer::Buffer;
use perdix::pty::portable::{PortablePtyWriter, ZeroCopyPortablePty};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::{Duration, Instant};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔══════════════════════════════════════════════╗");
    println!("║    Perdix GPU→PTY Cross-Platform Demo       ║");
    println!("╚══════════════════════════════════════════════╝");
    
    // Check args
    let args: Vec<String> = std::env::args().collect();
    let zerocopy = args.contains(&"--zerocopy".to_string());
    let n_messages = args.iter()
        .find_map(|arg| arg.strip_prefix("--messages="))
        .and_then(|n| n.parse::<u32>().ok())
        .unwrap_or(100);
    
    println!("\nConfiguration:");
    println!("  Mode: {}", if zerocopy { "Zero-copy streaming" } else { "Buffered writer" });
    println!("  Messages: {}", n_messages);
    println!("  Platform: {}", std::env::consts::OS);
    
    // Initialize buffer
    let buffer = Buffer::new(4096)?;
    let (mut producer, consumer) = buffer.split();
    
    // Set async mode
    std::env::set_var("PERDIX_ASYNC", "1");
    
    if zerocopy {
        // Zero-copy mode: direct GPU→PTY streaming
        println!("\n🚀 Zero-copy GPU→PTY streaming mode");
        
        let pty = ZeroCopyPortablePty::new()?;
        let stop_flag = Arc::new(AtomicBool::new(false));
        let stop_clone = Arc::clone(&stop_flag);
        
        // Start streaming in background
        let stream_handle = thread::spawn(move || {
            pty.stream_from_buffer(consumer, stop_clone)
        });
        
        // Launch GPU kernel
        launch_gpu_messages(&mut producer, n_messages)?;
        
        // Let it stream
        thread::sleep(Duration::from_secs(3));
        
        // Stop streaming
        stop_flag.store(true, Ordering::Relaxed);
        let total = stream_handle.join().unwrap();
        
        println!("\n✅ Streamed {} messages from GPU to PTY", total);
        
    } else {
        // Buffered mode with separate reader/writer threads
        println!("\n📝 Buffered PTY mode with echo");
        
        // Create PTY with shell
        let pty_writer = PortablePtyWriter::new()?;
        
        // Clone for reader thread
        let pty_reader = PortablePtyWriter::new()?;
        
        // Start reader thread to echo PTY output
        let reader_handle = pty_reader.start_reader_thread();
        
        // Start writer thread
        let (stop_flag, writer_handle) = pty_writer.start_writer_thread(consumer);
        
        // Give PTY time to initialize
        thread::sleep(Duration::from_millis(500));
        
        // Launch GPU kernel
        launch_gpu_messages(&mut producer, n_messages)?;
        
        // Let it run
        println!("\n⏳ GPU streaming to PTY for 5 seconds...");
        thread::sleep(Duration::from_secs(5));
        
        // Stop writer
        stop_flag.store(true, Ordering::Relaxed);
        let total = writer_handle.join().unwrap();
        
        println!("\n✅ Wrote {} messages from GPU to PTY", total);
        
        // Reader will stop when PTY closes
        let _ = reader_handle.join();
    }
    
    println!("\n╔══════════════════════════════════════════════╗");
    println!("║         GPU→PTY Streaming Complete!         ║");
    println!("╚══════════════════════════════════════════════╝");
    
    Ok(())
}

#[cfg(feature = "cuda")]
fn launch_gpu_messages(producer: &mut perdix::buffer::Producer, n: u32) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🚀 Launching async GPU kernel with {} messages...", n);
    
    let start = Instant::now();
    producer.run_test(n)?;
    println!("  Kernel launched in {:?} (async - no blocking)", start.elapsed());
    
    Ok(())
}

#[cfg(not(feature = "cuda"))]
fn launch_gpu_messages(producer: &mut perdix::buffer::Producer, n: u32) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n📝 CUDA not enabled - using CPU producer");
    
    for i in 0..n {
        let msg = format!("\x1b[32mCPU Message #{:03}\x1b[0m\r\n", i);
        producer.try_produce(msg.as_bytes())?;
    }
    
    Ok(())
}
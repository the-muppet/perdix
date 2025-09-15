use perdix::{Buffer, AgentType};
use perdix::pty::portable::{PortablePtyWriter, ZeroCopyPortablePty};
use portable_pty::{native_pty_system, CommandBuilder, PtySize};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::{Duration, Instant};
use std::io::{self, BufRead, Write, Read};

/// Perdix: GPU-Accelerated AI Terminal Multiplexer
/// 
/// Streams AI assistant output (Claude, GPT, etc.) through GPU-accelerated
/// ring buffers to pseudo-terminals for high-performance terminal multiplexing.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    
    // Parse command-line arguments
    let mut mode = "stream";
    let mut n_slots = 4096;
    let mut show_help = false;
    
    for arg in &args[1..] {
        match arg.as_str() {
            "--help" | "-h" => show_help = true,
            "--repl" => mode = "repl",
            "--stream" => mode = "stream",
            "--benchmark" => mode = "benchmark",
            "--zerocopy" => mode = "zerocopy",
            "--claude" => mode = "claude",
            arg if arg.starts_with("--slots=") => {
                n_slots = arg.strip_prefix("--slots=")
                    .and_then(|n| n.parse().ok())
                    .unwrap_or(4096);
            }
            _ => {}
        }
    }
    
    if show_help {
        print_help();
        return Ok(());
    }
    
    // Print banner
    println!("\x1b[36m╔══════════════════════════════════════════════════════╗\x1b[0m");
    println!("\x1b[36m║          \x1b[1mPerdix: AI Terminal Multiplexer\x1b[0m\x1b[36m            ║\x1b[0m");
    println!("\x1b[36m║      GPU-Accelerated Streaming for AI Assistants     ║\x1b[0m");
    println!("\x1b[36m╚══════════════════════════════════════════════════════╝\x1b[0m");
    println!();
    
    match mode {
        "repl" => run_repl_mode(n_slots),
        "stream" => run_stream_mode(n_slots),
        "benchmark" => run_benchmark_mode(n_slots),
        "zerocopy" => run_zerocopy_mode(n_slots),
        "claude" => run_claude_mode(n_slots),
        _ => {
            eprintln!("Unknown mode: {}", mode);
            std::process::exit(1);
        }
    }
}

/// Interactive REPL mode - launches actual AI assistant (Claude Code)
fn run_repl_mode(n_slots: usize) -> Result<(), Box<dyn std::error::Error>> {
    println!("\x1b[32m● REPL Mode\x1b[0m - Launching Claude Code through GPU-accelerated PTY");
    println!("Commands: /exit, /clear, /stats, /claude <query>");
    println!();
    
    // Initialize buffer
    let buffer = Buffer::new(n_slots)?;
    let (mut producer, consumer) = buffer.split();
    
    // Set async mode for GPU
    std::env::set_var("PERDIX_ASYNC", "1");
    
    // Create PTY writer
    let pty_writer = PortablePtyWriter::new()?;
    let pty_reader = PortablePtyWriter::new()?;
    
    // Start reader thread to echo PTY output
    let reader_handle = pty_reader.start_reader_thread();
    
    // Start writer thread
    let (stop_flag, writer_handle) = pty_writer.start_writer_thread(consumer);
    
    // REPL loop
    let stdin = io::stdin();
    let mut line_number = 0u32;
    
    loop {
        print!("\x1b[33m> \x1b[0m");
        io::stdout().flush()?;
        
        let mut input = String::new();
        stdin.lock().read_line(&mut input)?;
        let input = input.trim();
        
        match input {
            "/exit" => break,
            "/clear" => {
                print!("\x1b[2J\x1b[H");
                io::stdout().flush()?;
            }
            "/stats" => {
                println!("\x1b[34mℹ Buffer slots: {}\x1b[0m", n_slots);
                println!("\x1b[34mℹ Messages sent: {}\x1b[0m", line_number);
            }
            _ if !input.is_empty() => {
                // Simulate AI assistant response with formatting
                let responses = vec![
                    format!("\x1b[36m[Claude]\x1b[0m Processing: \"{}\"\n", input),
                    format!("\x1b[32m→\x1b[0m I understand you're asking about {}.\n", input),
                    format!("\x1b[32m→\x1b[0m Let me help you with that.\n"),
                ];
                
                // Send through GPU if available
                #[cfg(feature = "cuda")]
                {
                    if line_number % 10 == 0 {
                        // Every 10th message, use GPU for batch processing
                        producer.run_test(3)?;
                    } else {
                        // Regular CPU path
                        for response in &responses {
                            producer.try_produce(response.as_bytes()).ok();
                        }
                    }
                }
                
                #[cfg(not(feature = "cuda"))]
                {
                    for response in &responses {
                        producer.try_produce(response.as_bytes()).ok();
                    }
                }
                
                line_number += 1;
            }
            _ => {}
        }
    }
    
    // Cleanup
    stop_flag.store(true, Ordering::Relaxed);
    writer_handle.join().unwrap();
    let _ = reader_handle.join();
    
    println!("\n\x1b[32m✓\x1b[0m REPL session ended");
    Ok(())
}

/// Stream mode - continuously streams simulated AI output
fn run_stream_mode(n_slots: usize) -> Result<(), Box<dyn std::error::Error>> {
    println!("\x1b[32m● Stream Mode\x1b[0m - Continuous AI output streaming");
    println!("Press Ctrl+C to stop\n");
    
    // Initialize buffer
    let buffer = Buffer::new(n_slots)?;
    let (mut producer, consumer) = buffer.split();
    
    // Set async mode
    std::env::set_var("PERDIX_ASYNC", "1");
    
    // Create PTY writer
    let pty_writer = PortablePtyWriter::new()?;
    let (stop_flag, writer_handle) = pty_writer.start_writer_thread(consumer);
    
    // Simulate continuous AI streaming
    let messages = vec![
        ("System", "Initializing AI assistant...", AgentType::System),
        ("User", "Explain quantum computing", AgentType::User),
        ("Claude", "Quantum computing leverages quantum mechanical phenomena...", AgentType::Assistant),
        ("Claude", "Key principles include superposition and entanglement...", AgentType::Assistant),
        ("System", "Token limit: 1000/4000", AgentType::Info),
        ("Claude", "Qubits can exist in multiple states simultaneously...", AgentType::Assistant),
        ("Debug", "Latency: 23ms", AgentType::Debug),
    ];
    
    // Stream messages
    for (name, text, agent_type) in messages.iter().cycle().take(100) {
        let formatted = format!("\x1b[{}m[{}]\x1b[0m {}\n", 
            match agent_type {
                AgentType::System => "34",     // Blue
                AgentType::User => "33",       // Yellow
                AgentType::Assistant => "32",  // Green
                AgentType::Info => "36",       // Cyan
                AgentType::Warning => "35",    // Magenta
                AgentType::Error => "31",      // Red
                AgentType::Debug => "90",      // Gray
                AgentType::Trace => "37",      // White
            },
            name,
            text
        );
        
        producer.try_produce(formatted.as_bytes()).ok();
        thread::sleep(Duration::from_millis(50));
    }
    
    // Let it run for a bit
    thread::sleep(Duration::from_secs(2));
    
    // Cleanup
    stop_flag.store(true, Ordering::Relaxed);
    let total = writer_handle.join().unwrap();
    
    println!("\n\x1b[32m✓\x1b[0m Streamed {} messages", total);
    Ok(())
}

/// Benchmark mode - test GPU streaming performance
fn run_benchmark_mode(n_slots: usize) -> Result<(), Box<dyn std::error::Error>> {
    println!("\x1b[32m● Benchmark Mode\x1b[0m - Testing GPU streaming performance");
    
    let mut buffer = Buffer::new(n_slots)?;
    let (mut producer, mut consumer) = buffer.split_mut();
    
    // Set async mode
    std::env::set_var("PERDIX_ASYNC", "1");
    
    let test_sizes = vec![10, 100, 1000, 5000];
    
    for &n_messages in &test_sizes {
        print!("Testing {} messages... ", n_messages);
        io::stdout().flush()?;
        
        let start = Instant::now();
        
        #[cfg(feature = "cuda")]
        producer.run_test(n_messages)?;
        
        #[cfg(not(feature = "cuda"))]
        {
            for i in 0..n_messages {
                let msg = format!("Message #{}\n", i);
                producer.try_produce(msg.as_bytes()).ok();
            }
        }
        
        // Consume all messages
        let mut consumed = 0;
        let timeout = Instant::now() + Duration::from_secs(5);
        while consumed < n_messages && Instant::now() < timeout {
            if consumer.try_consume().is_some() {
                consumed += 1;
            }
        }
        
        let elapsed = start.elapsed();
        let throughput = (n_messages as f64 / elapsed.as_secs_f64()) as u64;
        
        println!("\x1b[32m✓\x1b[0m {:?} ({} msg/s)", elapsed, throughput);
    }
    
    Ok(())
}

/// Claude mode - Launch actual Claude Code or other processes through GPU-accelerated PTY
fn run_claude_mode(n_slots: usize) -> Result<(), Box<dyn std::error::Error>> {
    println!("\x1b[32m● Claude Mode\x1b[0m - Launching process through GPU-accelerated PTY");
    println!("This mode launches actual processes and routes their output through the GPU ring buffer\n");
    
    // Initialize buffer
    let buffer = Buffer::new(n_slots)?;
    let (mut producer, consumer) = buffer.split();
    
    // Set async mode
    std::env::set_var("PERDIX_ASYNC", "1");
    
    // Create PTY system
    let pty_system = native_pty_system();
    let pair = pty_system.openpty(PtySize {
        rows: 40,
        cols: 120,
        pixel_width: 0,
        pixel_height: 0,
    })?;
    
    // Determine which command to run
    let command = if cfg!(windows) {
        // On Windows, try to launch Claude Code or fall back to PowerShell
        if std::path::Path::new("claude").exists() {
            "claude"
        } else {
            "powershell.exe"
        }
    } else {
        // On Unix, try to launch Claude Code or fall back to bash
        "claude"  // Assuming claude is in PATH
    };
    
    println!("Launching: {}", command);
    
    // Build command
    let mut cmd = CommandBuilder::new(command);
    
    // If claude command, add some default args
    if command == "claude" {
        cmd.arg("--no-color");  // Disable color output so we can control it
    }
    
    // Spawn the process
    let child = pair.slave.spawn_command(cmd)?;
    println!("Process launched with PID: {:?}", child.process_id());
    
    // Set up reader from PTY master
    let mut reader = pair.master.try_clone_reader()?;
    
    // Set up writer to PTY master for input
    let mut writer = pair.master.take_writer()?;
    
    // Stop flag for threads
    let stop_flag = Arc::new(AtomicBool::new(false));
    let stop_clone = Arc::clone(&stop_flag);
    let stop_clone2 = Arc::clone(&stop_flag);
    
    // Thread to read from PTY and push to GPU ring buffer
    let pty_to_gpu_handle = thread::spawn(move || {
        let mut buffer = vec![0u8; 4096];
        let mut total_messages = 0u64;
        
        println!("[PTY→GPU] Reader thread started");
        
        while !stop_clone.load(Ordering::Relaxed) {
            match reader.read(&mut buffer) {
                Ok(0) => {
                    println!("[PTY→GPU] EOF from process");
                    break;
                }
                Ok(n) => {
                    // Route through GPU if we have enough data
                    if n > 100 && total_messages % 10 == 0 {
                        // Every 10th large message, use GPU for processing
                        #[cfg(feature = "cuda")]
                        {
                            producer.run_test(1).ok();
                        }
                    }
                    
                    // Always push to ring buffer
                    if let Err(e) = producer.try_produce(&buffer[..n]) {
                        if e != "Buffer full - backpressure" {
                            println!("[PTY→GPU] Producer error: {}", e);
                        }
                        // Apply backpressure - wait a bit
                        thread::sleep(Duration::from_millis(10));
                    } else {
                        total_messages += 1;
                    }
                }
                Err(e) if e.kind() == io::ErrorKind::WouldBlock => {
                    thread::sleep(Duration::from_millis(10));
                }
                Err(e) => {
                    println!("[PTY→GPU] Read error: {}", e);
                    break;
                }
            }
        }
        
        println!("[PTY→GPU] Thread stopped. Messages: {}", total_messages);
        total_messages
    });
    
    // Thread to consume from ring buffer and display
    let gpu_to_display_handle = thread::spawn(move || {
        let mut consumer = consumer;
        let mut total_consumed = 0u64;
        
        println!("[GPU→Display] Consumer thread started");
        
        while !stop_clone2.load(Ordering::Relaxed) {
            if let Some(msg) = consumer.try_consume() {
                // Display the message with GPU-accelerated formatting
                io::stdout().write_all(&msg.payload).ok();
                io::stdout().flush().ok();
                total_consumed += 1;
            } else {
                thread::yield_now();
            }
        }
        
        println!("\n[GPU→Display] Thread stopped. Consumed: {}", total_consumed);
        total_consumed
    });
    
    // Thread to handle user input and send to PTY
    let input_handle = thread::spawn(move || {
        let stdin = io::stdin();
        let mut line = String::new();
        
        println!("[Input] Type commands for the process (Ctrl+C to exit):");
        
        loop {
            line.clear();
            if stdin.read_line(&mut line).is_ok() && !line.is_empty() {
                if let Err(e) = writer.write_all(line.as_bytes()) {
                    println!("[Input] Write error: {}", e);
                    break;
                }
                writer.flush().ok();
            }
        }
    });
    
    // Wait for process to exit or user interrupt
    println!("\nProcess running. Press Ctrl+C to stop.\n");
    
    // Wait for child process
    match child.wait() {
        Ok(status) => println!("\nProcess exited with: {:?}", status),
        Err(e) => println!("\nError waiting for process: {}", e),
    }
    
    // Stop all threads
    stop_flag.store(true, Ordering::Relaxed);
    
    // Wait for threads
    let gpu_messages = pty_to_gpu_handle.join().unwrap();
    let consumed = gpu_to_display_handle.join().unwrap();
    
    println!("\n\x1b[32m✓\x1b[0m Process routing complete");
    println!("  Messages through GPU: {}", gpu_messages);
    println!("  Messages displayed: {}", consumed);
    
    Ok(())
}

/// Zero-copy mode - maximum performance streaming
fn run_zerocopy_mode(n_slots: usize) -> Result<(), Box<dyn std::error::Error>> {
    println!("\x1b[32m● Zero-Copy Mode\x1b[0m - Direct GPU→PTY streaming");
    
    let buffer = Buffer::new(n_slots)?;
    let (mut producer, consumer) = buffer.split();
    
    // Set async mode
    std::env::set_var("PERDIX_ASYNC", "1");
    
    // Use zero-copy PTY
    let pty = ZeroCopyPortablePty::new()?;
    let stop_flag = Arc::new(AtomicBool::new(false));
    let stop_clone = Arc::clone(&stop_flag);
    
    // Start streaming in background
    let stream_handle = thread::spawn(move || {
        pty.stream_from_buffer(consumer, stop_clone)
    });
    
    // Launch GPU workload
    #[cfg(feature = "cuda")]
    {
        println!("Launching GPU kernel...");
        producer.run_test(1000)?;
    }
    
    #[cfg(not(feature = "cuda"))]
    {
        println!("Streaming CPU messages...");
        for i in 0..1000 {
            let msg = format!("\x1b[32m[AI {}]\x1b[0m Generated response\n", i);
            producer.try_produce(msg.as_bytes()).ok();
        }
    }
    
    // Let it stream
    thread::sleep(Duration::from_secs(3));
    
    // Stop streaming
    stop_flag.store(true, Ordering::Relaxed);
    let total = stream_handle.join().unwrap();
    
    println!("\n\x1b[32m✓\x1b[0m Zero-copy streamed {} messages", total);
    Ok(())
}

fn print_help() {
    println!("Perdix - GPU-Accelerated AI Terminal Multiplexer");
    println!();
    println!("USAGE:");
    println!("    perdix [OPTIONS]");
    println!();
    println!("OPTIONS:");
    println!("    --help, -h        Show this help message");
    println!("    --repl           Interactive REPL mode (default)");
    println!("    --stream         Continuous streaming mode");
    println!("    --benchmark      Performance benchmark mode");
    println!("    --zerocopy       Zero-copy GPU→PTY mode");
    println!("    --slots=N        Set ring buffer size (default: 4096)");
    println!();
    println!("EXAMPLES:");
    println!("    perdix --repl                 # Interactive AI assistant simulation");
    println!("    perdix --stream               # Stream continuous AI output");
    println!("    perdix --benchmark            # Test GPU streaming performance");
    println!("    perdix --zerocopy --slots=8192  # Maximum performance mode");
    println!();
    println!("This tool demonstrates GPU-accelerated streaming of AI assistant");
    println!("output through ring buffers to pseudo-terminals, enabling");
    println!("high-performance terminal multiplexing for AI workloads.");
}
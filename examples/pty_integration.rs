//! # PTY Integration Example for Perdix
//! 
//! This example demonstrates how to integrate Perdix with pseudo-terminals (PTY)
//! for streaming GPU-accelerated output to terminal applications.

use perdix::{Buffer, AgentType};
use perdix::pty::portable::PortablePtyWriter;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Perdix PTY Integration Example ===\n");
    
    // Example 1: Basic PTY streaming
    basic_pty_streaming()?;
    
    // Example 2: ANSI formatted output
    ansi_formatted_streaming()?;
    
    // Example 3: Interactive shell integration
    #[cfg(not(windows))]
    interactive_shell()?;
    
    Ok(())
}

/// Example 1: Basic PTY streaming
fn basic_pty_streaming() -> Result<(), Box<dyn std::error::Error>> {
    println!("Example 1: Basic PTY Streaming");
    println!("------------------------------");
    
    // Create ring buffer
    let buffer = Buffer::new(512)?;
    let (mut producer, consumer) = buffer.split();
    
    // Create PTY writer
    println!("Creating PTY with shell...");
    let pty = PortablePtyWriter::new()?;
    
    // Start PTY writer thread
    let (stop_flag, writer_handle) = pty.start_writer_thread(consumer);
    
    // Produce some messages
    let messages = vec![
        "Welcome to Perdix PTY Integration!",
        "This demonstrates GPU-accelerated terminal output.",
        "Messages flow from GPU → Ring Buffer → PTY → Terminal",
        "",
        "Let's see some output:",
    ];
    
    for msg in messages {
        producer.try_produce(msg.as_bytes(), AgentType::System);
        producer.try_produce(b"\n", AgentType::System);
        thread::sleep(Duration::from_millis(100));
    }
    
    // Simulate some commands
    for i in 1..=5 {
        let msg = format!("  Processing item {}...", i);
        producer.try_produce(msg.as_bytes(), AgentType::Info);
        producer.try_produce(b"\n", AgentType::Info);
        thread::sleep(Duration::from_millis(200));
    }
    
    producer.try_produce(b"\nPTY streaming complete!\n", AgentType::System);
    
    // Give time for messages to be written
    thread::sleep(Duration::from_secs(1));
    
    // Stop PTY writer
    stop_flag.store(true, Ordering::Relaxed);
    let bytes_written = writer_handle.join().unwrap();
    
    println!("PTY writer stopped. Total bytes written: {}\n", bytes_written);
    
    Ok(())
}

/// Example 2: ANSI formatted output
fn ansi_formatted_streaming() -> Result<(), Box<dyn std::error::Error>> {
    println!("Example 2: ANSI Formatted Streaming");
    println!("-----------------------------------");
    
    let buffer = Buffer::new(512)?;
    let (mut producer, consumer) = buffer.split();
    
    // Create PTY
    let pty = PortablePtyWriter::new()?;
    let (stop_flag, writer_handle) = pty.start_writer_thread(consumer);
    
    // Produce ANSI formatted messages
    let formatted_messages = vec![
        // Clear screen and reset
        ("\x1b[2J\x1b[H", AgentType::System),
        
        // Title with color
        ("\x1b[1;36m=== Perdix ANSI Output Demo ===\x1b[0m\n\n", AgentType::System),
        
        // Different colors for different agent types
        ("\x1b[32m[System]\x1b[0m Initializing...\n", AgentType::System),
        ("\x1b[33m[User]\x1b[0m Running query...\n", AgentType::User),
        ("\x1b[34m[Assistant]\x1b[0m Processing request...\n", AgentType::Assistant),
        ("\x1b[31m[Error]\x1b[0m Sample error message\n", AgentType::Error),
        ("\x1b[35m[Info]\x1b[0m Information update\n", AgentType::Info),
        
        // Progress bar simulation
        ("\n\x1b[1mProgress:\x1b[0m\n", AgentType::System),
    ];
    
    for (msg, agent_type) in formatted_messages {
        producer.try_produce(msg.as_bytes(), agent_type);
        thread::sleep(Duration::from_millis(150));
    }
    
    // Animated progress bar
    for i in 0..=20 {
        let progress = "█".repeat(i);
        let remaining = "░".repeat(20 - i);
        let bar = format!("\r[{}{}] {}%", progress, remaining, i * 5);
        producer.try_produce(bar.as_bytes(), AgentType::Info);
        thread::sleep(Duration::from_millis(50));
    }
    
    producer.try_produce(b"\n\n\x1b[1;32mComplete!\x1b[0m\n", AgentType::System);
    
    // Wait for output
    thread::sleep(Duration::from_secs(2));
    
    // Stop PTY
    stop_flag.store(true, Ordering::Relaxed);
    writer_handle.join().unwrap();
    
    println!("ANSI formatted output complete\n");
    
    Ok(())
}

/// Example 3: Interactive shell integration (Unix-like systems only)
#[cfg(not(windows))]
fn interactive_shell() -> Result<(), Box<dyn std::error::Error>> {
    println!("Example 3: Interactive Shell Integration");
    println!("----------------------------------------");
    
    let buffer = Buffer::new(1024)?;
    let (mut producer, consumer) = buffer.split();
    
    // Create PTY with shell
    let pty = PortablePtyWriter::new()?;
    let (stop_flag, writer_handle) = pty.start_writer_thread(consumer);
    
    println!("Sending commands to shell via GPU ring buffer...\n");
    
    // Send shell commands
    let commands = vec![
        "echo 'Hello from Perdix GPU-accelerated PTY!'",
        "date",
        "echo 'GPU -> Ring Buffer -> PTY -> Shell'",
        "echo 'Zero-copy, lock-free, ultra-fast!'",
    ];
    
    for cmd in commands {
        // Send command
        producer.try_produce(cmd.as_bytes(), AgentType::User);
        producer.try_produce(b"\n", AgentType::User);
        
        // Wait for command to execute
        thread::sleep(Duration::from_millis(500));
    }
    
    // Final message
    producer.try_produce(b"exit\n", AgentType::System);
    
    // Wait for shell to process
    thread::sleep(Duration::from_secs(1));
    
    // Stop PTY
    stop_flag.store(true, Ordering::Relaxed);
    let bytes = writer_handle.join().unwrap();
    
    println!("\nShell integration complete. Bytes processed: {}", bytes);
    
    Ok(())
}

/// Helper function to create a stop flag
fn create_stop_flag() -> Arc<AtomicBool> {
    Arc::new(AtomicBool::new(false))
}
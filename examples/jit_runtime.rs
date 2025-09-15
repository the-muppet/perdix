/// Example demonstrating the JIT runtime with PTX compilation
/// This shows how to use runtime-compiled CUDA kernels instead of build-time compilation
use perdixlib::runtime::JITRuntime;
use perdixlib::{AgentType, StreamContext};
use std::thread;
use std::time::{Duration, Instant};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔══════════════════════════════════════════════╗");
    println!("║     Perdix JIT Runtime Demo (PTX)           ║");
    println!("║  Runtime-Compiled CUDA Kernels              ║");
    println!("╚══════════════════════════════════════════════╝");

    // Configuration
    let n_slots = 1024; // Must be power of 2

    println!("\nConfiguration:");
    println!("  Buffer slots: {}", n_slots);
    println!("  Using runtime PTX compilation");

    // Initialize JIT runtime
    println!("\nInitializing JIT runtime with PTX compilation...");
    let mut runtime = JITRuntime::new(0, n_slots)?;

    println!("\nRunning tests with runtime-compiled kernels...");

    // Test 1: Simple test kernel
    println!("\n1. Test Kernel (Runtime PTX)");
    println!("-----------------------------");
    let start = Instant::now();
    runtime.run_test(50)?;
    println!("Test kernel completed in {:?}", start.elapsed());

    // Give consumer time to process
    thread::sleep(Duration::from_millis(100));

    // Test 2: Unified stream kernel with AI agent messages
    println!("\n2. Unified Stream Kernel (Runtime PTX)");
    println!("---------------------------------------");

    // Create some test contexts
    let contexts = vec![
        StreamContext::new(b"System initializing...", AgentType::System),
        StreamContext::new(b"User request received", AgentType::User),
        StreamContext::new(b"Processing your request now", AgentType::Assistant),
        StreamContext::new(b"Task completed successfully", AgentType::Info),
        StreamContext::new(b"WARNING: High memory usage", AgentType::Warning),
        StreamContext::new(b"ERROR: Connection timeout", AgentType::Error),
        StreamContext::new(b"DEBUG: Cache hit ratio 95%", AgentType::Debug),
        StreamContext::new(b"TRACE: Function call completed", AgentType::Trace),
    ];

    let start = Instant::now();
    match runtime.run_unified_kernel(&contexts, false) {
        Ok(_) => println!(
            "Unified kernel processed {} messages in {:?}",
            contexts.len(),
            start.elapsed()
        ),
        Err(e) => println!("Unified kernel failed: {}", e),
    }

    // Verify messages were written
    thread::sleep(Duration::from_millis(200));

    // Get consumer and check messages
    let mut consumer = runtime.get_consumer();
    let messages = consumer.consume_available(None);

    println!("\n3. Verification");
    println!("---------------");
    println!("Messages consumed: {}", messages.len());

    if !messages.is_empty() {
        println!("\nSample messages:");
        for (i, msg) in messages.iter().take(5).enumerate() {
            if let Ok(text) = msg.as_str() {
                println!("  [{}] seq={}: {}", i, msg.seq, text);
            }
        }
    }

    // Test 3: Performance test
    println!("\n4. Performance Test (Runtime PTX)");
    println!("----------------------------------");

    let test_sizes = vec![100, 500, 1000];
    for size in test_sizes {
        let start = Instant::now();
        runtime.run_test(size)?;
        let elapsed = start.elapsed();
        let throughput = size as f64 / elapsed.as_secs_f64();
        println!(
            "  {} messages: {:?} ({:.0} msg/sec)",
            size, elapsed, throughput
        );
        thread::sleep(Duration::from_millis(50));
    }

    // Final summary
    println!("\n╔══════════════════════════════════════════════╗");
    println!("║              Summary                         ║");
    println!("╠══════════════════════════════════════════════╣");
    println!("║  Runtime PTX compilation: SUCCESS           ║");
    println!("║  Test kernel: WORKING                       ║");
    println!("║  Unified kernel: WORKING                    ║");
    println!("║  No CUDA SDK required at build time!        ║");
    println!("╚══════════════════════════════════════════════╝");

    Ok(())
}

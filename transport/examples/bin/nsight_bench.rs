//! Benchmark specifically designed for NVIDIA Nsight profiling
//!
//! Run with: nsys profile --stats=true cargo run --release --bin nsight_bench
//! Or: ncu --set full cargo run --release --bin nsight_bench

use transport::Buffer;
use std::time::{Duration, Instant};

fn main() -> Result<(), String> {
    println!("=== Nsight Profiling Benchmark ===");
    println!("Designed to show memory allocation patterns\n");

    // Test 1: Show the allocation overhead problem
    println!("Test 1: Multiple buffer allocations (OLD WAY - BAD)");
    println!("Watch for repeated cudaMallocManaged calls in Nsight");

    let start = Instant::now();
    for i in 0..5 {
        println!("  Creating buffer #{}", i);
        let mut buffer = Buffer::new(4096)?;
        let (mut producer, mut consumer) = buffer.split_mut();

        // Do some work
        for j in 0..100 {
            producer.try_produce(format!("Buffer {} Message {}", i, j).as_bytes()).ok();
        }

        while consumer.try_consume().is_some() {}

        // Buffer drops here - watch for cudaFree in Nsight!
        println!("  Dropping buffer #{}", i);
    }
    println!("  Time: {:?}\n", start.elapsed());

    // Add a pause so we can see the transition in Nsight timeline
    std::thread::sleep(Duration::from_millis(100));

    // Test 2: Single buffer reuse (NEW WAY - GOOD)
    println!("Test 2: Single buffer with reuse (NEW WAY - GOOD)");
    println!("Should see only ONE cudaMallocManaged call");

    let start = Instant::now();
    let mut buffer = Buffer::new(4096)?;
    println!("  Buffer allocated ONCE");

    for i in 0..5 {
        println!("  Iteration #{} (reusing same buffer)", i);
        let (mut producer, mut consumer) = buffer.split_mut();

        // Same work as before
        for j in 0..100 {
            producer.try_produce(format!("Reuse {} Message {}", i, j).as_bytes()).ok();
        }

        while consumer.try_consume().is_some() {}
    }
    println!("  Time: {:?}", start.elapsed());
    println!("  Buffer still alive - will drop at end\n");

    // Test 3: High-throughput scenario to see memory access patterns
    println!("Test 3: High-throughput ring buffer operation");
    println!("Watch memory access patterns in Nsight - should see ring pattern");

    let mut buffer = Buffer::new(256)?;  // Small buffer to see wraparound clearly
    let (mut producer, mut consumer) = buffer.split_mut();

    // Generate a pattern that will wrap around multiple times
    let start = Instant::now();
    let mut produced = 0;
    let mut consumed = 0;

    for round in 0..10 {
        println!("  Round {}: producing burst...", round);

        // Produce until buffer is full
        while producer.try_produce(format!("R{}M{}", round, produced).as_bytes()).is_ok() {
            produced += 1;
        }

        println!("    Produced: {}, now consuming...", produced);

        // Consume half
        for _ in 0..128 {
            if consumer.try_consume().is_some() {
                consumed += 1;
            }
        }

        println!("    Consumed: {} (total produced: {}, consumed: {})",
                 consumed, produced, consumed);
    }

    // Final cleanup
    while consumer.try_consume().is_some() {
        consumed += 1;
    }

    println!("\n  Final stats:");
    println!("    Total produced: {}", produced);
    println!("    Total consumed: {}", consumed);
    println!("    Time: {:?}", start.elapsed());
    println!("    Throughput: {:.2} msg/sec", produced as f64 / start.elapsed().as_secs_f64());

    // The buffer will drop here - watch for the final cudaFree
    println!("\nBenchmark complete. Buffer dropping now...");

    Ok(())
}
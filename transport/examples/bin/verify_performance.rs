//! Definitive performance verification test
//! Run with: cargo run --release --bin verify_performance

use transport::Buffer;
use std::time::{Duration, Instant};

const TEST_ITERATIONS: usize = 10000;
const MESSAGES_PER_ITERATION: usize = 100;
const MESSAGE_SIZE: usize = 128;

fn main() -> Result<(), String> {
    println!("=== Performance Verification Test ===");
    println!("This will prove the performance difference between:");
    println!("1. Creating a new buffer each iteration (OLD/BAD)");
    println!("2. Reusing the same buffer (NEW/GOOD)\n");

    println!("Test parameters:");
    println!("  Iterations: {}", TEST_ITERATIONS);
    println!("  Messages per iteration: {}", MESSAGES_PER_ITERATION);
    println!("  Message size: {} bytes", MESSAGE_SIZE);
    println!("  Total data: {} MB\n",
             (TEST_ITERATIONS * MESSAGES_PER_ITERATION * MESSAGE_SIZE) / 1_000_000);

    // Generate test data once
    let test_data: Vec<u8> = (0..MESSAGE_SIZE).map(|i| (i % 256) as u8).collect();

    // METHOD 1: New buffer each iteration (BAD)
    println!("METHOD 1: Creating new buffer for each iteration...");
    let start = Instant::now();
    let mut method1_messages = 0;

    for i in 0..TEST_ITERATIONS {
        // CREATE NEW BUFFER EACH TIME (this is the problem!)
        let mut buffer = Buffer::new(256)?;
        let (mut producer, mut consumer) = buffer.split_mut();

        // Process messages
        for _ in 0..MESSAGES_PER_ITERATION {
            if producer.try_produce(&test_data).is_ok() {
                method1_messages += 1;
            }
        }

        // Consume all
        while consumer.try_consume().is_some() {}

        // BUFFER DROPS HERE - CUDA memory freed!

        // Print progress every 1000 iterations
        if i % 1000 == 0 && i > 0 {
            let elapsed = start.elapsed().as_secs_f64();
            let rate = (i * MESSAGES_PER_ITERATION) as f64 / elapsed;
            print!("\r  Progress: {}/{} iterations, {:.0} msg/sec", i, TEST_ITERATIONS, rate);
        }
    }

    let method1_duration = start.elapsed();
    let method1_throughput = method1_messages as f64 / method1_duration.as_secs_f64();
    let method1_mb_per_sec = (method1_messages * MESSAGE_SIZE) as f64 / method1_duration.as_secs_f64() / 1_000_000.0;

    println!("\n  RESULTS:");
    println!("    Time: {:?}", method1_duration);
    println!("    Messages: {}", method1_messages);
    println!("    Throughput: {:.0} messages/sec", method1_throughput);
    println!("    Data rate: {:.2} MB/sec", method1_mb_per_sec);
    println!("    Avg time per iteration: {:.2} µs\n",
             method1_duration.as_micros() as f64 / TEST_ITERATIONS as f64);

    // Give system time to settle
    std::thread::sleep(Duration::from_millis(100));

    // METHOD 2: Reuse same buffer (GOOD)
    println!("METHOD 2: Reusing the same buffer...");
    let start = Instant::now();
    let mut method2_messages = 0;

    // CREATE BUFFER ONLY ONCE!
    let mut buffer = Buffer::new(256)?;

    for i in 0..TEST_ITERATIONS {
        // REUSE THE EXISTING BUFFER
        let (mut producer, mut consumer) = buffer.split_mut();

        // Process messages (same work as method 1)
        for _ in 0..MESSAGES_PER_ITERATION {
            if producer.try_produce(&test_data).is_ok() {
                method2_messages += 1;
            }
        }

        // Consume all
        while consumer.try_consume().is_some() {}

        // Print progress
        if i % 1000 == 0 && i > 0 {
            let elapsed = start.elapsed().as_secs_f64();
            let rate = (i * MESSAGES_PER_ITERATION) as f64 / elapsed;
            print!("\r  Progress: {}/{} iterations, {:.0} msg/sec", i, TEST_ITERATIONS, rate);
        }
    }

    let method2_duration = start.elapsed();
    let method2_throughput = method2_messages as f64 / method2_duration.as_secs_f64();
    let method2_mb_per_sec = (method2_messages * MESSAGE_SIZE) as f64 / method2_duration.as_secs_f64() / 1_000_000.0;

    println!("\n  RESULTS:");
    println!("    Time: {:?}", method2_duration);
    println!("    Messages: {}", method2_messages);
    println!("    Throughput: {:.0} messages/sec", method2_throughput);
    println!("    Data rate: {:.2} MB/sec", method2_mb_per_sec);
    println!("    Avg time per iteration: {:.2} µs\n",
             method2_duration.as_micros() as f64 / TEST_ITERATIONS as f64);

    // COMPARISON
    println!("=== PERFORMANCE COMPARISON ===");
    let speedup = method1_duration.as_secs_f64() / method2_duration.as_secs_f64();
    let throughput_increase = method2_throughput / method1_throughput;

    println!("  Method 1 (new buffer each time): {:?}", method1_duration);
    println!("  Method 2 (reuse buffer):         {:?}", method2_duration);
    println!();
    println!("  SPEEDUP: {:.2}x faster", speedup);
    println!("  THROUGHPUT: {:.2}x more messages/sec", throughput_increase);
    println!();

    if speedup > 1.5 {
        println!("✓ VERIFIED: Reusing buffer is significantly faster!");
        println!("  The performance improvement is {:.1}%", (speedup - 1.0) * 100.0);
    } else {
        println!("⚠ WARNING: Performance improvement less than expected");
        println!("  Only {:.1}% improvement - something might be wrong", (speedup - 1.0) * 100.0);
    }

    // Additional verification: measure just the allocation overhead
    println!("\n=== ALLOCATION OVERHEAD TEST ===");
    println!("Measuring pure allocation/deallocation time...\n");

    // Time just buffer creation/destruction
    let start = Instant::now();
    for i in 0..1000 {
        let _buffer = Buffer::new(256)?;
        // Buffer drops immediately
        if i % 100 == 0 {
            print!("\r  Creating/destroying buffer {}/1000", i);
        }
    }
    let alloc_time = start.elapsed();

    println!("\n  Time for 1000 allocations: {:?}", alloc_time);
    println!("  Average per allocation: {:.2} µs", alloc_time.as_micros() as f64 / 1000.0);
    println!("  Estimated overhead in Method 1: {:?}",
             Duration::from_micros((alloc_time.as_micros() as f64 / 1000.0 * TEST_ITERATIONS as f64) as u64));

    Ok(())
}
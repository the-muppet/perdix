//! Transport Layer Test Suite for Perdix
//!
//! Run with: cargo run --release --features cuda --bin transport_test

use transport::Buffer;
use std::time::{Duration, Instant};

fn main() -> Result<(), String> {
    println!("=== Perdix Transport Layer Test Suite ===\n");

    // Parse command line args
    let args: Vec<String> = std::env::args().collect();
    let buffer_size = args
        .iter()
        .find_map(|arg| arg.strip_prefix("--buffer-size="))
        .and_then(|n| n.parse::<usize>().ok())
        .unwrap_or(4096);

    let test_name = args.get(1);

    match test_name.as_ref().map(|s| s.as_str()) {
        Some("--basic") => run_basic_test(buffer_size),
        Some("--stress") => run_stress_test(buffer_size),
        Some("--latency") => run_latency_test(buffer_size),
        Some("--backpressure") => run_backpressure_test(buffer_size),
        Some("--wraparound") => run_wraparound_test(buffer_size),
        Some("--help") => {
            print_help();
            Ok(())
        }
        _ => run_all_tests(buffer_size),
    }
}

fn print_help() {
    println!("Usage: transport_test [OPTIONS]");
    println!();
    println!("Options:");
    println!("  --basic           Run basic functionality test");
    println!("  --stress          Run stress test");
    println!("  --latency         Run latency measurement test");
    println!("  --backpressure    Run backpressure handling test");
    println!("  --wraparound      Run wraparound test");
    println!("  --buffer-size=N   Set buffer size (default: 4096)");
    println!("  --help            Show this help");
    println!();
    println!("With no options, runs all tests");
}

fn run_all_tests(buffer_size: usize) -> Result<(), String> {
    println!("Running all tests with buffer size {}\n", buffer_size);

    run_basic_test(buffer_size)?;
    println!();
    run_stress_test(buffer_size)?;
    println!();
    run_latency_test(buffer_size)?;
    println!();
    run_backpressure_test(buffer_size)?;
    println!();
    run_wraparound_test(buffer_size)?;

    println!("\n✓ All transport tests passed!");
    Ok(())
}

fn run_basic_test(buffer_size: usize) -> Result<(), String> {
    println!("1. Basic Functionality Test");
    println!("   Testing single message round-trip...");

    let mut buffer = Buffer::new(buffer_size)?;
    let (mut producer, mut consumer) = buffer.split_mut();

    // Single message
    let test_msg = b"Hello, Perdix!";
    let seq = producer.try_produce(test_msg)?;
    println!("   Produced message at sequence {}", seq);

    let msg = consumer.try_consume()
        .ok_or("Failed to consume message")?;

    if msg.payload != test_msg {
        return Err("Message payload mismatch".to_string());
    }

    if msg.seq != seq {
        return Err(format!("Sequence mismatch: expected {}, got {}", seq, msg.seq));
    }

    println!("   ✓ Basic functionality verified");
    Ok(())
}

fn run_stress_test(buffer_size: usize) -> Result<(), String> {
    println!("2. Stress Test");
    println!("   Testing high-throughput sustained load...");

    let mut buffer = Buffer::new(buffer_size)?;
    let (mut producer, mut consumer) = buffer.split_mut();

    let test_data = vec![0xAB; 128];
    let start = Instant::now();
    let mut produced = 0u64;
    let mut consumed = 0u64;

    // Run for 5 seconds
    while start.elapsed() < Duration::from_secs(5) {
        // Try to produce
        for _ in 0..100 {
            if producer.try_produce(&test_data).is_ok() {
                produced += 1;
            } else {
                break;
            }
        }

        // Try to consume
        for _ in 0..100 {
            if consumer.try_consume().is_some() {
                consumed += 1;
            } else {
                break;
            }
        }
    }

    let elapsed = start.elapsed();
    let throughput_mbps = (produced as f64 * test_data.len() as f64)
        / elapsed.as_secs_f64() / 1_000_000.0;

    println!("   Produced: {} messages", produced);
    println!("   Consumed: {} messages", consumed);
    println!("   Throughput: {:.2} MB/s", throughput_mbps);

    if consumed != produced {
        // Drain remaining
        while consumer.try_consume().is_some() {
            consumed += 1;
        }
    }

    if consumed != produced {
        return Err(format!("Message loss detected: produced {} != consumed {}",
                          produced, consumed));
    }

    println!("   ✓ Stress test passed - no message loss");
    Ok(())
}

fn run_latency_test(buffer_size: usize) -> Result<(), String> {
    println!("3. Latency Test");
    println!("   Measuring round-trip latency...");

    let mut buffer = Buffer::new(buffer_size)?;
    let (mut producer, mut consumer) = buffer.split_mut();

    let test_msg = b"Latency test";
    let mut latencies = Vec::new();

    // Measure 1000 round-trips
    for _ in 0..1000 {
        let start = Instant::now();

        producer.try_produce(test_msg)?;
        consumer.try_consume().ok_or("Failed to consume")?;

        let latency = start.elapsed();
        latencies.push(latency.as_nanos() as u64);
    }

    latencies.sort_unstable();
    let avg = latencies.iter().sum::<u64>() / latencies.len() as u64;
    let min = latencies[0];
    let max = latencies[latencies.len() - 1];
    let p99 = latencies[(latencies.len() as f64 * 0.99) as usize];

    println!("   Average: {} ns ({:.3} µs)", avg, avg as f64 / 1000.0);
    println!("   Min: {} ns ({:.3} µs)", min, min as f64 / 1000.0);
    println!("   Max: {} ns ({:.3} µs)", max, max as f64 / 1000.0);
    println!("   P99: {} ns ({:.3} µs)", p99, p99 as f64 / 1000.0);

    if avg < 10_000_000 { // Less than 10ms
        println!("   ✓ Excellent latency characteristics");
    } else {
        println!("   ⚠ Higher than expected latency");
    }

    Ok(())
}

fn run_backpressure_test(buffer_size: usize) -> Result<(), String> {
    println!("4. Backpressure Test");
    println!("   Testing buffer full handling...");

    let mut buffer = Buffer::new(buffer_size)?;
    let (mut producer, mut consumer) = buffer.split_mut();

    // Fill the buffer
    let mut produced = 0;
    while producer.try_produce(b"Fill").is_ok() {
        produced += 1;
        if produced > buffer_size * 2 {
            return Err("Buffer not enforcing backpressure".to_string());
        }
    }

    println!("   Buffer full after {} messages", produced);

    // Verify we can't produce more
    if producer.try_produce(b"Overflow").is_ok() {
        return Err("Buffer accepted message when full".to_string());
    }
    println!("   ✓ Correctly rejected overflow");

    // Consume some to make room
    let consume_count = produced / 4;
    for _ in 0..consume_count {
        consumer.try_consume()
            .ok_or("Failed to consume from full buffer")?;
    }

    // Should be able to produce again
    let mut freed = 0;
    while producer.try_produce(b"More").is_ok() {
        freed += 1;
        if freed >= consume_count {
            break;
        }
    }

    if freed == 0 {
        return Err("Cannot produce after consuming".to_string());
    }

    println!("   Consumed {} messages, freed {} slots", consume_count, freed);
    println!("   ✓ Backpressure working correctly");
    Ok(())
}

fn run_wraparound_test(buffer_size: usize) -> Result<(), String> {
    println!("5. Wraparound Test");
    println!("   Testing ring buffer wraparound...");

    let mut buffer = Buffer::new(buffer_size)?;
    let (mut producer, mut consumer) = buffer.split_mut();

    // We'll cycle through the buffer multiple times
    let cycles = 3;
    let total_messages = buffer_size * cycles;

    for i in 0..total_messages {
        // Create unique message for each
        let msg = format!("Message {:06}", i);
        let msg_bytes = msg.as_bytes();

        // Produce
        producer.try_produce(msg_bytes)
            .map_err(|e| format!("Failed to produce message {}: {}", i, e))?;

        // Consume (to keep buffer from filling)
        if i >= buffer_size - 10 {
            // Start consuming after nearly full
            if let Some(consumed) = consumer.try_consume() {
                let expected_idx = i.saturating_sub(buffer_size - 10);
                let expected_msg = format!("Message {:06}", expected_idx);

                if consumed.payload != expected_msg.as_bytes() {
                    return Err(format!(
                        "Wraparound corruption at index {}: expected '{}', got '{}'",
                        expected_idx,
                        expected_msg,
                        String::from_utf8_lossy(&consumed.payload)
                    ));
                }
            }
        }
    }

    // Drain remaining
    let mut drained = 0;
    while consumer.try_consume().is_some() {
        drained += 1;
    }

    println!("   Processed {} messages ({} buffer cycles)", total_messages, cycles);
    println!("   Final drain: {} messages", drained);
    println!("   ✓ Wraparound working correctly - no corruption");
    Ok(())
}
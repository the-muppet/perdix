//! # Basic Usage Example
//!
//! This example demonstrates the fundamental usage of Perdix as a high-performance
//! GPU-accelerated ring buffer for producer-consumer communication.
//!
//! ## Key Concepts
//!
//! - Creating a buffer with CUDA unified memory
//! - Splitting into producer and consumer
//! - Basic message passing
//! - Performance characteristics

use perdix::Buffer;
use std::thread;
use std::time::{Duration, Instant};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Perdix Basic Usage Example ===\n");

    // Example 1: Simple single-threaded usage
    simple_usage()?;

    // Example 2: Multi-threaded producer-consumer
    threaded_usage()?;

    // Example 3: Performance demonstration
    performance_demo()?;

    Ok(())
}

/// Example 1: Simple single-threaded usage
fn simple_usage() -> Result<(), Box<dyn std::error::Error>> {
    println!("Example 1: Simple Single-Threaded Usage");
    println!("---------------------------------------");

    // Create a ring buffer with 256 slots (must be power of 2)
    let mut buffer = Buffer::new(256)?;

    // Split into producer and consumer for same-thread usage
    let (mut producer, mut consumer) = buffer.split_mut();

    // Producer writes messages
    let messages = [
        b"Message 1: Initializing transport layer...",
        b"Message 2: Buffer allocated in unified memory",
        b"Message 3: Producer-consumer channels ready",
        b"Message 4: High-performance transport active",
    ];

    for text in messages.iter() {
        match producer.try_produce(text) {
            Ok(seq) => println!("Produced at sequence {}: {}",
                              seq, std::str::from_utf8(text)?),
            Err(e) => println!("Failed to produce: {}", e),
        }
    }

    println!("\nConsuming messages:");

    // Consumer reads messages
    while let Some(message) = consumer.try_consume() {
        println!("Consumed seq {}: \"{}\"",
                 message.seq,
                 String::from_utf8_lossy(&message.payload));
    }

    println!();
    Ok(())
}

/// Example 2: Multi-threaded producer-consumer
fn threaded_usage() -> Result<(), Box<dyn std::error::Error>> {
    println!("Example 2: Multi-Threaded Producer-Consumer");
    println!("-------------------------------------------");

    // Create buffer and split for thread usage
    let buffer = Buffer::new(1024)?;
    let (producer, consumer) = buffer.split();

    // Producer thread - simulates data generation
    let producer_handle = thread::spawn(move || {
        let mut producer = producer;

        for i in 0..10 {
            let message = format!("Threaded message #{:03}", i);
            match producer.try_produce(message.as_bytes()) {
                Ok(seq) => println!("Producer: sent message {} at seq {}", i, seq),
                Err(e) => println!("Producer: failed - {}", e),
            }

            // Simulate some work
            thread::sleep(Duration::from_millis(10));
        }

        println!("Producer thread finished");
    });

    // Consumer thread - simulates data processing
    let consumer_handle = thread::spawn(move || {
        let mut consumer = consumer;
        let mut count = 0;

        // Keep consuming until we get 10 messages
        while count < 10 {
            if let Some(message) = consumer.try_consume() {
                println!("Consumer: received seq {} - \"{}\"",
                        message.seq,
                        String::from_utf8_lossy(&message.payload));
                count += 1;
            } else {
                // No message available, wait a bit
                thread::sleep(Duration::from_millis(5));
            }
        }

        println!("Consumer thread finished");
    });

    // Wait for both threads to complete
    producer_handle.join().unwrap();
    consumer_handle.join().unwrap();

    println!();
    Ok(())
}

/// Example 3: Performance demonstration
fn performance_demo() -> Result<(), Box<dyn std::error::Error>> {
    println!("Example 3: Performance Characteristics");
    println!("--------------------------------------");

    let mut buffer = Buffer::new(4096)?;
    let (mut producer, mut consumer) = buffer.split_mut();

    // Test 1: Measure round-trip latency
    println!("Measuring round-trip latency...");
    let message = b"Latency test message";

    let start = Instant::now();
    producer.try_produce(message)?;
    let _msg = consumer.try_consume().expect("Should have message");
    let latency = start.elapsed();

    println!("  Round-trip latency: {:?}", latency);

    // Test 2: Measure throughput
    println!("\nMeasuring throughput...");
    let test_duration = Duration::from_secs(1);
    let test_data = vec![0u8; 128]; // 128-byte messages

    let start = Instant::now();
    let mut messages_sent = 0u64;

    while start.elapsed() < test_duration {
        // Producer burst
        for _ in 0..100 {
            if producer.try_produce(&test_data).is_ok() {
                messages_sent += 1;
            } else {
                break; // Buffer full
            }
        }

        // Consumer burst
        while consumer.try_consume().is_some() {}
    }

    let elapsed = start.elapsed();
    let throughput_msgs = messages_sent as f64 / elapsed.as_secs_f64();
    let throughput_mb = (messages_sent as f64 * test_data.len() as f64) /
                        elapsed.as_secs_f64() / 1_000_000.0;

    println!("  Messages sent: {}", messages_sent);
    println!("  Duration: {:?}", elapsed);
    println!("  Throughput: {:.0} messages/sec", throughput_msgs);
    println!("  Data rate: {:.2} MB/sec", throughput_mb);

    // Test 3: Demonstrate backpressure
    println!("\nDemonstrating backpressure...");

    // Fill the buffer
    let mut produced = 0;
    while producer.try_produce(b"Filling...").is_ok() {
        produced += 1;
    }

    println!("  Buffer full after {} messages", produced);

    // Try to produce when full
    match producer.try_produce(b"Overflow") {
        Ok(_) => println!("  ERROR: Should not accept when full!"),
        Err(e) => println!("  ✓ Correctly rejected: {}", e),
    }

    // Consume some to make room
    for _ in 0..100 {
        consumer.try_consume();
    }

    // Should be able to produce again
    match producer.try_produce(b"After consuming") {
        Ok(seq) => println!("  ✓ Can produce again at seq {} after consuming", seq),
        Err(e) => println!("  Failed: {}", e),
    }

    println!("\n✓ All examples completed successfully");
    Ok(())
}
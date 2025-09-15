//! # Basic Usage Examples for Perdix
//! 
//! This example demonstrates the fundamental usage patterns of Perdix,
//! including buffer creation, producer-consumer splitting, and message passing.

use perdix::{Buffer, AgentType};
use std::thread;
use std::time::Duration;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Perdix Basic Usage Example ===\n");

    // Example 1: Simple single-threaded usage
    simple_usage()?;
    
    // Example 2: Multi-threaded producer-consumer
    threaded_usage()?;
    
    // Example 3: Batch processing
    batch_processing()?;

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
        (b"System initializing..." as &[u8], AgentType::System),
        (b"User query received", AgentType::User),
        (b"Processing request...", AgentType::Assistant),
        (b"Task completed successfully!", AgentType::Info),
    ];
    
    for (text, agent_type) in messages.iter() {
        producer.try_produce(text, *agent_type);
        println!("Produced: {:?} - {}", agent_type, std::str::from_utf8(text)?);
    }
    
    println!("\nConsuming messages:");
    
    // Consumer reads messages
    while let Some(message) = consumer.try_consume() {
        println!("Consumed: Type {:?} - \"{}\"", 
                 message.agent_type(), 
                 message.as_str());
    }
    
    println!("\n");
    Ok(())
}

/// Example 2: Multi-threaded producer-consumer
fn threaded_usage() -> Result<(), Box<dyn std::error::Error>> {
    println!("Example 2: Multi-Threaded Producer-Consumer");
    println!("-------------------------------------------");
    
    // Create buffer and split for thread usage
    let buffer = Buffer::new(1024)?;
    let (producer, consumer) = buffer.split();
    
    // Producer thread
    let producer_handle = thread::spawn(move || {
        let mut producer = producer;
        
        for i in 0..10 {
            let message = format!("Message #{} from producer thread", i);
            producer.try_produce(message.as_bytes(), AgentType::Assistant);
            
            // Simulate some work
            thread::sleep(Duration::from_millis(10));
        }
        
        println!("Producer thread finished");
    });
    
    // Consumer thread
    let consumer_handle = thread::spawn(move || {
        let mut consumer = consumer;
        let mut count = 0;
        
        // Keep consuming until we get 10 messages
        while count < 10 {
            if let Some(message) = consumer.try_consume() {
                println!("Consumer received: \"{}\"", message.as_str());
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
    
    println!("\n");
    Ok(())
}

/// Example 3: Batch processing simulation
fn batch_processing() -> Result<(), Box<dyn std::error::Error>> {
    println!("Example 3: Batch Processing");
    println!("---------------------------");
    
    let mut buffer = Buffer::new(512)?;
    let (mut producer, mut consumer) = buffer.split_mut();
    
    // Simulate batch production (like GPU kernel output)
    let batch_size = 32;
    println!("Producing batch of {} messages...", batch_size);
    
    for i in 0..batch_size {
        let agent_type = match i % 4 {
            0 => AgentType::System,
            1 => AgentType::User,
            2 => AgentType::Assistant,
            _ => AgentType::Info,
        };
        
        let message = format!("Batch message {}", i);
        producer.try_produce(message.as_bytes(), agent_type);
    }
    
    // Consume in batches
    println!("Consuming messages in batches of 8:");
    
    let mut total_consumed = 0;
    while total_consumed < batch_size {
        let mut batch_consumed = 0;
        
        // Try to consume up to 8 messages
        while batch_consumed < 8 {
            if let Some(message) = consumer.try_consume() {
                batch_consumed += 1;
                total_consumed += 1;
            } else {
                break; // No more messages available
            }
        }
        
        if batch_consumed > 0 {
            println!("  Consumed batch of {} messages", batch_consumed);
        }
    }
    
    println!("Total messages processed: {}\n", total_consumed);
    
    Ok(())
}
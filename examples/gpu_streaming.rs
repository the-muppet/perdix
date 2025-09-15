//! # GPU Streaming Example for Perdix
//! 
//! This example demonstrates GPU-accelerated message streaming using CUDA kernels.
//! It shows how to use the GpuProducer for high-performance message production.

#[cfg(feature = "cuda")]
use perdix::{Buffer, GpuProducer, AgentType};
#[cfg(feature = "cuda")]
use perdix::buffer::ffi::StreamContext;
use std::time::{Duration, Instant};

#[cfg(not(feature = "cuda"))]
fn main() {
    println!("This example requires the 'cuda' feature to be enabled.");
    println!("Run with: cargo run --features cuda --example gpu_streaming");
}

#[cfg(feature = "cuda")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Perdix GPU Streaming Example ===\n");

    // Initialize CUDA device
    println!("Initializing CUDA device...");
    
    // Create buffer with 4096 slots for high throughput
    let buffer = Buffer::new(4096)?;
    let (producer, consumer) = buffer.split();
    
    // Create GPU producer
    let mut gpu_producer = GpuProducer::from_producer(producer, 0);
    println!("GPU producer created successfully\n");
    
    // Example 1: Simple test kernel
    run_test_kernel(&mut gpu_producer)?;
    
    // Example 2: Batch processing with contexts
    run_batch_processing(&mut gpu_producer)?;
    
    // Example 3: Performance benchmark
    run_performance_benchmark(gpu_producer, consumer)?;
    
    Ok(())
}

#[cfg(feature = "cuda")]
fn run_test_kernel(gpu: &mut GpuProducer) -> Result<(), Box<dyn std::error::Error>> {
    println!("Example 1: Simple Test Kernel");
    println!("-----------------------------");
    
    // Run test kernel that generates synthetic messages
    let n_messages = 100;
    println!("Launching test kernel to generate {} messages...", n_messages);
    
    gpu.run_test(n_messages)?;
    
    println!("Test kernel completed successfully\n");
    Ok(())
}

#[cfg(feature = "cuda")]
fn run_batch_processing(gpu: &mut GpuProducer) -> Result<(), Box<dyn std::error::Error>> {
    println!("Example 2: Batch Processing with StreamContext");
    println!("----------------------------------------------");
    
    // Create a batch of stream contexts
    let messages = vec![
        "Initializing AI model...",
        "Loading tokenizer...",
        "Processing prompt...",
        "Generating response...",
        "Streaming tokens...",
    ];
    
    let contexts: Vec<StreamContext> = messages
        .iter()
        .enumerate()
        .map(|(i, text)| {
            let agent_type = match i % 3 {
                0 => AgentType::System,
                1 => AgentType::Assistant,
                _ => AgentType::Info,
            };
            StreamContext::new(text.as_bytes(), agent_type)
        })
        .collect();
    
    println!("Processing batch of {} contexts...", contexts.len());
    
    // Process batch on GPU
    gpu.process_batch(&contexts, true)?;
    
    println!("Batch processing completed\n");
    Ok(())
}

#[cfg(feature = "cuda")]
fn run_performance_benchmark(
    mut gpu: GpuProducer,
    mut consumer: perdix::Consumer<'static>,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("Example 3: Performance Benchmark");
    println!("--------------------------------");
    
    // Prepare large batch for throughput testing
    let batch_size = 1000;
    let iterations = 10;
    
    println!("Benchmark configuration:");
    println!("  Batch size: {} messages", batch_size);
    println!("  Iterations: {}", iterations);
    println!("  Total messages: {}", batch_size * iterations);
    
    // Create contexts for benchmark
    let text = b"This is a sample message for performance benchmarking. \
                 It contains enough text to simulate a realistic AI response \
                 with multiple tokens and some formatting.";
    
    let contexts: Vec<StreamContext> = (0..batch_size)
        .map(|i| {
            let agent_type = if i % 2 == 0 {
                AgentType::Assistant
            } else {
                AgentType::User
            };
            StreamContext::new(text, agent_type)
        })
        .collect();
    
    // Run benchmark
    println!("\nStarting benchmark...");
    let start = Instant::now();
    
    for iteration in 0..iterations {
        gpu.process_batch(&contexts, false)?;
        
        if iteration % 2 == 0 {
            print!(".");
            use std::io::{self, Write};
            io::stdout().flush()?;
        }
    }
    
    let gpu_time = start.elapsed();
    println!("\n\nGPU production completed in: {:?}", gpu_time);
    
    // Measure consumption rate
    let mut consumed = 0;
    let consume_start = Instant::now();
    
    while consumed < batch_size * iterations {
        if let Some(_message) = consumer.try_consume() {
            consumed += 1;
        } else {
            // All messages consumed
            if consumed > 0 {
                break;
            }
            std::thread::sleep(Duration::from_micros(1));
        }
    }
    
    let consume_time = consume_start.elapsed();
    println!("CPU consumption completed in: {:?}", consume_time);
    
    // Calculate metrics
    let total_messages = consumed as f64;
    let total_bytes = total_messages * text.len() as f64;
    let gpu_seconds = gpu_time.as_secs_f64();
    let consume_seconds = consume_time.as_secs_f64();
    
    println!("\n=== Performance Metrics ===");
    println!("Messages processed: {}", consumed);
    println!("GPU throughput: {:.2} M msgs/sec", total_messages / gpu_seconds / 1_000_000.0);
    println!("CPU throughput: {:.2} M msgs/sec", total_messages / consume_seconds / 1_000_000.0);
    println!("Data rate: {:.2} GB/s", total_bytes / gpu_seconds / 1_000_000_000.0);
    println!("Average latency: {:.2} μs/msg", gpu_seconds * 1_000_000.0 / total_messages);
    
    Ok(())
}
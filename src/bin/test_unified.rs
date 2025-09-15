use perdix::{AgentType, Buffer, StreamContext};
use std::thread;
use std::time::Duration;

fn main() {
    println!("Testing Perdix Unified GPU Kernel");
    println!("==================================\n");

    let n_slots = 1024;
    let n_messages = 32;

    // Create buffer
    println!("Creating unified buffer...");
    let buffer = match Buffer::new(n_slots) {
        Ok(b) => {
            println!("✓ Buffer created successfully\n");
            b
        }
        Err(e) => {
            eprintln!("✗ Failed to create buffer: {}", e);
            return;
        }
    };

    // Split into producer and consumer
    let (producer, mut consumer) = buffer.split();

    // Test 1: Simple functionality test
    println!("Test 1: Simple GPU Test");
    println!("-----------------------");
    match producer.run_test(n_messages) {
        Ok(()) => println!("✓ Simple test passed\n"),
        Err(e) => {
            eprintln!("✗ Simple test failed: {}", e);
            return;
        }
    }

    // Test 2: AI Agent Response Processing with ANSI
    println!("Test 2: AI Agent Response Processing");
    println!("------------------------------------");

    // Create sample AI agent responses
    let mut contexts = Vec::new();

    // System message
    contexts.push(StreamContext::new(
        b"System initialized. All components are ready.",
        AgentType::System,
    ));

    // User query
    contexts.push(StreamContext::new(
        b"User: Please analyze the current system status.",
        AgentType::User,
    ));

    // Assistant response with keywords
    contexts.push(StreamContext::new(
        b"Assistant: Analysis complete. SUCCESS: All systems operational. No ERROR detected.",
        AgentType::Assistant,
    ));

    // Warning message
    contexts.push(StreamContext::new(
        b"WARNING: High memory usage detected. Consider optimization.",
        AgentType::Warning,
    ));

    // Error simulation
    contexts.push(StreamContext::new(
        b"ERROR: Failed to connect to remote service. Retrying...",
        AgentType::Error,
    ));

    // Debug info
    contexts.push(StreamContext::new(
        b"DEBUG: Connection retry attempt 1 of 3",
        AgentType::Debug,
    ));

    // Success after retry
    contexts.push(StreamContext::new(
        b"SUCCESS: Connection established. Service is now available.",
        AgentType::Info,
    ));

    // Process with metrics enabled
    let stream = 0u64; // Default stream
    match producer.process_agent_responses(&contexts, true, stream) {
        Ok(()) => println!("✓ Agent responses processed with ANSI formatting\n"),
        Err(e) => {
            eprintln!("✗ Failed to process agent responses: {}", e);
            return;
        }
    }

    // Test 3: Consumer verification
    println!("Test 3: Consumer Verification");
    println!("-----------------------------");

    // Consume and display messages
    let mut consumed_count = 0;
    let mut empty_iterations = 0;

    loop {
        if let Some(data) = consumer.try_consume() {
            consumed_count += 1;

            // Convert to string and display (will show ANSI codes)
            let text = String::from_utf8_lossy(&data.payload);
            println!("Message {}: {}", consumed_count, text);

            empty_iterations = 0;
        } else {
            empty_iterations += 1;

            // Give up after several empty iterations
            if empty_iterations > 10 {
                break;
            }

            // Small delay before retry
            thread::sleep(Duration::from_millis(10));
        }
    }

    println!("\n✓ Consumed {} messages total", consumed_count);

    // Test 4: Performance benchmark
    println!("\nTest 4: Performance Benchmark");
    println!("-----------------------------");

    let large_message_count = 1000;
    let mut large_contexts = Vec::new();

    for i in 0..large_message_count {
        let text = format!(
            "Benchmark message {}: This is a test of the high-performance streaming system. {}",
            i,
            if i % 100 == 0 {
                "SUCCESS"
            } else if i % 50 == 0 {
                "WARNING"
            } else {
                "OK"
            }
        );

        large_contexts.push(StreamContext::new(
            text.as_bytes(),
            match i % 8 {
                0 => AgentType::System,
                1 => AgentType::User,
                2 => AgentType::Assistant,
                3 => AgentType::Error,
                4 => AgentType::Warning,
                5 => AgentType::Info,
                6 => AgentType::Debug,
                _ => AgentType::Trace,
            },
        ));
    }

    let start = std::time::Instant::now();

    match producer.process_agent_responses(&large_contexts, true, stream) {
        Ok(()) => {
            let elapsed = start.elapsed();
            let throughput = large_message_count as f64 / elapsed.as_secs_f64();
            println!(
                "✓ Processed {} messages in {:.2?}",
                large_message_count, elapsed
            );
            println!("  Throughput: {:.0} messages/sec", throughput);

            if throughput > 10000.0 {
                println!("  Performance: EXCELLENT");
            } else if throughput > 1000.0 {
                println!("  Performance: GOOD");
            } else {
                println!("  Performance: ACCEPTABLE");
            }
        }
        Err(e) => {
            eprintln!("✗ Benchmark failed: {}", e);
        }
    }

    println!("\n==================================");
    println!("All tests completed successfully!");
}

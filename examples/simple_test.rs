use perdixlib::{Buffer, AgentType, StreamContext};
use std::thread;
use std::time::Duration;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Perdix Simple Test");
    println!("==================");
    
    // Create buffer with 1024 slots
    let n_slots = 1024;
    println!("Creating buffer with {} slots...", n_slots);
    let mut buffer = Buffer::new(n_slots)?;
    
    // Split into producer and consumer
    let (mut producer, mut consumer) = buffer.split_mut();
    
    // Test 1: CPU-based produce/consume
    println!("\nTest 1: CPU produce/consume");
    println!("----------------------------");
    
    // Produce some messages
    for i in 0..10 {
        let msg = format!("Test message {}", i);
        match producer.try_produce(msg.as_bytes()) {
            Ok(seq) => println!("Produced message {} with seq {}", i, seq),
            Err(e) => println!("Failed to produce: {}", e),
        }
    }
    
    // Consume messages
    thread::sleep(Duration::from_millis(10));
    let messages = consumer.consume_batch(20);
    println!("Consumed {} messages:", messages.len());
    for msg in messages {
        if let Ok(text) = msg.as_str() {
            println!("  [seq={}] {}", msg.seq, text);
        }
    }
    
    // Test 2: CUDA kernel test (if available)
    #[cfg(feature = "cuda")]
    {
        println!("\nTest 2: CUDA kernel test");
        println!("-------------------------");
        
        match producer.run_test(5) {
            Ok(_) => {
                println!("CUDA test kernel executed successfully");
                
                // Give kernel time to complete
                thread::sleep(Duration::from_millis(100));
                
                // Consume messages produced by kernel
                let gpu_messages = consumer.consume_batch(10);
                println!("Consumed {} GPU messages:", gpu_messages.len());
                for msg in gpu_messages {
                    if let Ok(text) = msg.as_str() {
                        println!("  [GPU seq={}] {}", msg.seq, text);
                    }
                }
            }
            Err(e) => println!("CUDA test failed: {}", e),
        }
    }
    
    // Test 3: Stream context processing (AI agent simulation)
    #[cfg(feature = "cuda")]
    {
        println!("\nTest 3: AI agent stream processing");
        println!("-----------------------------------");
        
        let contexts = vec![
            StreamContext::new(b"Hello from the AI assistant!", AgentType::Assistant),
            StreamContext::new(b"Processing your request...", AgentType::System),
            StreamContext::new(b"Here is the result: SUCCESS", AgentType::Info),
            StreamContext::new(b"WARNING: Low memory", AgentType::Warning),
            StreamContext::new(b"ERROR: Connection timeout", AgentType::Error),
        ];
        
        match producer.process_agent_responses(&contexts, false, 0) {
            Ok(_) => {
                println!("Processed {} agent messages", contexts.len());
                
                thread::sleep(Duration::from_millis(100));
                
                let agent_messages = consumer.consume_batch(10);
                println!("Consumed {} agent messages:", agent_messages.len());
                for msg in agent_messages {
                    let agent_type = (msg.flags >> 8) & 0xFF;
                    if let Ok(text) = msg.as_str() {
                        println!("  [Agent={} seq={}] {}", agent_type, msg.seq, text);
                    }
                }
            }
            Err(e) => println!("Agent processing failed: {}", e),
        }
    }
    
    println!("\n=== Test Complete ===");
    Ok(())
}
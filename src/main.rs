use perdixlib::{Buffer, AgentType, StreamContext};
use std::thread;
use std::time::{Duration, Instant};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔══════════════════════════════════════════════╗");
    println!("║         Perdix GPU Ring Buffer Demo         ║");
    println!("║    High-Performance GPU-CPU Streaming       ║");
    println!("╚══════════════════════════════════════════════╝");
    
    // Configuration
    let n_slots = 4096;  // Must be power of 2
    let n_messages = 100;
    
    println!("\nConfiguration:");
    println!("  Buffer slots: {}", n_slots);
    println!("  Test messages: {}", n_messages);
    
    // Initialize buffer
    println!("\nInitializing unified memory buffer...");
    let buffer = Buffer::new(n_slots)?;
    let (mut producer, consumer) = buffer.split();
    
    // Create stop flag for consumer thread
    let stop_flag = Arc::new(AtomicBool::new(false));
    let stop_clone = stop_flag.clone();
    
    // Spawn consumer thread
    let consumer_handle = thread::spawn(move || {
        let mut consumer = consumer;
        let mut total_consumed = 0u64;
        let mut last_report = Instant::now();
        
        println!("[Consumer] Thread started");
        
        while !stop_clone.load(Ordering::Relaxed) {
            // Consume available messages
            let messages = consumer.consume_available(Some(100));
            
            if !messages.is_empty() {
                total_consumed += messages.len() as u64;
                
                // Sample first message for display
                if let Some(first) = messages.first() {
                    if let Ok(text) = first.as_str() {
                        println!("[Consumer] Sample message (seq={}): {}", first.seq, text);
                    }
                }
            }
            
            // Report stats every second
            if last_report.elapsed() > Duration::from_secs(1) {
                let available = consumer.available();
                println!("[Consumer] Stats: consumed={}, available={}", total_consumed, available);
                last_report = Instant::now();
            }
            
            // Small sleep to prevent busy waiting
            if messages.is_empty() {
                thread::sleep(Duration::from_micros(100));
            }
        }
        
        // Drain remaining messages
        let remaining = consumer.drain();
        total_consumed += remaining.len() as u64;
        
        println!("[Consumer] Thread stopping. Total consumed: {}", total_consumed);
        total_consumed
    });
    
    // Test different producer modes
    println!("\n=== Running Tests ===");
    
    // Test 1: CPU Producer
    println!("\n1. CPU Producer Test");
    println!("--------------------");
    let start = Instant::now();
    
    for i in 0..20 {
        let msg = format!("CPU Message #{:04} [timestamp={}]", i, start.elapsed().as_micros());
        producer.try_produce(msg.as_bytes()).ok();
    }
    
    println!("CPU messages produced in {:?}", start.elapsed());
    thread::sleep(Duration::from_millis(100));
    
    // Test 2: CUDA Simple Test Kernel
    #[cfg(feature = "cuda")]
    {
        println!("\n2. CUDA Test Kernel");
        println!("--------------------");
        let start = Instant::now();
        
        match producer.run_test(50) {
            Ok(_) => println!("CUDA test kernel completed in {:?}", start.elapsed()),
            Err(e) => println!("CUDA test failed: {}", e),
        }
        
        thread::sleep(Duration::from_millis(100));
    }
    
    // Test 3: AI Agent Streaming Simulation
    #[cfg(feature = "cuda")]
    {
        println!("\n3. AI Agent Streaming");
        println!("----------------------");
        
        // Simulate different AI agents producing messages
        let agents = vec![
            (AgentType::System, "System initializing..."),
            (AgentType::User, "What is the weather today?"),
            (AgentType::Assistant, "I'll check the weather for you. The current conditions show partly cloudy skies with a temperature of 72°F."),
            (AgentType::Info, "Weather data retrieved successfully"),
            (AgentType::User, "Can you also check tomorrow's forecast?"),
            (AgentType::Assistant, "Tomorrow's forecast shows sunny conditions with a high of 78°F and a low of 65°F."),
            (AgentType::Warning, "API rate limit approaching"),
            (AgentType::Error, "Network timeout on secondary request"),
            (AgentType::Debug, "Retry attempt 1 of 3"),
            (AgentType::Assistant, "Despite the network issue, I was able to retrieve the forecast from cache."),
        ];
        
        let contexts: Vec<StreamContext> = agents.iter()
            .map(|(agent_type, text)| StreamContext::new(text.as_bytes(), *agent_type))
            .collect();
        
        let start = Instant::now();
        match producer.process_agent_responses(&contexts, true, 0) {
            Ok(_) => println!("Processed {} agent messages in {:?}", contexts.len(), start.elapsed()),
            Err(e) => println!("Agent processing failed: {}", e),
        }
        
        thread::sleep(Duration::from_millis(200));
    }
    
    // Test 4: Throughput test
    println!("\n4. Throughput Test");
    println!("------------------");
    let start = Instant::now();
    let mut produced = 0u64;
    
    while start.elapsed() < Duration::from_secs(2) {
        let msg = format!("Throughput test {}", produced);
        if producer.try_produce(msg.as_bytes()).is_ok() {
            produced += 1;
        } else {
            // Buffer full, wait a bit
            thread::sleep(Duration::from_micros(10));
        }
    }
    
    let elapsed = start.elapsed();
    let throughput = produced as f64 / elapsed.as_secs_f64();
    println!("Produced {} messages in {:?}", produced, elapsed);
    println!("Throughput: {:.0} messages/second", throughput);
    
    // Stop consumer and wait
    println!("\n=== Shutting down ===");
    thread::sleep(Duration::from_millis(500));
    stop_flag.store(true, Ordering::Relaxed);
    
    let total_consumed = consumer_handle.join().unwrap();
    
    // Final report
    println!("\n╔══════════════════════════════════════════════╗");
    println!("║                Final Report                 ║");
    println!("╠══════════════════════════════════════════════╣");
    println!("║  Total messages consumed: {:18} ║", total_consumed);
    println!("║  Buffer size: {:30} ║", format!("{} slots", n_slots));
    println!("║  Status: SUCCESS                            ║");
    println!("╚══════════════════════════════════════════════╝");
    
    Ok(())
}
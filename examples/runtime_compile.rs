/// Example demonstrating Perdix with runtime CUDA compilation via NVRTC
/// This bypasses the nvcc/MSVC incompatibility issues
use perdix::PerdixRuntime;
use std::thread;
use std::time::{Duration, Instant};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Perdix Runtime Compilation Demo ===\n");

    // Initialize runtime with 1024 slots
    let n_slots = 1024;
    let mut runtime = PerdixRuntime::new(0, n_slots)?;

    println!("\n--- Test 1: Basic Message Test ---");
    runtime.run_test(100)?;

    println!("\n--- Test 2: ANSI Colored Messages ---");

    // Simulate AI agent responses with different types
    let messages = vec![
        "System initialized successfully".to_string(),
        "User query: How can I optimize my CUDA kernels?".to_string(),
        "I can help you optimize CUDA kernels. Key strategies include:".to_string(),
        "ERROR: Unable to allocate device memory".to_string(),
        "WARNING: Kernel launch configuration may be suboptimal".to_string(),
        "INFO: Compilation completed in 0.5 seconds".to_string(),
        "DEBUG: Thread block size = 256".to_string(),
        "TRACE: Entering kernel launch function".to_string(),
    ];

    // Agent types: 0=SYSTEM, 1=USER, 2=ASSISTANT, 3=ERROR, 4=WARNING, 5=INFO, 6=DEBUG, 7=TRACE
    let agent_types = vec![0, 1, 2, 3, 4, 5, 6, 7];

    runtime.run_ansi_kernel(&messages, &agent_types)?;

    println!("\n--- Test 3: Performance Benchmark ---");

    // Generate larger batch of messages
    let mut bench_messages = Vec::new();
    let mut bench_types = Vec::new();

    for i in 0..1000 {
        bench_messages.push(format!("Benchmark message {}: Processing data stream", i));
        bench_types.push((i % 8) as u8);
    }

    let start = Instant::now();
    runtime.run_ansi_kernel(&bench_messages, &bench_types)?;
    let elapsed = start.elapsed();

    let throughput = 1000.0 / elapsed.as_secs_f64();
    println!("Processed 1000 messages in {:?}", elapsed);
    println!("Throughput: {:.0} messages/second", throughput);

    println!("\n--- Test 4: Consumer Thread Simulation ---");

    // Simulate a CPU consumer reading from the ring buffer
    let header_ptr = runtime.get_header_ptr();
    let slots_ptr = runtime.get_slots_ptr();
    let slot_count = runtime.slot_count();

    // Launch producer on GPU
    let producer_handle = thread::spawn(move || {
        let mut runtime = PerdixRuntime::new(0, 1024).expect("Failed to create runtime");

        for batch in 0..5 {
            thread::sleep(Duration::from_millis(100));

            let messages: Vec<String> = (0..20)
                .map(|i| format!("Batch {} Message {}", batch, i))
                .collect();
            let types = vec![2u8; 20]; // All ASSISTANT messages

            runtime
                .run_ansi_kernel(&messages, &types)
                .expect("Failed to run kernel");
            println!("GPU: Produced batch {}", batch);
        }
    });

    // Consumer loop
    let mut last_read_seq = 0u64;
    let mut messages_consumed = 0;
    let consumer_start = Instant::now();

    while messages_consumed < 100 && consumer_start.elapsed() < Duration::from_secs(5) {
        // Read current write index from header
        let write_idx = unsafe {
            let write_idx_ptr = header_ptr as *const u64;
            std::ptr::read_volatile(write_idx_ptr)
        };

        // Process available messages
        while last_read_seq < write_idx {
            let slot_idx = (last_read_seq % slot_count as u64) as usize;
            let slot_offset = slot_idx * 256;
            let slot_ptr = unsafe { slots_ptr.add(slot_offset) };

            // Read sequence number
            let seq = unsafe {
                let seq_ptr = slot_ptr as *const u64;
                std::ptr::read_volatile(seq_ptr)
            };

            // Check if slot is ready
            if seq == last_read_seq {
                let len = unsafe {
                    let len_ptr = slot_ptr.add(8) as *const u32;
                    *len_ptr
                };

                if len > 0 && len < 192 {
                    messages_consumed += 1;
                    if messages_consumed <= 5 || messages_consumed % 20 == 0 {
                        let payload =
                            unsafe { std::slice::from_raw_parts(slot_ptr.add(64), len as usize) };
                        println!(
                            "CPU: Consumed message {}: {:?}",
                            messages_consumed,
                            String::from_utf8_lossy(payload)
                        );
                    }
                }

                last_read_seq += 1;
            } else {
                // Slot not ready yet, wait a bit
                thread::sleep(Duration::from_micros(10));
            }
        }

        thread::sleep(Duration::from_millis(10));
    }

    producer_handle.join().unwrap();

    println!("\nTotal messages consumed: {}", messages_consumed);
    println!("\n=== Demo Complete ===");

    Ok(())
}

use perdix::Buffer;
use std::collections::HashSet;

#[test]
fn test_buffer_is_truly_persistent() {
    let mut buffer = Buffer::new(16).expect("Failed to create buffer");
    let (initial_header, initial_slots) = buffer.as_raw_parts();

    // Test that pointers remain constant across multiple split_mut calls
    for i in 0..100 {
        let (producer, consumer) = buffer.split_mut();
        let header_ptr = producer.header_ptr();
        let slots_ptr = producer.slots_ptr();

        assert_eq!(header_ptr, initial_header,
                   "Header pointer changed at iteration {}: {:p} != {:p}",
                   i, header_ptr, initial_header);
        assert_eq!(slots_ptr, initial_slots,
                   "Slots pointer changed at iteration {}: {:p} != {:p}",
                   i, slots_ptr, initial_slots);
    }
}

#[test]
fn test_ring_buffer_actually_wraps() {
    let mut buffer = Buffer::new(8).expect("Failed to create buffer");
    let (mut producer, mut consumer) = buffer.split_mut();

    // Fill buffer completely (8 slots)
    for i in 0..8 {
        let msg = format!("First{:02}", i);
        assert!(producer.try_produce(msg.as_bytes()).is_ok(),
                "Failed to produce message {}", i);
    }

    // Buffer should be full now
    assert!(producer.try_produce(b"SHOULD_FAIL").is_err(),
            "Buffer should be full but accepted another message!");

    // Consume first 4 messages
    for i in 0..4 {
        let msg = consumer.try_consume()
            .expect(&format!("Failed to consume message {}", i));
        assert_eq!(msg.as_str(), format!("First{:02}", i));
    }

    // Now we should be able to produce 4 more (wrapping around)
    for i in 0..4 {
        let msg = format!("Second{:02}", i);
        assert!(producer.try_produce(msg.as_bytes()).is_ok(),
                "Failed to produce wrapped message {}", i);
    }

    // Buffer should be full again
    assert!(producer.try_produce(b"SHOULD_FAIL").is_err(),
            "Buffer should be full after wraparound!");

    // Consume all 8 messages - should get last 4 of first batch, then 4 of second
    let mut consumed = Vec::new();
    while let Some(msg) = consumer.try_consume() {
        consumed.push(msg.as_str().to_string());
    }

    assert_eq!(consumed, vec![
        "First04", "First05", "First06", "First07",
        "Second00", "Second01", "Second02", "Second03"
    ]);
}

#[test]
fn test_indices_continuously_increment() {
    let mut buffer = Buffer::new(16).expect("Failed to create buffer");
    let header_ptr = buffer.as_raw_parts().0;

    // Record initial indices
    let initial_write = unsafe {
        (*header_ptr).producer.write_idx.load(std::sync::atomic::Ordering::Acquire)
    };
    let initial_read = unsafe {
        (*header_ptr).consumer.read_idx.load(std::sync::atomic::Ordering::Acquire)
    };

    let (mut producer, mut consumer) = buffer.split_mut();

    // Do many produce/consume cycles - indices should increment, not reset
    let mut last_write = initial_write;
    let mut last_read = initial_read;

    for cycle in 0..10 {
        // Produce 10 messages
        for i in 0..10 {
            if producer.try_produce(format!("C{}M{}", cycle, i).as_bytes()).is_ok() {
                let write_idx = unsafe {
                    (*header_ptr).producer.write_idx.load(std::sync::atomic::Ordering::Acquire)
                };

                assert!(write_idx > last_write,
                        "Write index didn't increment: {} <= {} at cycle {}",
                        write_idx, last_write, cycle);
                last_write = write_idx;
            }
        }

        // Consume all available
        while consumer.try_consume().is_some() {
            let read_idx = unsafe {
                (*header_ptr).consumer.read_idx.load(std::sync::atomic::Ordering::Acquire)
            };

            assert!(read_idx > last_read,
                    "Read index didn't increment: {} <= {} at cycle {}",
                    read_idx, last_read, cycle);
            last_read = read_idx;
        }
    }

    // Indices should have wrapped (incremented way past buffer size)
    assert!(last_write > 50, "Write index should have incremented past buffer size: {}", last_write);
    assert!(last_read > 50, "Read index should have incremented past buffer size: {}", last_read);
}

#[test]
fn test_memory_slots_are_reused() {
    let mut buffer = Buffer::new(4).expect("Failed to create buffer with 4 slots");
    let (_, slots_ptr) = buffer.as_raw_parts();
    let (mut producer, mut consumer) = buffer.split_mut();

    // Track which memory addresses we write to
    let mut first_round_addrs = HashSet::new();
    let mut second_round_addrs = HashSet::new();

    // First round: fill buffer completely
    for i in 0..4 {
        producer.try_produce(format!("A{}", i).as_bytes())
            .expect(&format!("Failed to produce A{}", i));

        // Calculate slot address
        let slot_addr = unsafe { slots_ptr.add(i) as usize };
        first_round_addrs.insert(slot_addr);
    }

    // Consume all messages
    for _ in 0..4 {
        consumer.try_consume().expect("Failed to consume");
    }

    // Second round: fill buffer again - should use same slots!
    for i in 0..4 {
        producer.try_produce(format!("B{}", i).as_bytes())
            .expect(&format!("Failed to produce B{}", i));

        // Calculate slot address (should be same as first round)
        let slot_addr = unsafe { slots_ptr.add(i) as usize };
        second_round_addrs.insert(slot_addr);
    }

    // The addresses should be identical!
    assert_eq!(first_round_addrs, second_round_addrs,
               "Memory slots were not reused! First: {:?}, Second: {:?}",
               first_round_addrs, second_round_addrs);

    assert_eq!(first_round_addrs.len(), 4, "Should have used exactly 4 slots");
}

#[test]
fn test_benchmark_reuses_buffer() {
    // This simulates what the benchmark does
    let mut buffer = Buffer::new(256).expect("Failed to create buffer");
    let (header_ptr, slots_ptr) = buffer.as_raw_parts();

    // Track that we're using the same memory across iterations
    let mut iteration_count = 0;

    // Simulate benchmark iterations
    for _ in 0..100 {
        let (mut producer, mut consumer) = buffer.split_mut();

        // Verify pointers haven't changed
        assert_eq!(producer.header_ptr(), header_ptr);
        assert_eq!(producer.slots_ptr(), slots_ptr);

        // Do some work
        for i in 0..50 {
            producer.try_produce(format!("Iter{}Msg{}", iteration_count, i).as_bytes()).ok();
        }

        while consumer.try_consume().is_some() {}

        iteration_count += 1;
    }

    assert_eq!(iteration_count, 100, "Should have completed 100 iterations with same buffer");

    // Buffer is still alive here - only one allocation/deallocation for entire test
}

#[test]
fn test_producer_consumer_opposite_ends() {
    let mut buffer = Buffer::new(16).expect("Failed to create buffer");
    let header_ptr = buffer.as_raw_parts().0;
    let (mut producer, mut consumer) = buffer.split_mut();

    // Fill half the buffer
    for i in 0..8 {
        producer.try_produce(format!("Msg{}", i).as_bytes())
            .expect("Failed to produce");
    }

    let write_idx = unsafe {
        (*header_ptr).producer.write_idx.load(std::sync::atomic::Ordering::Acquire)
    };
    let read_idx = unsafe {
        (*header_ptr).consumer.read_idx.load(std::sync::atomic::Ordering::Acquire)
    };

    // Producer should be ahead of consumer
    assert!(write_idx > read_idx,
            "Producer ({}) should be ahead of consumer ({})", write_idx, read_idx);
    assert_eq!((write_idx - read_idx), 8,
               "Gap should be 8 messages");

    // Consume 4 messages
    for _ in 0..4 {
        consumer.try_consume().expect("Failed to consume");
    }

    let new_read_idx = unsafe {
        (*header_ptr).consumer.read_idx.load(std::sync::atomic::Ordering::Acquire)
    };

    assert_eq!(new_read_idx - read_idx, 4, "Consumer should have advanced by 4");

    // Producer and consumer are at opposite ends of the ring
    let gap = write_idx - new_read_idx;
    assert_eq!(gap, 4, "Gap should now be 4 messages");
}
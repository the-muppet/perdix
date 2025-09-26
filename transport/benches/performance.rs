//! # Performance Benchmarks for Perdix Transport Layer
//!
//! This benchmark suite measures the core performance characteristics
//! of the Perdix ring buffer as a high-performance transport layer.
//!
//! ## Benchmarks
//!
//! - **Throughput**: Raw data transfer rate
//! - **Latency**: Single message round-trip time
//! - **Message Sizes**: Performance across different payload sizes
//! - **Burst Patterns**: Producer/consumer burst scenarios

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use perdix::Buffer;
use std::time::Duration;

/// Benchmark throughput with persistent buffer
fn benchmark_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("throughput");
    group.measurement_time(Duration::from_secs(5));

    // Different data sizes to test
    let test_sizes = [(1024, "1KB"), (64 * 1024, "64KB"), (1024 * 1024, "1MB")];

    for (data_size, label) in test_sizes {
        group.throughput(Throughput::Bytes(data_size as u64));

        // Create buffer once before benchmarking
        let mut buffer = Buffer::new(4096).expect("Failed to create buffer");

        group.bench_function(label, |b| {
            let data = vec![0xAB; data_size];

            b.iter(|| {
                let (mut producer, mut consumer) = buffer.split_mut();

                // Process data in chunks
                for chunk in data.chunks(240) {  // Max payload size
                    producer.try_produce(chunk).ok();
                }

                // Consume all messages
                while let Some(msg) = consumer.try_consume() {
                    black_box(msg);
                }
            });
        });
    }

    group.finish();
}

/// Benchmark single message latency
fn benchmark_latency(c: &mut Criterion) {
    let mut group = c.benchmark_group("latency");

    let mut buffer = Buffer::new(256).expect("Failed to create buffer");

    group.bench_function("single_message", |b| {
        let test_msg = b"Test message for latency measurement";

        b.iter(|| {
            let (mut producer, mut consumer) = buffer.split_mut();

            producer.try_produce(test_msg).expect("Failed to produce");
            let msg = consumer.try_consume().expect("Failed to consume");
            black_box(msg);
        });
    });

    group.finish();
}

/// Benchmark with varying message sizes
fn benchmark_message_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("message_sizes");

    let mut buffer = Buffer::new(2048).expect("Failed to create buffer");

    for size in [8, 16, 32, 64, 128, 192, 240].iter() {
        group.throughput(Throughput::Bytes(*size as u64 * 1000)); // 1000 messages

        group.bench_function(format!("{}_bytes", size), |b| {
            let msg = vec![0x42; *size];

            b.iter(|| {
                let (mut producer, mut consumer) = buffer.split_mut();

                // Send 1000 messages
                for _ in 0..1000 {
                    producer.try_produce(&msg).expect("Failed to produce");
                }

                // Consume all
                let mut count = 0;
                while let Some(msg) = consumer.try_consume() {
                    black_box(msg);
                    count += 1;
                }
                assert_eq!(count, 1000);
            });
        });
    }

    group.finish();
}

/// Benchmark producer-consumer patterns
fn benchmark_patterns(c: &mut Criterion) {
    let mut group = c.benchmark_group("patterns");

    let mut buffer = Buffer::new(4096).expect("Failed to create buffer");

    // Pattern 1: Interleaved (produce-consume-produce-consume)
    group.bench_function("interleaved", |b| {
        b.iter(|| {
            let (mut producer, mut consumer) = buffer.split_mut();

            for i in 0..100 {
                let msg = format!("Message {}", i);
                producer.try_produce(msg.as_bytes()).expect("Failed to produce");

                if let Some(msg) = consumer.try_consume() {
                    black_box(msg);
                }
            }

            // Drain remaining
            while let Some(msg) = consumer.try_consume() {
                black_box(msg);
            }
        });
    });

    // Pattern 2: Burst (produce many, then consume all)
    for burst_size in [100, 500, 1000, 2000].iter() {
        group.bench_function(format!("burst_{}", burst_size), |b| {
            let msg = b"Burst message";

            b.iter(|| {
                let (mut producer, mut consumer) = buffer.split_mut();

                // Producer burst
                for _ in 0..*burst_size {
                    producer.try_produce(msg).expect("Failed to produce");
                }

                // Consumer burst
                let mut consumed = 0;
                while let Some(msg) = consumer.try_consume() {
                    black_box(msg);
                    consumed += 1;
                }
                assert_eq!(consumed, *burst_size);
            });
        });
    }

    group.finish();
}

/// Benchmark backpressure handling
fn benchmark_backpressure(c: &mut Criterion) {
    let mut group = c.benchmark_group("backpressure");

    let mut buffer = Buffer::new(256).expect("Failed to create buffer");

    group.bench_function("overflow_handling", |b| {
        b.iter(|| {
            let (mut producer, mut consumer) = buffer.split_mut();

            // Fill buffer
            let mut produced = 0;
            while producer.try_produce(b"Fill").is_ok() {
                produced += 1;
            }

            // Try overflow (should fail)
            for _ in 0..10 {
                let result = producer.try_produce(b"Overflow");
                black_box(result.is_err());
            }

            // Consume half
            for _ in 0..produced/2 {
                consumer.try_consume();
            }

            // Produce more (should succeed)
            for _ in 0..produced/2 {
                producer.try_produce(b"More").ok();
            }

            // Drain all
            while consumer.try_consume().is_some() {}
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    benchmark_throughput,
    benchmark_latency,
    benchmark_message_sizes,
    benchmark_patterns,
    benchmark_backpressure
);
criterion_main!(benches);
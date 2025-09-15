use criterion::{black_box, criterion_group, criterion_main, Criterion};
use perdix::Buffer;
use std::time::{Duration, Instant};

fn benchmark_message_latency(c: &mut Criterion) {
    let mut group = c.benchmark_group("message_latency");
    
    group.bench_function("end_to_end", |b| {
        let mut buffer = Buffer::new(1024).unwrap();
        let (mut producer, mut consumer) = buffer.split_mut();
        let data = b"Latency test message";
        
        b.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            
            for _ in 0..iters {
                let start = Instant::now();
                let _ = producer.try_produce(data);
                let _msg = black_box(consumer.try_consume().unwrap());
                total += start.elapsed();
            }
            
            total
        });
    });
    
    group.bench_function("produce_only", |b| {
        let mut buffer = Buffer::new(1024).unwrap();
        let (mut producer, _consumer) = buffer.split_mut();
        let data = b"Produce latency test";
        
        b.iter(|| {
            let _ = producer.try_produce(data);
        });
    });
    
    group.bench_function("consume_only", |b| {
        let mut buffer = Buffer::new(1024).unwrap();
        let (mut producer, mut consumer) = buffer.split_mut();
        
        // Pre-fill buffer
        for i in 0..512 {
            let data = format!("Message {}", i);
            let _ = producer.try_produce(data.as_bytes());
        }
        
        b.iter(|| {
            black_box(consumer.try_consume());
        });
    });
    
    group.finish();
}

fn benchmark_contention(c: &mut Criterion) {
    let mut group = c.benchmark_group("contention");
    
    group.bench_function("high_contention", |b| {
        let mut buffer = Buffer::new(1024).unwrap();
        let (mut producer, mut consumer) = buffer.split_mut();
        
        // Pre-fill buffer to create contention
        for _ in 0..512 {
            let _ = producer.try_produce(b"Contention test");
        }
        
        // Benchmark produce/consume under contention
        b.iter(|| {
            // Try to produce (may fail if full)
            let _ = producer.try_produce(b"New message");
            // Try to consume (should usually succeed)
            black_box(consumer.try_consume());
        });
    });
    
    group.finish();
}

#[cfg(feature = "cuda")]
fn benchmark_gpu_latency(c: &mut Criterion) {
    use perdix::GpuProducer;
    
    let mut group = c.benchmark_group("gpu_latency");
    
    // Since we can't currently benchmark GPU producer without the specific kernel methods,
    // we'll skip this benchmark for now
    println!("Note: GPU latency benchmarks require additional kernel implementation");
    
    group.finish();
}

// Create two versions of criterion_group based on features
#[cfg(feature = "cuda")]
criterion_group!(
    benches,
    benchmark_message_latency,
    benchmark_contention,
    benchmark_gpu_latency
);

#[cfg(not(feature = "cuda"))]
criterion_group!(
    benches,
    benchmark_message_latency,
    benchmark_contention
);

criterion_main!(benches);
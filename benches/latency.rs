use criterion::{black_box, criterion_group, criterion_main, Criterion};
use perdix::{Buffer, AgentType};
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
                producer.try_produce(data, AgentType::System);
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
            producer.try_produce(data, AgentType::System);
        });
    });
    
    group.bench_function("consume_only", |b| {
        let mut buffer = Buffer::new(1024).unwrap();
        let (mut producer, mut consumer) = buffer.split_mut();
        
        // Pre-fill buffer
        for i in 0..512 {
            let data = format!("Message {}", i);
            producer.try_produce(data.as_bytes(), AgentType::Info);
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
        use std::thread;
        use std::sync::Arc;
        use std::sync::atomic::{AtomicBool, Ordering};
        
        let buffer = Buffer::new(1024).unwrap();
        let (producer, consumer) = buffer.split();
        let stop = Arc::new(AtomicBool::new(false));
        let stop_clone = stop.clone();
        
        // Spawn aggressive producer
        let producer_handle = thread::spawn(move || {
            let mut producer = producer;
            let data = b"Contention test";
            while !stop_clone.load(Ordering::Relaxed) {
                producer.try_produce(data, AgentType::Debug);
            }
        });
        
        // Benchmark consumer under contention
        b.iter_custom(|iters| {
            let mut consumer = consumer.clone();
            let start = Instant::now();
            
            for _ in 0..iters {
                while consumer.try_consume().is_none() {
                    std::hint::spin_loop();
                }
            }
            
            start.elapsed()
        });
        
        stop.store(true, Ordering::Relaxed);
        producer_handle.join().unwrap();
    });
    
    group.finish();
}

#[cfg(feature = "cuda")]
fn benchmark_gpu_latency(c: &mut Criterion) {
    use perdix::GpuProducer;
    
    let mut group = c.benchmark_group("gpu_latency");
    
    group.bench_function("gpu_to_cpu", |b| {
        let buffer = Buffer::new(1024).unwrap();
        let (_, mut consumer) = buffer.split_mut();
        let mut gpu = GpuProducer::new(buffer, 0).unwrap();
        
        b.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            
            for _ in 0..iters {
                let start = Instant::now();
                
                // Launch GPU kernel
                gpu.launch_test_kernel(1).unwrap();
                
                // Wait for message
                while consumer.try_consume().is_none() {
                    std::hint::spin_loop();
                }
                
                total += start.elapsed();
            }
            
            total
        });
    });
    
    group.finish();
}

criterion_group!(
    benches,
    benchmark_message_latency,
    benchmark_contention,
    #[cfg(feature = "cuda")]
    benchmark_gpu_latency
);
criterion_main!(benches);
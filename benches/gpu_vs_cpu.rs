#[cfg(feature = "cuda")]
use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use perdix::{Buffer, AgentType};

#[cfg(not(feature = "cuda"))]
fn main() {
    println!("GPU benchmarks require --features cuda");
}

#[cfg(feature = "cuda")]
fn benchmark_cpu_baseline(c: &mut Criterion) {
    let mut group = c.benchmark_group("cpu_baseline");
    
    for n_messages in [100, 1000, 10000].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(n_messages),
            n_messages,
            |b, &n_messages| {
                let mut buffer = Buffer::new(16384).unwrap();
                let (mut producer, mut consumer) = buffer.split_mut();
                
                b.iter(|| {
                    // CPU produce
                    for i in 0..n_messages {
                        let msg = format!("CPU Message {}", i);
                        producer.try_produce(msg.as_bytes(), AgentType::System);
                    }
                    
                    // Consume all
                    let mut consumed = 0;
                    while let Some(_) = consumer.try_consume() {
                        consumed += 1;
                    }
                    black_box(consumed);
                });
            },
        );
    }
    
    group.finish();
}

#[cfg(feature = "cuda")]
fn benchmark_gpu_accelerated(c: &mut Criterion) {
    use perdix::GpuProducer;
    
    let mut group = c.benchmark_group("gpu_accelerated");
    
    for n_messages in [100, 1000, 10000].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(n_messages),
            n_messages,
            |b, &n_messages| {
                let buffer = Buffer::new(16384).unwrap();
                let (_, mut consumer) = buffer.split_mut();
                let mut gpu = GpuProducer::new(buffer, 0).unwrap();
                
                b.iter(|| {
                    // GPU produce
                    gpu.launch_test_kernel(*n_messages as i32).unwrap();
                    
                    // Consume all
                    let mut consumed = 0;
                    while consumed < *n_messages {
                        if let Some(_) = consumer.try_consume() {
                            consumed += 1;
                        }
                    }
                    black_box(consumed);
                });
            },
        );
    }
    
    group.finish();
}

#[cfg(feature = "cuda")]
fn benchmark_mixed_workload(c: &mut Criterion) {
    use perdix::GpuProducer;
    use std::thread;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU64, Ordering};
    
    let mut group = c.benchmark_group("mixed_workload");
    
    group.bench_function("gpu_produce_cpu_consume", |b| {
        let buffer = Buffer::new(8192).unwrap();
        let (producer, consumer) = buffer.split();
        let mut gpu = GpuProducer::new(buffer, 0).unwrap();
        
        let total_consumed = Arc::new(AtomicU64::new(0));
        let total_clone = total_consumed.clone();
        
        // Consumer thread
        let consumer_handle = thread::spawn(move || {
            let mut consumer = consumer;
            let mut local_count = 0u64;
            
            loop {
                if let Some(_) = consumer.try_consume() {
                    local_count += 1;
                    if local_count >= 10000 {
                        break;
                    }
                }
            }
            
            total_clone.store(local_count, Ordering::Relaxed);
        });
        
        b.iter(|| {
            // Launch GPU work
            gpu.launch_test_kernel(10000).unwrap();
            
            // Wait for consumer
            consumer_handle.join().unwrap();
            
            black_box(total_consumed.load(Ordering::Relaxed));
        });
    });
    
    group.finish();
}

#[cfg(feature = "cuda")]
criterion_group!(
    benches,
    benchmark_cpu_baseline,
    benchmark_gpu_accelerated,
    benchmark_mixed_workload
);

#[cfg(feature = "cuda")]
criterion_main!(benches);
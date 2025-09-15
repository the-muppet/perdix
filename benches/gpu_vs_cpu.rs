#[cfg(feature = "cuda")]
use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use perdix::Buffer;

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
                        producer.try_produce(msg.as_bytes());
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
    let mut group = c.benchmark_group("gpu_accelerated");
    
    // Since we can't currently benchmark GPU producer without the specific kernel methods,
    // we'll skip this benchmark for now
    println!("Note: GPU acceleration benchmarks require additional kernel implementation");
    
    group.finish();
}

#[cfg(feature = "cuda")]
fn benchmark_mixed_workload(c: &mut Criterion) {
    let mut group = c.benchmark_group("mixed_workload");
    
    // Since we can't currently benchmark GPU producer without the specific kernel methods,
    // we'll skip this benchmark for now
    println!("Note: Mixed workload benchmarks require additional kernel implementation");
    
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
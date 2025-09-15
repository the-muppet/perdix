use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use perdix::Buffer;
use std::time::Duration;

fn benchmark_single_message(c: &mut Criterion) {
    let mut group = c.benchmark_group("single_message");
    
    for size in [64, 128, 240].iter() {
        group.throughput(Throughput::Bytes(*size as u64));
        group.bench_with_input(format!("size_{}", size), size, |b, &size| {
            let mut buffer = Buffer::new(1024).unwrap();
            let (mut producer, mut consumer) = buffer.split_mut();
            let data = vec![b'A'; size];
            
            b.iter(|| {
                let _ = producer.try_produce(&data);
                black_box(consumer.try_consume());
            });
        });
    }
    
    group.finish();
}

fn benchmark_batch_messages(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_messages");
    group.measurement_time(Duration::from_secs(10));
    
    for batch_size in [32, 64, 128, 256].iter() {
        group.throughput(Throughput::Elements(*batch_size as u64));
        group.bench_with_input(format!("batch_{}", batch_size), batch_size, |b, &batch_size| {
            let mut buffer = Buffer::new(4096).unwrap();
            let (mut producer, mut consumer) = buffer.split_mut();
            let data = b"Test message for benchmarking throughput";
            
            b.iter(|| {
                // Produce batch
                for _ in 0..batch_size {
                    let _ = producer.try_produce(data);
                }
                
                // Consume batch
                for _ in 0..batch_size {
                    black_box(consumer.try_consume());
                }
            });
        });
    }
    
    group.finish();
}

fn benchmark_sustained_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("sustained_throughput");
    group.measurement_time(Duration::from_secs(30));
    group.sample_size(10);
    
    group.bench_function("1MB_transfer", |b| {
        let mut buffer = Buffer::new(8192).unwrap();
        let (mut producer, mut consumer) = buffer.split_mut();
        let chunk = vec![b'X'; 240]; // Max payload size
        let chunks_per_mb = (1024 * 1024) / 240;
        
        b.iter(|| {
            // Produce 1MB of data
            for _ in 0..chunks_per_mb {
                let _ = producer.try_produce(&chunk);
            }
            
            // Consume all
            let mut consumed = 0;
            while let Some(_) = consumer.try_consume() {
                consumed += 1;
            }
            black_box(consumed);
        });
    });
    
    group.finish();
}

criterion_group!(
    benches,
    benchmark_single_message,
    benchmark_batch_messages,
    benchmark_sustained_throughput
);
criterion_main!(benches);
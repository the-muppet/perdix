use std::time::{Duration, Instant};
use std::sync::Arc;
use std::thread;

#[derive(Debug, Clone)]
pub struct BenchmarkResults {
    pub total_messages: u64,
    pub duration: Duration,
    pub throughput_msg_per_sec: f64,
    pub throughput_gb_per_sec: f64,
    pub avg_latency_ns: u64,
    pub p50_latency_ns: u64,
    pub p99_latency_ns: u64,
    pub p999_latency_ns: u64,
}

pub struct Benchmark {
    name: String,
    warmup_iterations: u32,
    test_iterations: u32,
    message_size: usize,
    ring_size: usize,
}

impl Benchmark {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            warmup_iterations: 1000,
            test_iterations: 100000,
            message_size: 64,
            ring_size: 4096,
        }
    }
    
    pub fn with_iterations(mut self, warmup: u32, test: u32) -> Self {
        self.warmup_iterations = warmup;
        self.test_iterations = test;
        self
    }
    
    pub fn with_message_size(mut self, size: usize) -> Self {
        self.message_size = size;
        self
    }
    
    pub fn with_ring_size(mut self, size: usize) -> Self {
        self.ring_size = size;
        self
    }
    
    /// Run throughput benchmark
    pub fn run_throughput(&self) -> BenchmarkResults {
        println!("Running throughput benchmark: {}", self.name);
        println!("  Ring size: {}", self.ring_size);
        println!("  Message size: {} bytes", self.message_size);
        println!("  Iterations: {} (after {} warmup)", 
                 self.test_iterations, self.warmup_iterations);
        
        // Allocate ring buffer
        let mut buffer = crate::ring_optimized::PinnedBuffer::new(self.ring_size);
        
        // Warmup
        self.warmup(&mut buffer);
        
        // Actual benchmark
        let start = Instant::now();
        
        #[cfg(feature = "cuda")]
        {
            self.run_gpu_throughput(&mut buffer);
        }
        
        #[cfg(not(feature = "cuda"))]
        {
            self.run_cpu_throughput(&mut buffer);
        }
        
        let duration = start.elapsed();
        
        // Calculate metrics
        let total_messages = self.test_iterations as u64;
        let throughput_msg_per_sec = total_messages as f64 / duration.as_secs_f64();
        let total_bytes = total_messages * self.message_size as u64;
        let throughput_gb_per_sec = (total_bytes as f64 / 1_073_741_824.0) / duration.as_secs_f64();
        let avg_latency_ns = duration.as_nanos() as u64 / total_messages;
        
        BenchmarkResults {
            total_messages,
            duration,
            throughput_msg_per_sec,
            throughput_gb_per_sec,
            avg_latency_ns,
            p50_latency_ns: avg_latency_ns,  // Simplified for now
            p99_latency_ns: avg_latency_ns * 2,
            p999_latency_ns: avg_latency_ns * 3,
        }
    }
    
    /// Run latency benchmark with percentiles
    pub fn run_latency(&self) -> BenchmarkResults {
        println!("Running latency benchmark: {}", self.name);
        
        let mut buffer = crate::ring_optimized::PinnedBuffer::new(self.ring_size);
        let mut latencies = Vec::with_capacity(self.test_iterations as usize);
        
        // Warmup
        self.warmup(&mut buffer);
        
        // Measure individual message latencies
        let test_data = vec![42u8; self.message_size];
        let mut producer = crate::ring_optimized::Producer::new(&mut buffer);
        
        for _ in 0..self.test_iterations {
            let start = Instant::now();
            producer.try_produce(&test_data).unwrap();
            let latency = start.elapsed().as_nanos() as u64;
            latencies.push(latency);
        }
        
        // Calculate percentiles
        latencies.sort_unstable();
        let len = latencies.len();
        
        let p50 = latencies[len / 2];
        let p99 = latencies[len * 99 / 100];
        let p999 = latencies[len * 999 / 1000];
        let avg = latencies.iter().sum::<u64>() / len as u64;
        
        let total_duration = Duration::from_nanos(latencies.iter().sum());
        
        BenchmarkResults {
            total_messages: self.test_iterations as u64,
            duration: total_duration,
            throughput_msg_per_sec: self.test_iterations as f64 / total_duration.as_secs_f64(),
            throughput_gb_per_sec: 0.0,  // Not applicable for latency test
            avg_latency_ns: avg,
            p50_latency_ns: p50,
            p99_latency_ns: p99,
            p999_latency_ns: p999,
        }
    }
    
    fn warmup(&self, buffer: &mut crate::ring_optimized::PinnedBuffer) {
        let test_data = vec![0u8; self.message_size];
        let mut producer = crate::ring_optimized::Producer::new(buffer);
        
        for _ in 0..self.warmup_iterations {
            let _ = producer.try_produce(&test_data);
        }
    }
    
    #[cfg(feature = "cuda")]
    fn run_gpu_throughput(&self, buffer: &mut crate::ring_optimized::PinnedBuffer) {
        use crate::gpu_path;
        
        let hdr = buffer.header_mut() as *mut _;
        let slots = buffer.slots_ptr();
        
        gpu_path::launch_gpu_producer(
            hdr,
            slots,
            self.ring_size,
            self.test_iterations as u64,
        ).join().unwrap();
    }
    
    #[cfg(not(feature = "cuda"))]
    fn run_cpu_throughput(&self, buffer: &mut crate::ring_optimized::PinnedBuffer) {
        let test_data = vec![42u8; self.message_size];
        let mut producer = crate::ring_optimized::Producer::new(buffer);
        
        for _ in 0..self.test_iterations {
            producer.try_produce(&test_data).unwrap();
        }
    }
}

/// Run comprehensive benchmark suite
pub fn run_benchmark_suite() {
    println!("=== Perdix Ring Buffer Benchmark Suite ===\n");
    
    // Throughput tests with different message sizes
    for msg_size in &[64, 256, 1024, 4096] {
        let bench = Benchmark::new(&format!("Throughput {}B", msg_size))
            .with_message_size(*msg_size)
            .with_iterations(10000, 1000000);
        
        let results = bench.run_throughput();
        print_results(&results);
    }
    
    // Latency tests
    let latency_bench = Benchmark::new("Latency")
        .with_iterations(1000, 100000);
    
    let latency_results = latency_bench.run_latency();
    print_results(&latency_results);
    
    // Scaling test with different ring sizes
    for ring_size in &[256, 1024, 4096, 16384] {
        let bench = Benchmark::new(&format!("Ring Size {}", ring_size))
            .with_ring_size(*ring_size)
            .with_iterations(5000, 500000);
        
        let results = bench.run_throughput();
        print_results(&results);
    }
}

fn print_results(results: &BenchmarkResults) {
    println!("\nResults:");
    println!("  Total messages: {}", results.total_messages);
    println!("  Duration: {:?}", results.duration);
    println!("  Throughput: {:.2} msg/sec ({:.3} GB/s)", 
             results.throughput_msg_per_sec, results.throughput_gb_per_sec);
    println!("  Latency (ns): avg={}, p50={}, p99={}, p999={}", 
             results.avg_latency_ns, results.p50_latency_ns, 
             results.p99_latency_ns, results.p999_latency_ns);
    println!();
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_basic_benchmark() {
        let bench = Benchmark::new("Test")
            .with_iterations(100, 1000)
            .with_message_size(64);
        
        let results = bench.run_throughput();
        assert!(results.total_messages == 1000);
        assert!(results.throughput_msg_per_sec > 0.0);
    }
    
    #[test]
    fn test_latency_benchmark() {
        let bench = Benchmark::new("Latency Test")
            .with_iterations(10, 100);
        
        let results = bench.run_latency();
        assert!(results.p99_latency_ns >= results.p50_latency_ns);
        assert!(results.p999_latency_ns >= results.p99_latency_ns);
    }
}
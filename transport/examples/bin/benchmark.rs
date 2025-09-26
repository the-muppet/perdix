//! Comprehensive performance benchmark for Perdix
//! Measures throughput, latency, and efficiency like kitty terminal
//!
//! Run with: cargo run --release --bin benchmark

use transport::Buffer;
use std::time::{Duration, Instant};

const WARMUP_ITERATIONS: usize = 1000;
const TEST_DURATION_SECS: u64 = 5;
const BUFFER_SIZE: usize = 4096;

#[derive(Debug)]
struct BenchmarkResult {
    name: String,
    throughput_mbps: f64,
    messages_per_sec: f64,
    avg_latency_ns: u64,
    min_latency_ns: u64,
    max_latency_ns: u64,
    p99_latency_ns: u64,
}

fn main() -> Result<(), String> {
    println!("=== Perdix Performance Benchmark ===");
    println!("Measuring throughput, latency, and efficiency\n");

    // Create single persistent buffer for all tests
    let mut buffer = Buffer::new(BUFFER_SIZE)?;
    println!("Buffer initialized with {} slots\n", BUFFER_SIZE);

    // Warmup
    println!("Warming up...");
    warmup(&mut buffer)?;

    // Run benchmarks
    let mut results = Vec::new();

    // Test 1: ASCII throughput
    println!("\n1. ASCII Text Throughput");
    results.push(benchmark_throughput(
        &mut buffer,
        "ASCII",
        generate_ascii_data(128),
    )?);

    // Test 2: Unicode throughput
    println!("\n2. Unicode Text Throughput");
    results.push(benchmark_throughput(
        &mut buffer,
        "Unicode",
        generate_unicode_data(128),
    )?);

    // Test 3: Terminal escape sequences (CSI)
    println!("\n3. CSI Escape Sequences Throughput");
    results.push(benchmark_throughput(
        &mut buffer,
        "CSI",
        generate_csi_data(128),
    )?);

    // Test 4: Binary data
    println!("\n4. Binary Data Throughput");
    results.push(benchmark_throughput(
        &mut buffer,
        "Binary",
        generate_binary_data(128),
    )?);

    // Test 5: Message latency
    println!("\n5. Single Message Latency");
    results.push(benchmark_latency(&mut buffer)?);

    // Test 6: Burst throughput
    println!("\n6. Burst Pattern Throughput");
    results.push(benchmark_burst(&mut buffer)?);

    // Print summary table
    print_results_table(&results);

    // Compare with theoretical limits
    print_theoretical_comparison(&results);

    Ok(())
}

fn warmup(buffer: &mut Buffer) -> Result<(), String> {
    let (mut producer, mut consumer) = buffer.split_mut();
    let data = vec![0u8; 64];

    for _ in 0..WARMUP_ITERATIONS {
        producer.try_produce(&data).ok();
        consumer.try_consume();
    }

    Ok(())
}

fn benchmark_throughput(
    buffer: &mut Buffer,
    name: &str,
    test_data: Vec<u8>,
) -> Result<BenchmarkResult, String> {
    let (mut producer, mut consumer) = buffer.split_mut();

    let start = Instant::now();
    let mut total_bytes = 0;
    let mut total_messages = 0;
    let mut latencies = Vec::new();

    // Run for fixed duration
    let test_duration = Duration::from_secs(TEST_DURATION_SECS);

    while start.elapsed() < test_duration {
        let msg_start = Instant::now();

        // Produce
        if producer.try_produce(&test_data).is_ok() {
            total_bytes += test_data.len();
            total_messages += 1;
        }

        // Consume
        if let Some(_) = consumer.try_consume() {
            let latency = msg_start.elapsed().as_nanos() as u64;
            latencies.push(latency);
        }
    }

    // Drain remaining
    while consumer.try_consume().is_some() {}

    let elapsed = start.elapsed();
    let throughput_mbps = (total_bytes as f64) / elapsed.as_secs_f64() / 1_000_000.0;
    let messages_per_sec = total_messages as f64 / elapsed.as_secs_f64();

    // Calculate latency stats
    latencies.sort_unstable();
    let avg_latency = if latencies.is_empty() {
        0
    } else {
        latencies.iter().sum::<u64>() / latencies.len() as u64
    };

    let min_latency = latencies.first().copied().unwrap_or(0);
    let max_latency = latencies.last().copied().unwrap_or(0);
    let p99_latency = latencies
        .get((latencies.len() as f64 * 0.99) as usize)
        .copied()
        .unwrap_or(0);

    println!("  Throughput: {:.2} MB/s", throughput_mbps);
    println!("  Messages: {:.0}/sec", messages_per_sec);
    println!("  Avg latency: {} ns", avg_latency);

    Ok(BenchmarkResult {
        name: name.to_string(),
        throughput_mbps,
        messages_per_sec,
        avg_latency_ns: avg_latency,
        min_latency_ns: min_latency,
        max_latency_ns: max_latency,
        p99_latency_ns: p99_latency,
    })
}

fn benchmark_latency(buffer: &mut Buffer) -> Result<BenchmarkResult, String> {
    let (mut producer, mut consumer) = buffer.split_mut();
    let test_msg = b"Latency test message";

    let mut latencies = Vec::new();

    // Measure 10000 round trips
    for _ in 0..10000 {
        let start = Instant::now();

        producer.try_produce(test_msg).expect("Failed to produce");
        let _msg = consumer.try_consume().expect("Failed to consume");

        let latency = start.elapsed().as_nanos() as u64;
        latencies.push(latency);
    }

    latencies.sort_unstable();

    let avg = latencies.iter().sum::<u64>() / latencies.len() as u64;
    let min = latencies[0];
    let max = latencies[latencies.len() - 1];
    let p99 = latencies[(latencies.len() as f64 * 0.99) as usize];

    println!("  Avg: {} ns", avg);
    println!("  Min: {} ns", min);
    println!("  Max: {} ns", max);
    println!("  P99: {} ns", p99);

    Ok(BenchmarkResult {
        name: "Latency".to_string(),
        throughput_mbps: 0.0,
        messages_per_sec: 1_000_000_000.0 / avg as f64,
        avg_latency_ns: avg,
        min_latency_ns: min,
        max_latency_ns: max,
        p99_latency_ns: p99,
    })
}

fn benchmark_burst(buffer: &mut Buffer) -> Result<BenchmarkResult, String> {
    let (mut producer, mut consumer) = buffer.split_mut();
    let burst_size = 1000;
    let msg_data = vec![0xAB; 128];

    let start = Instant::now();
    let mut total_bytes = 0;
    let mut total_messages = 0;

    // Run burst pattern for 5 seconds
    let test_duration = Duration::from_secs(TEST_DURATION_SECS);

    while start.elapsed() < test_duration {
        // Producer burst
        for _ in 0..burst_size {
            if producer.try_produce(&msg_data).is_ok() {
                total_bytes += msg_data.len();
                total_messages += 1;
            } else {
                break;
            }
        }

        // Consumer burst
        while consumer.try_consume().is_some() {}
    }

    let elapsed = start.elapsed();
    let throughput_mbps = (total_bytes as f64) / elapsed.as_secs_f64() / 1_000_000.0;
    let messages_per_sec = total_messages as f64 / elapsed.as_secs_f64();

    println!("  Burst size: {}", burst_size);
    println!("  Throughput: {:.2} MB/s", throughput_mbps);
    println!("  Messages: {:.0}/sec", messages_per_sec);

    Ok(BenchmarkResult {
        name: "Burst".to_string(),
        throughput_mbps,
        messages_per_sec,
        avg_latency_ns: 0,
        min_latency_ns: 0,
        max_latency_ns: 0,
        p99_latency_ns: 0,
    })
}

fn generate_ascii_data(size: usize) -> Vec<u8> {
    (0..size)
        .map(|i| {
            let chars = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 \n";
            chars[i % chars.len()]
        })
        .collect()
}

fn generate_unicode_data(size: usize) -> Vec<u8> {
    let unicode_str = "Hello 世界 🦀 Rust! ";
    unicode_str
        .bytes()
        .cycle()
        .take(size)
        .collect()
}

fn generate_csi_data(size: usize) -> Vec<u8> {
    let patterns: Vec<&[u8]> = vec![
        b"\x1b[31mRed\x1b[0m",
        b"\x1b[1mBold\x1b[0m",
        b"\x1b[2J\x1b[H",
        b"Normal text",
    ];

    let mut data = Vec::new();
    let mut i = 0;
    while data.len() < size {
        let pattern = patterns[i % patterns.len()];
        data.extend_from_slice(pattern);
        i += 1;
    }
    data.truncate(size);
    data
}

fn generate_binary_data(size: usize) -> Vec<u8> {
    (0..size).map(|i| (i * 7 + 13) as u8).collect()
}

fn print_results_table(results: &[BenchmarkResult]) {
    println!("\n=== PERFORMANCE SUMMARY ===");
    println!("{:<12} | {:>12} | {:>12} | {:>12} | {:>12}",
             "Test", "Throughput", "Messages/sec", "Avg Latency", "P99 Latency");
    println!("{:-<12}-+-{:-<12}-+-{:-<12}-+-{:-<12}-+-{:-<12}",
             "", "", "", "", "");

    for r in results {
        if r.throughput_mbps > 0.0 {
            println!("{:<12} | {:>10.2} MB/s | {:>12.0} | {:>9} ns | {:>9} ns",
                     r.name,
                     r.throughput_mbps,
                     r.messages_per_sec,
                     r.avg_latency_ns,
                     r.p99_latency_ns);
        } else {
            println!("{:<12} | {:>12} | {:>12.0} | {:>9} ns | {:>9} ns",
                     r.name,
                     "N/A",
                     r.messages_per_sec,
                     r.avg_latency_ns,
                     r.p99_latency_ns);
        }
    }

    // Calculate average throughput (excluding latency test)
    let avg_throughput: f64 = results
        .iter()
        .filter(|r| r.throughput_mbps > 0.0)
        .map(|r| r.throughput_mbps)
        .sum::<f64>()
        / results.iter().filter(|r| r.throughput_mbps > 0.0).count() as f64;

    println!("\nAverage throughput: {:.2} MB/s", avg_throughput);
}

fn print_theoretical_comparison(results: &[BenchmarkResult]) {
    println!("\n=== THEORETICAL LIMITS ===");

    // DDR4-3200 theoretical bandwidth: ~25.6 GB/s
    // PCIe 3.0 x16: ~16 GB/s
    // Typical L3 cache: ~200 GB/s

    let best_throughput = results
        .iter()
        .map(|r| r.throughput_mbps)
        .fold(0.0, f64::max);

    let cache_bandwidth = 200_000.0; // MB/s
    let memory_bandwidth = 25_600.0; // MB/s
    let pcie_bandwidth = 16_000.0;   // MB/s

    println!("  L3 Cache bandwidth:  {:>8.2} GB/s", cache_bandwidth / 1000.0);
    println!("  Memory bandwidth:    {:>8.2} GB/s", memory_bandwidth / 1000.0);
    println!("  PCIe 3.0 x16:        {:>8.2} GB/s", pcie_bandwidth / 1000.0);
    println!();
    println!("  Your throughput:     {:>8.2} GB/s", best_throughput / 1000.0);
    println!("  Cache utilization:   {:>8.2}%", (best_throughput / cache_bandwidth) * 100.0);
    println!("  Memory utilization:  {:>8.2}%", (best_throughput / memory_bandwidth) * 100.0);

    if best_throughput > 1000.0 {
        println!("\n✓ Excellent performance - achieving GB/s throughput!");
    } else if best_throughput > 500.0 {
        println!("\n✓ Good performance - achieving high throughput!");
    } else {
        println!("\n⚠ Performance seems limited - check implementation");
    }
}
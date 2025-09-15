#include <cuda_runtime.h>
         #include <cooperative_groups.h>
         #include <cuda/atomic>
         #include <stdint.h>
         #include <stdio.h>

         namespace cg = cooperative_groups;

         // ============================================================================
         // DATA STRUCTURES
         // ============================================================================

         // Agent types for AI response categorization
         enum class AgentType : uint8_t {
             SYSTEM = 0,      // Blue
             USER = 1,        // Green
             ASSISTANT = 2,   // Cyan
             ERROR_TYPE = 3,  // Red
             WARNING = 4,     // Yellow
             INFO = 5,        // White
             DEBUG = 6,       // Magenta
             TRACE = 7        // Bright Black
         };

         // slot structure - 256 bytes, cache-aligned
         struct __align__(256) Slot{
             // First cache line - hot data for consumer
             cuda::atomic<uint64_t, cuda::thread_scope_system> seq;         // 8 bytes - sequence number
             uint32_t len;                  // 4 bytes - payload length
             uint32_t flags;                // 4 bytes - metadata
             uint64_t timestamp;            // 8 bytes - timing
             uint8_t _pad1[40];            // Pad to 64 bytes

             // Payload in separate cache lines (192 bytes)
             union {
                 uint8_t  bytes[192];
                 uint4    vectors[12];     // For vectorized access
                 float4   float_vectors[12]; // Alternative vectorization
             } payload;
         };

         // header with better cache line usage
         struct __align__(256) Header{
             // Producer line - exclusively GPU accessed
             struct __align__(64) {
                 cuda::atomic<uint64_t, cuda::thread_scope_system> write_idx;
                 uint64_t messages_produced;
                 uint64_t last_batch_cycles;
                 uint8_t _pad[40];
             } producer;

             // Consumer line - exclusively CPU accessed
             struct __align__(64) {
                 volatile uint64_t read_idx;
                 uint64_t messages_consumed;
                 uint64_t last_drain_time;
                 uint8_t _pad[40];
             } consumer;

             // Configuration - read-only after init
             struct __align__(64) {
                 uint64_t wrap_mask;
                 uint32_t slot_count;
                 uint32_t payload_size;
                 uint32_t batch_size;
                 uint32_t backpressure_threshold;
                 uint8_t _pad[40];
             } config;

             // Control - infrequent access
             struct __align__(64) {
                 cuda::atomic<uint32_t> backpressure;
                 cuda::atomic<uint32_t> stop;
                 uint32_t epoch;
                 uint32_t error_count;
                 uint8_t _pad[48];
             } control;
         };

         // Stream context for coalesced access
         struct __align__(32) StreamContext{
             const uint8_t* __restrict__ text;  // 8 bytes
             uint32_t text_len;                 // 4 bytes
             uint32_t stream_id;                // 4 bytes
             uint64_t timestamp;                // 8 bytes
             AgentType agent_type;              // 1 byte
             uint8_t flags;                     // 1 byte (continuation, ansi)
             uint8_t _pad[6];                   // Pad to 32 bytes
         };

         // Performance metrics with better atomics
         struct KernelMetrics{
             uint64_t cycles_start;
             uint64_t cycles_end;
             uint32_t messages_processed;
             uint32_t atomic_retries;
             uint32_t backpressure_events;
             uint32_t cache_misses;
         };

         // ============================================================================
         // HELPER FUNCTIONS
         // ============================================================================

         // Compile-time ANSI codes table
         __constant__ char ANSI_CODES[8][6] = {
             "\033[34m",  // SYSTEM - Blue
             "\033[32m",  // USER - Green
             "\033[36m",  // ASSISTANT - Cyan
             "\033[31m",  // ERROR - Red
             "\033[33m",  // WARNING - Yellow
             "\033[37m",  // INFO - White
             "\033[35m",  // DEBUG - Magenta
             "\033[90m"   // TRACE - Bright Black
         };

         __constant__ char ANSI_RESET[5] = "\033[0m";

         // keyword detection using warp voting
         __device__ __forceinline__ bool detect_keyword_optimized(
             const uint8_t* __restrict__ text,
             uint32_t pos,
             uint32_t text_len,
             const uint32_t keyword_hash,
             uint32_t keyword_len
         ) {
             if (pos + keyword_len > text_len) return false;

             // Quick hash check first
             uint32_t hash = 0;
             #pragma unroll
             for (uint32_t i = 0; i < keyword_len && i < 8; i++) {
                 hash = hash * 31 + text[pos + i];
             }

             return hash == keyword_hash;
         }

         // Vectorized copy with optimal unrolling
         template<int BYTES>
         __device__ __forceinline__ void copy_payload_optimized(
             uint4* __restrict__ dst,
             const uint4* __restrict__ src
         ) {
             constexpr int VECTORS = BYTES / sizeof(uint4);

             #pragma unroll
             for (int i = 0; i < VECTORS; i++) {
                 // Use 128-bit loads/stores
                 dst[i] = __ldg(&src[i]);
             }
         }

         // ============================================================================
         // UNIFIED KERNEL
         // ============================================================================

         template<int BLOCK_SIZE = 256, int PAYLOAD_SIZE = 192, bool ENABLE_METRICS = false>
         __global__ void __launch_bounds__(BLOCK_SIZE, 4)
         unified_kernel_optimized(
             Slot* __restrict__ slots,
             Header* __restrict__ hdr,
             const StreamContext* __restrict__ contexts,
             const uint32_t n_messages,
             KernelMetrics* __restrict__ metrics
         ) {
             // Thread identifiers
             const int tid = threadIdx.x;
             const int bid = blockIdx.x;
             const int warp_id = tid >> 5;
             const int lane_id = tid & 31;
             const int global_tid = bid * BLOCK_SIZE + tid;

             // Shared memory layout for bank conflict avoidance
             extern __shared__ char shared_mem[];

             // Payload buffers with padding to avoid bank conflicts
             uint8_t* shared_payloads = (uint8_t*)shared_mem;
             // Sequence numbers in separate section
             uint64_t* shared_seqs = (uint64_t*)&shared_mem[BLOCK_SIZE * (PAYLOAD_SIZE + 8)];
             // Work indices for cooperative batching
             uint32_t* shared_work = (uint32_t*)&shared_seqs[BLOCK_SIZE];

             // Initialize metrics
             KernelMetrics local_metrics = {};
             if constexpr (ENABLE_METRICS) {
                 if (tid == 0) {
                     local_metrics.cycles_start = clock64();
                 }
             }

             // Get cooperative groups
             cg::thread_block block = cg::this_thread_block();
             cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

             // Cache configuration in registers
             const uint64_t wrap_mask = hdr->config.wrap_mask;
             const uint32_t backpressure_limit = hdr->config.backpressure_threshold;

             // Grid-stride loop with improved batching
             for (uint32_t msg_base = bid * BLOCK_SIZE;
                  msg_base < n_messages;
                  msg_base += gridDim.x * BLOCK_SIZE) {

                 const uint32_t msg_id = msg_base + tid;
                 const bool has_work = msg_id < n_messages;

                 // Early exit on stop signal
                 if (hdr->control.stop.load(cuda::memory_order_relaxed) != 0) {
                     break;
                 }

                 // OPTIMIZED: Warp-level batched sequence reservation
                 uint64_t warp_seq_base = UINT64_MAX;
                 uint32_t active_mask = __ballot_sync(0xFFFFFFFF, has_work);

                 if (active_mask != 0) {
                     uint32_t warp_count = __popc(active_mask);

                     // Single thread reserves for entire warp
                     if (lane_id == 0 && warp_id == 0) {
                         // Check backpressure with relaxed ordering
                         uint64_t current_write = hdr->producer.write_idx.load(cuda::memory_order_relaxed);
                         uint64_t current_read = *((volatile uint64_t*)&hdr->consumer.read_idx);

                         if ((current_write - current_read) < backpressure_limit) {
                             // Atomic fetch_add with explicit memory ordering
                             warp_seq_base = hdr->producer.write_idx.fetch_add(
                                 warp_count,
                                 cuda::memory_order_acq_rel
                             );

                             if constexpr (ENABLE_METRICS) {
                                 atomicAdd(&hdr->producer.messages_produced, warp_count);
                             }
                         } else {
                             // Signal backpressure
                             hdr->control.backpressure.store(1, cuda::memory_order_release);

                             if constexpr (ENABLE_METRICS) {
                                 local_metrics.backpressure_events++;
                             }
                         }
                     }

                     // Efficient broadcast using shuffle
                     warp_seq_base = __shfl_sync(active_mask, warp_seq_base, 0);
                 }

                 if (warp_seq_base == UINT64_MAX) {
                     // Exponential backoff instead of fixed delay
                     uint32_t backoff = 10 << (tid & 3);
                     __nanosleep(backoff);
                     continue;
                 }

                 // Calculate per-thread sequence
                 const uint32_t thread_offset = __popc(active_mask & ((1U << lane_id) - 1));
                 const uint64_t my_seq = warp_seq_base + thread_offset;
                 shared_seqs[tid] = my_seq;

                 // Load context with coalesced access
                 StreamContextctx;
                 if (has_work) {
                     ctx = contexts[msg_id];
                 }

                 // Build message in shared memory
                 uint8_t* my_payload = &shared_payloads[tid * (PAYLOAD_SIZE + 8)];
                 uint32_t output_len = 0;

                 // ANSI formatting
                 if (ctx.flags & 0x02) { // ANSI enabled flag
                     // Direct copy from constant memory
                     const char* color = ANSI_CODES[static_cast<uint8_t>(ctx.agent_type)];
                     #pragma unroll
                     for (int i = 0; i < 5; i++) {
                         my_payload[output_len++] = color[i];
                     }
                 }

                 // text copy with bounds checking
                 const uint32_t copy_len = min(ctx.text_len,
                                               static_cast<uint32_t>(PAYLOAD_SIZE - output_len - 4));

                 // Vectorized copy for aligned data
                 if ((uintptr_t)ctx.text % 16 == 0 && copy_len >= 16) {
                     uint4* dst_vec = (uint4*)&my_payload[output_len];
                     const uint4* src_vec = (const uint4*)ctx.text;
                     const uint32_t vec_count = copy_len / 16;

                     #pragma unroll 4
                     for (uint32_t i = 0; i < vec_count; i++) {
                         dst_vec[i] = __ldg(&src_vec[i]);
                     }
                     output_len += vec_count * 16;

                     // Handle remainder
                     for (uint32_t i = vec_count * 16; i < copy_len; i++) {
                         my_payload[output_len++] = ctx.text[i];
                     }
                 } else {
                     // Byte-wise copy for unaligned data
                     #pragma unroll 8
                     for (uint32_t i = 0; i < copy_len; i++) {
                         my_payload[output_len++] = ctx.text[i];
                     }
                 }

                 // Add reset sequence if needed
                 if ((ctx.flags & 0x02) && !(ctx.flags & 0x01)) {
                     #pragma unroll
                     for (int i = 0; i < 4; i++) {
                         my_payload[output_len++] = ANSI_RESET[i];
                     }
                 }

                 // Synchronize shared memory writes
                 __syncwarp(active_mask);

                 // Calculate slot with improved masking
                 const uint64_t slot_idx = my_seq & wrap_mask;
                 Slot* my_slot = &slots[slot_idx];

                 // OPTIMIZED: Prefetch slot line
                 #if __CUDA_ARCH__ >= 800
                 __builtin_prefetch(my_slot, 1, 3); // Write, L1 cache
                 #endif

                 // OPTIMIZED: Vectorized payload write
                 copy_payload_optimized<PAYLOAD_SIZE>(
                     my_slot->payload.vectors,
                     (uint4*)my_payload
                 );

                 // Write metadata
                 my_slot->len = output_len;
                 my_slot->flags = (ctx.stream_id << 16) |
                                 (static_cast<uint32_t>(ctx.agent_type) << 8) |
                                 ctx.flags;
                 my_slot->timestamp = clock64();

                 // OPTIMIZED: Single fence for all writes
                 __threadfence_system();

                 // Atomic publish with release semantics
                 my_slot->seq.store(my_seq, cuda::memory_order_release);

                 if constexpr (ENABLE_METRICS) {
                     local_metrics.messages_processed++;
                 }
             }

             // Finalize metrics
             if constexpr (ENABLE_METRICS) {
                 if (tid == 0) {
                     local_metrics.cycles_end = clock64();
                     metrics[bid] = local_metrics;
                 }
             }
         }

         // ============================================================================
         // ULTRA-PERSISTENT KERNEL
         // ============================================================================

         template<int WARPS_PER_BLOCK = 4>
         __global__ void __launch_bounds__(WARPS_PER_BLOCK * 32, 8)
         persistent_kernel_ultra(
             Slot* __restrict__ slots,
             Header* __restrict__ hdr,
             StreamContext* __restrict__ work_queue,
             cuda::atomic<uint64_t>* __restrict__ work_counter,
             const uint32_t spin_limit
         ) {
             const int tid = threadIdx.x;
             const int warp_id = tid >> 5;
             const int lane_id = tid & 31;

             // Local buffer in registers
             uint4 local_vectors[12];

             // Spinning loop for ultra-low latency
             uint32_t spins = 0;
             while (spins++ < spin_limit) {
                 // Check stop signal
                 if (hdr->control.stop.load(cuda::memory_order_acquire) != 0) {
                     break;
                 }

                 // Try to claim work cooperatively at warp level
                 uint64_t work_id = UINT64_MAX;

                 if (lane_id == 0) {
                     // Use compare-exchange for better contention handling
                     uint64_t expected = work_counter->load(cuda::memory_order_relaxed);
                     uint64_t desired = expected + 1;

                     if (work_counter->compare_exchange_weak(
                         expected, desired,
                         cuda::memory_order_acq_rel,
                         cuda::memory_order_relaxed)) {
                         work_id = expected;
                     }
                 }

                 // Broadcast work ID
                 work_id = __shfl_sync(0xFFFFFFFF, work_id, 0);

                 if (work_id == UINT64_MAX || work_queue[work_id].text == nullptr) {
                     // Adaptive spinning with exponential backoff
                     uint32_t backoff = 1 << min(spins / 1000, 10U);
                     __nanosleep(backoff);
                     continue;
                 }

                 // Process work with zero-copy access
                 StreamContext* work = &work_queue[work_id];

                 // Reserve slot with single atomic
                 uint64_t seq = hdr->producer.write_idx.fetch_add(1, cuda::memory_order_acq_rel);
                 uint64_t slot_idx = seq & hdr->config.wrap_mask;
                 Slot* slot = &slots[slot_idx];

                 // Build message directly in registers
                 uint32_t output_len = 0;
                 uint8_t* output = (uint8_t*)local_vectors;

                 // Fast path for common case
                 if (work->text_len <= 192) {
                     // Vectorized copy from pinned memory
                     const uint4* src = (const uint4*)work->text;
                     uint32_t vec_count = (work->text_len + 15) / 16;

                     #pragma unroll
                     for (uint32_t i = 0; i < vec_count && i < 12; i++) {
                         local_vectors[i] = __ldg(&src[i]);
                     }
                     output_len = work->text_len;
                 }

                 // Direct write to slot
                 #pragma unroll
                 for (int i = 0; i < 12; i++) {
                     slot->payload.vectors[i] = local_vectors[i];
                 }

                 slot->len = output_len;
                 slot->flags = work->stream_id;
                 slot->timestamp = clock64();

                 // Single fence and publish
                 __threadfence_system();
                 slot->seq.store(seq, cuda::memory_order_release);

                 // Reset work item
                 atomicExch((unsigned long long*)&work->text, 0);

                 // Reset spin counter on successful work
                 spins = 0;
             }
         }

         // ============================================================================
         // LAUNCH FUNCTIONS
         // ============================================================================

         extern "C" int launch_optimized_kernel(
             void* slots,
             void* hdr,
             void* contexts,
             uint32_t n_messages,
             bool enable_metrics,
             cudaStream_t stream
         ) {
             // Get device properties
             cudaDeviceProp prop;
             cudaGetDeviceProperties(&prop, 0);

             // Optimal configuration for RTX 4070 (SM 8.9)
             const int threads_per_block = 256;
             const int target_occupancy = 50; // 50% occupancy for better cache usage

             // Calculate blocks based on SM count and target occupancy
             const int max_blocks_per_sm = prop.maxThreadsPerMultiProcessor / threads_per_block;
             const int blocks_for_occupancy = (prop.multiProcessorCount * max_blocks_per_sm * target_occupancy) / 100;

             // Adjust for workload
             int blocks = min(
                 (n_messages + threads_per_block - 1) / threads_per_block,
                 blocks_for_occupancy
             );
             blocks = max(blocks, prop.multiProcessorCount); // At least 1 per SM

             // Calculate shared memory
             const size_t shared_size = threads_per_block * (192 + 8) + // Payloads
                                       threads_per_block * sizeof(uint64_t) + // Sequences
                                       threads_per_block * sizeof(uint32_t);  // Work indices

             // Configure cache
             cudaFuncSetAttribute(
                 unified_kernel_optimized<256, 192, false>,
                 cudaFuncAttributePreferredSharedMemoryCarveout,
                 50 // 50% shared memory
             );

             // Allocate metrics if needed
             KernelMetrics* d_metrics = nullptr;
             if (enable_metrics) {
                 cudaMalloc(&d_metrics, blocks * sizeof(KernelMetrics));
             }

             // Launch kernel
             if (enable_metrics) {
                 unified_kernel_optimized<256, 192, true><<<blocks, threads_per_block, shared_size, stream>>>(
                     (Slot*)slots,
                     (Header*)hdr,
                     (StreamContext*)contexts,
                     n_messages,
                     d_metrics
                 );
             } else {
                 unified_kernel_optimized<256, 192, false><<<blocks, threads_per_block, shared_size, stream>>>(
                     (Slot*)slots,
                     (Header*)hdr,
                     (StreamContext*)contexts,
                     n_messages,
                     nullptr
                 );
             }

             cudaError_t err = cudaGetLastError();
             if (err != cudaSuccess) {
                 if (d_metrics) cudaFree(d_metrics);
                 return -1;
             }

             if (enable_metrics && d_metrics) {
                 cudaStreamSynchronize(stream);

                 // Process metrics
                 KernelMetrics* h_metrics = new KernelMetrics[blocks];
                 cudaMemcpy(h_metrics, d_metrics, blocks * sizeof(KernelMetrics),
                           cudaMemcpyDeviceToHost);

                 uint64_t total_cycles = 0;
                 uint32_t total_messages = 0;

                 for (int i = 0; i < blocks; i++) {
                     total_cycles += (h_metrics[i].cycles_end - h_metrics[i].cycles_start);
                     total_messages += h_metrics[i].messages_processed;
                 }

                 printf("Kernel Performance:\n");
                 printf("  Messages: %u\n", total_messages);
                 printf("  Avg cycles/message: %llu\n", total_cycles / max(total_messages, 1U));
                 printf("  Throughput: %.2f GB/s\n",
                        (total_messages * 192.0) / (total_cycles / prop.clockRate / 1000.0) / 1e9);

                 delete[] h_metrics;
                 cudaFree(d_metrics);
             }

             return 0;
         }

         extern "C" int launch_ultra_persistent_kernel(
             void* slots,
             void* hdr,
             void* work_queue,
             void* work_counter,
             uint32_t spin_limit,
             cudaStream_t stream
         ) {
             cudaDeviceProp prop;
             cudaGetDeviceProperties(&prop, 0);

             // Use exactly 1 block per SM for persistent kernel
             const int blocks = prop.multiProcessorCount;
             const int threads = 128; // 4 warps per block

             // Configure for maximum L1 cache
             cudaFuncSetAttribute(
                 persistent_kernel_ultra<4>,
                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                 0 // No dynamic shared memory
             );

             persistent_kernel_ultra<4><<<blocks, threads, 0, stream>>>(
                 (Slot*)slots,
                 (Header*)hdr,
                 (StreamContext*)work_queue,
                 (cuda::atomic<uint64_t>*)work_counter,
                 spin_limit
             );

             return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
         }
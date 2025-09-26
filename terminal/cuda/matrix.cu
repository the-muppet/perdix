/**
 * @file batch_optimizer.cu
 * @brief Optimized Batch Processing with Warp-Level Primitives
 *
 * This implementation provides:
 * - Warp-level cooperative processing
 * - Coalesced memory access patterns
 * - Minimal atomic contention through batching
 * - Efficient memory bandwidth utilization
 */

 #include <cuda_runtime.h>
 #include <cuda.h>
 #include <cooperative_groups.h>
 #include <cooperative_groups/reduce.h>
 #include <cuda/atomic>
 #include <stdint.h>
 #include <stdio.h>
 
 namespace cg = cooperative_groups;
 
 // ============================================================================
 // Advanced Data Structures
 // ============================================================================
 
 /**
  * @struct WarpBatch
  * @brief Batch descriptor for warp-level processing
  */
 struct WarpBatch {
     uint32_t message_ids[32];     // Message IDs for each lane
     uint32_t total_bytes;          // Total bytes in batch
     uint32_t active_mask;          // Mask of active lanes
     uint64_t base_sequence;        // Base sequence number
 };
 
 /**
  * @struct MessageDescriptor
  * @brief Compact message descriptor for batch processing
  */
 struct alignas(16) MessageDescriptor {
     uint32_t offset;               // Offset in data buffer
     uint16_t length;               // Message length
     uint8_t agent_type;            // Agent type
     uint8_t flags;                 // Message flags
     uint32_t timestamp;            // Timestamp
     uint32_t checksum;             // Optional checksum
 };
 
 /**
  * @struct ProcessingMetrics
  * @brief Performance metrics for batch processing
  */
 struct ProcessingMetrics {
     uint64_t messages_processed;
     uint64_t bytes_processed;
     uint64_t cycles_start;
     uint64_t cycles_end;
     uint32_t atomic_conflicts;
     uint32_t memory_transactions;
     float bandwidth_gbps;
     float messages_per_second;
 };
 
 /**
  * @struct OptimizedSlot
  * @brief Optimized slot structure for batch writes
  */
 struct alignas(256) OptimizedSlot {
     // Metadata cache line (64 bytes)
     volatile uint64_t sequence;
     uint32_t length;
     uint32_t checksum;
     uint16_t batch_id;
     uint8_t agent_type;
     uint8_t flags;
     uint32_t timestamp;
     uint8_t _pad1[40];
 
     // Payload (192 bytes)
     uint8_t payload[192];
 };
 
 static_assert(sizeof(OptimizedSlot) == 256, "OptimizedSlot must be 256 bytes");
 
 // ============================================================================
 // Warp-Level Primitives
 // ============================================================================
 
 /**
  * @brief Warp-level reduction for summing values
  */
 template<typename T>
 __device__ T warp_reduce_sum(T value) {
     unsigned mask = 0xFFFFFFFF;
     for (int offset = 16; offset > 0; offset /= 2) {
         value += __shfl_down_sync(mask, value, offset);
     }
     return value;
 }
 
 /**
  * @brief Warp-level scan (prefix sum)
  */
 template<typename T>
 __device__ T warp_scan_exclusive(T value, T* total = nullptr) {
     unsigned mask = 0xFFFFFFFF;
     T temp;
 
     // Scan within warp
     #pragma unroll
     for (int offset = 1; offset < 32; offset *= 2) {
         temp = __shfl_up_sync(mask, value, offset);
         if (threadIdx.x % 32 >= offset) {
             value += temp;
         }
     }
 
     // Get total if requested
     if (total) {
         *total = __shfl_sync(mask, value, 31);
     }
 
     // Convert to exclusive scan
     temp = __shfl_up_sync(mask, value, 1);
     return (threadIdx.x % 32 == 0) ? 0 : temp;
 }
 
 /**
  * @brief Warp-level ballot for finding active threads
  */
 __device__ uint32_t warp_ballot(bool predicate) {
     return __ballot_sync(0xFFFFFFFF, predicate);
 }
 
 /**
  * @brief Count leading zeros in warp ballot
  */
 __device__ int warp_find_first_set(uint32_t mask) {
     return __ffs(mask) - 1;
 }
 
 // ============================================================================
 // Memory Access Optimization
 // ============================================================================
 
 /**
  * @brief Coalesced read using vectorized loads
  */
 template<typename T>
 __device__ void coalesced_read(
     T* dst,
     const T* src,
     uint32_t count,
     uint32_t lane_id
 ) {
     // Use 128-bit loads for maximum bandwidth
     using Vec4 = uint4;
     const uint32_t vec_size = sizeof(Vec4) / sizeof(T);
 
     uint32_t vec_count = count / vec_size;
     Vec4* dst_vec = reinterpret_cast<Vec4*>(dst);
     const Vec4* src_vec = reinterpret_cast<const Vec4*>(src);
 
     // Vectorized loads
     for (uint32_t i = lane_id; i < vec_count; i += 32) {
         dst_vec[i] = src_vec[i];
     }
 
     // Handle remainder
     uint32_t remainder_start = vec_count * vec_size;
     for (uint32_t i = remainder_start + lane_id; i < count; i += 32) {
         dst[i] = src[i];
     }
 }
 
 /**
  * @brief Coalesced write with write combining
  */
 template<typename T>
 __device__ void coalesced_write(
     T* dst,
     const T* src,
     uint32_t count,
     uint32_t lane_id
 ) {
     // Use L2 cache write-through policy for streaming
     using Vec4 = uint4;
     const uint32_t vec_size = sizeof(Vec4) / sizeof(T);
 
     uint32_t vec_count = count / vec_size;
     Vec4* dst_vec = reinterpret_cast<Vec4*>(dst);
     const Vec4* src_vec = reinterpret_cast<const Vec4*>(src);
 
     // Vectorized stores with streaming hint
     #pragma unroll 4
     for (uint32_t i = lane_id; i < vec_count; i += 32) {
         // Use cache streaming hint for write-through
         __stcs(&dst_vec[i], src_vec[i]);
     }
 
     // Handle remainder
     uint32_t remainder_start = vec_count * vec_size;
     for (uint32_t i = remainder_start + lane_id; i < count; i += 32) {
         dst[i] = src[i];
     }
 }
 
 // ============================================================================
 // Optimized Batch Processing Kernel
 // ============================================================================
 
 /**
  * @brief Highly optimized batch processing kernel
  */
 __global__ void optimized_batch_kernel(
     const uint8_t* __restrict__ input_data,
     const MessageDescriptor* __restrict__ descriptors,
     uint32_t num_messages,
     OptimizedSlot* __restrict__ output_slots,
     volatile uint64_t* __restrict__ global_write_seq,
     uint32_t slot_mask,
     ProcessingMetrics* __restrict__ metrics,
     const bool enable_prefetch
 ) {
     // Thread and warp identification
     const uint32_t tid = threadIdx.x;
     const uint32_t warp_id = tid / 32;
     const uint32_t lane_id = tid % 32;
     const uint32_t warps_per_block = blockDim.x / 32;
     const uint32_t global_warp_id = blockIdx.x * warps_per_block + warp_id;
     const uint32_t total_warps = gridDim.x * warps_per_block;
 
     // Cooperative groups
     cg::thread_block block = cg::this_thread_block();
     cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
 
     // Shared memory for warp coordination
     extern __shared__ uint8_t shared_mem[];
     WarpBatch* warp_batches = reinterpret_cast<WarpBatch*>(shared_mem);
     uint8_t* shared_buffer = &shared_mem[sizeof(WarpBatch) * warps_per_block];
 
     // Metrics
     uint64_t local_messages = 0;
     uint64_t local_bytes = 0;
     uint64_t cycles_start = clock64();
 
     // Grid-stride loop for load balancing
     for (uint32_t batch_start = global_warp_id * 32;
          batch_start < num_messages;
          batch_start += total_warps * 32) {
 
         // Phase 1: Load message descriptors (coalesced)
         uint32_t msg_id = batch_start + lane_id;
         bool has_message = msg_id < num_messages;
 
         MessageDescriptor desc = {};
         if (has_message) {
             desc = descriptors[msg_id];
         }
 
         // Phase 2: Calculate batch statistics using warp primitives
         uint32_t active_mask = warp_ballot(has_message);
         uint32_t active_count = __popc(active_mask);
 
         if (active_count == 0) continue;
 
         // Calculate total bytes in batch
         uint32_t batch_bytes = warp_reduce_sum(has_message ? desc.length : 0);
 
         // Phase 3: Reserve sequences atomically (single atomic per warp)
         uint64_t base_seq = 0;
         if (lane_id == warp_find_first_set(active_mask)) {
             base_seq = atomicAdd((unsigned long long*)global_write_seq, active_count);
         }
         base_seq = __shfl_sync(active_mask, base_seq, warp_find_first_set(active_mask));
 
         // Phase 4: Calculate per-thread sequence using warp scan
         uint32_t thread_offset = __popc(active_mask & ((1U << lane_id) - 1));
         uint64_t my_seq = base_seq + thread_offset;
 
         // Phase 5: Prefetch slot locations (Ampere+ feature)
         #if __CUDA_ARCH__ >= 800
         if (enable_prefetch && has_message) {
             uint32_t slot_idx = my_seq & slot_mask;
             OptimizedSlot* slot = &output_slots[slot_idx];
             // Prefetch slot to L2 cache
             __builtin_prefetch(slot, 1, 3);
         }
         #endif
 
         // Phase 6: Process messages in parallel
         if (has_message) {
             uint32_t slot_idx = my_seq & slot_mask;
             OptimizedSlot* slot = &output_slots[slot_idx];
 
             // Load message data into shared memory for better access pattern
             uint32_t shared_offset = lane_id * 256;
             const uint8_t* msg_data = &input_data[desc.offset];
 
             // Optimized copy to shared memory
             if (desc.length <= 192) {
                 // Fast path for small messages (most common case)
                 uint4* shared_vec = reinterpret_cast<uint4*>(&shared_buffer[shared_offset]);
                 const uint4* msg_vec = reinterpret_cast<const uint4*>(msg_data);
 
                 uint32_t vec_count = (desc.length + 15) / 16;
                 #pragma unroll 4
                 for (uint32_t i = 0; i < vec_count && i < 12; i++) {
                     shared_vec[i] = msg_vec[i];
                 }
             }
 
             // Synchronize shared memory
             __syncwarp(active_mask);
 
             // Calculate checksum using warp shuffle
             uint32_t checksum = 0;
             if (desc.length > 0) {
                 #pragma unroll 4
                 for (uint32_t i = 0; i < desc.length && i < 192; i += 4) {
                     uint32_t word = *reinterpret_cast<const uint32_t*>(&shared_buffer[shared_offset + i]);
                     checksum ^= word;
                 }
                 // Reduce checksum across warp
                 checksum = warp_reduce_sum(checksum);
             }
 
             // Write to slot
             slot->length = desc.length;
             slot->checksum = checksum;
             slot->batch_id = global_warp_id;
             slot->agent_type = desc.agent_type;
             slot->flags = desc.flags;
             slot->timestamp = desc.timestamp;
 
             // Copy payload from shared memory (already coalesced)
             uint4* slot_vec = reinterpret_cast<uint4*>(slot->payload);
             uint4* shared_vec = reinterpret_cast<uint4*>(&shared_buffer[shared_offset]);
 
             #pragma unroll 8
             for (uint32_t i = 0; i < 12; i++) {  // 192 bytes = 12 * 16 bytes
                 slot_vec[i] = shared_vec[i];
             }
 
             // Memory fence for CPU visibility
             __threadfence_system();
 
             // Publish slot
             atomicExch((unsigned long long*)&slot->sequence, my_seq);
 
             // Update local metrics
             local_messages++;
             local_bytes += desc.length;
         }
 
         // Synchronize warp before next iteration
         warp.sync();
     }
 
     // Phase 7: Reduce metrics across block
     __shared__ uint64_t block_messages;
     __shared__ uint64_t block_bytes;
 
     if (tid == 0) {
         block_messages = 0;
         block_bytes = 0;
     }
     __syncthreads();
 
     // Warp-level reduction first
     local_messages = warp_reduce_sum(local_messages);
     local_bytes = warp_reduce_sum(local_bytes);
 
     // Block-level reduction
     if (lane_id == 0) {
         atomicAdd((unsigned long long*)&block_messages, local_messages);
         atomicAdd((unsigned long long*)&block_bytes, local_bytes);
     }
     __syncthreads();
 
     // Write metrics (block leader only)
     if (tid == 0 && metrics) {
         uint64_t cycles_end = clock64();
         atomicAdd((unsigned long long*)&metrics->messages_processed, block_messages);
         atomicAdd((unsigned long long*)&metrics->bytes_processed, block_bytes);
 
         // Update timing
         if (metrics->cycles_start == 0) {
             metrics->cycles_start = cycles_start;
         }
         metrics->cycles_end = cycles_end;
 
         // Calculate bandwidth
         float elapsed_ms = (cycles_end - cycles_start) / 1000000.0f;  // Approximate
         if (elapsed_ms > 0) {
             metrics->bandwidth_gbps = (block_bytes / (1024.0f * 1024.0f * 1024.0f)) / (elapsed_ms / 1000.0f);
             metrics->messages_per_second = block_messages / (elapsed_ms / 1000.0f);
         }
     }
 }
 
 // ============================================================================
 // Multi-Stream Batch Processor
 // ============================================================================
 
 /**
  * @brief Process multiple streams concurrently with load balancing
  */
 __global__ void multi_stream_batch_kernel(
     const uint8_t** __restrict__ stream_buffers,      // Array of stream buffers
     const MessageDescriptor** __restrict__ stream_descriptors,  // Descriptors per stream
     const uint32_t* __restrict__ stream_counts,       // Message count per stream
     uint32_t num_streams,                             // Number of streams
     OptimizedSlot* __restrict__ output_slots,
     volatile uint64_t* __restrict__ global_write_seq,
     uint32_t slot_mask,
     volatile uint32_t* __restrict__ stream_progress   // Progress per stream
 ) {
     // Thread identification
     const uint32_t tid = threadIdx.x;
     const uint32_t lane_id = tid % 32;
 
     // Cooperative groups
     cg::thread_block_tile<32> warp = cg::tiled_partition<32>(cg::this_thread_block());
 
     // Work stealing queue in shared memory
     __shared__ uint32_t work_queue[256];
     __shared__ uint32_t queue_head;
     __shared__ uint32_t queue_tail;
 
     // Initialize work queue with all streams
     if (tid == 0) {
         queue_head = 0;
         queue_tail = num_streams;
         for (uint32_t i = 0; i < num_streams; i++) {
             work_queue[i] = i;
         }
     }
     __syncthreads();
 
     // Main processing loop with work stealing
     while (true) {
         // Try to get work
         uint32_t stream_id = UINT32_MAX;
         if (lane_id == 0) {
             uint32_t head = atomicAdd(&queue_head, 0);
             uint32_t tail = atomicAdd(&queue_tail, 0);
 
             if (head < tail) {
                 stream_id = work_queue[atomicAdd(&queue_head, 1) % 256];
             }
         }
         stream_id = __shfl_sync(0xFFFFFFFF, stream_id, 0);
 
         if (stream_id == UINT32_MAX) break;  // No more work
 
         // Get stream information
         const uint8_t* stream_buffer = stream_buffers[stream_id];
         const MessageDescriptor* descriptors = stream_descriptors[stream_id];
         uint32_t count = stream_counts[stream_id];
         uint32_t progress = atomicAdd((unsigned int*)&stream_progress[stream_id], 0);
 
         // Process batch from this stream
         const uint32_t batch_size = 32;
         if (progress < count) {
             uint32_t batch_end = min(progress + batch_size, count);
             uint32_t my_msg_id = progress + lane_id;
             bool has_message = my_msg_id < batch_end;
 
             // Load descriptor
             MessageDescriptor desc = {};
             if (has_message) {
                 desc = descriptors[my_msg_id];
             }
 
             // Reserve sequences
             uint32_t active_mask = warp_ballot(has_message);
             uint32_t active_count = __popc(active_mask);
 
             if (active_count > 0) {
                 uint64_t base_seq = 0;
                 if (lane_id == warp_find_first_set(active_mask)) {
                     base_seq = atomicAdd((unsigned long long*)global_write_seq, active_count);
                     atomicAdd((unsigned int*)&stream_progress[stream_id], active_count);
                 }
                 base_seq = __shfl_sync(active_mask, base_seq, warp_find_first_set(active_mask));
 
                 if (has_message) {
                     uint32_t thread_offset = __popc(active_mask & ((1U << lane_id) - 1));
                     uint64_t my_seq = base_seq + thread_offset;
                     uint32_t slot_idx = my_seq & slot_mask;
 
                     // Process message
                     OptimizedSlot* slot = &output_slots[slot_idx];
                     const uint8_t* msg_data = &stream_buffer[desc.offset];
 
                     // Copy payload
                     uint32_t copy_len = min(desc.length, 192u);
                     for (uint32_t i = 0; i < copy_len; i++) {
                         slot->payload[i] = msg_data[i];
                     }
 
                     // Set metadata
                     slot->length = desc.length;
                     slot->agent_type = desc.agent_type;
                     slot->flags = desc.flags | (stream_id << 16);  // Encode stream ID
                     slot->timestamp = desc.timestamp;
                     slot->batch_id = stream_id;
 
                     // Memory fence and publish
                     __threadfence_system();
                     atomicExch((unsigned long long*)&slot->sequence, my_seq);
                 }
             }
 
             // Re-queue stream if more work remains
             if (lane_id == 0 && batch_end < count) {
                 uint32_t tail = atomicAdd(&queue_tail, 1);
                 work_queue[tail % 256] = stream_id;
             }
         }
 
         warp.sync();
     }
 }
 
 // ============================================================================
 // C Interface Functions
 // ============================================================================
 
 extern "C" {
 
 /**
  * @brief Launch optimized batch processing kernel
  */
 int launch_optimized_batch(
     const uint8_t* input_data,
     const MessageDescriptor* descriptors,
     uint32_t num_messages,
     OptimizedSlot* output_slots,
     uint64_t* global_write_seq,
     uint32_t slot_mask,
     ProcessingMetrics* metrics,
     bool enable_prefetch,
     cudaStream_t stream
 ) {
     // Get device properties
     cudaDeviceProp prop;
     cudaGetDeviceProperties(&prop, 0);
 
     // Calculate optimal launch configuration
     const int warp_size = 32;
     const int warps_per_sm = prop.maxThreadsPerMultiProcessor / warp_size;
     const int warps_per_block = min(8, warps_per_sm / 2);  // Balance occupancy
     const int threads_per_block = warps_per_block * warp_size;
 
     // Calculate grid size
     const int max_blocks = prop.multiProcessorCount * 2;  // Oversubscribe for hiding latency
     const int messages_per_block = threads_per_block;
     int blocks = min((num_messages + messages_per_block - 1) / messages_per_block, max_blocks);
 
     // Calculate shared memory
     size_t shared_mem = sizeof(WarpBatch) * warps_per_block +  // Batch descriptors
                        256 * threads_per_block;                // Thread-local buffers
 
     // Check shared memory limit
     if (shared_mem > prop.sharedMemPerBlock) {
         printf("Warning: Reducing shared memory usage\n");
         shared_mem = prop.sharedMemPerBlock;
     }
 
     // Launch kernel
     optimized_batch_kernel<<<blocks, threads_per_block, shared_mem, stream>>>(
         input_data,
         descriptors,
         num_messages,
         output_slots,
         global_write_seq,
         slot_mask,
         metrics,
         enable_prefetch
     );
 
     cudaError_t err = cudaGetLastError();
     if (err != cudaSuccess) {
         printf("Optimized batch kernel launch failed: %s\n", cudaGetErrorString(err));
         return -1;
     }
 
     return 0;
 }
 
 /**
  * @brief Launch multi-stream batch processor
  */
 int launch_multi_stream_batch(
     const uint8_t** stream_buffers,
     const MessageDescriptor** stream_descriptors,
     const uint32_t* stream_counts,
     uint32_t num_streams,
     OptimizedSlot* output_slots,
     uint64_t* global_write_seq,
     uint32_t slot_mask,
     uint32_t* stream_progress,
     cudaStream_t stream
 ) {
     // Launch configuration
     const int threads_per_block = 256;
     const int blocks = min((int)num_streams, 32);  // One block per stream up to limit
 
     // Launch kernel
     multi_stream_batch_kernel<<<blocks, threads_per_block, 0, stream>>>(
         stream_buffers,
         stream_descriptors,
         stream_counts,
         num_streams,
         output_slots,
         global_write_seq,
         slot_mask,
         stream_progress
     );
 
     cudaError_t err = cudaGetLastError();
     if (err != cudaSuccess) {
         printf("Multi-stream batch kernel launch failed: %s\n", cudaGetErrorString(err));
         return -1;
     }
 
     return 0;
 }
 
 /**
  * @brief Initialize batch processing system
  */
 int init_batch_processing(
     OptimizedSlot** slots,
     uint32_t slot_count,
     ProcessingMetrics** metrics
 ) {
     cudaError_t err;
 
     // Allocate slots
     size_t slots_size = slot_count * sizeof(OptimizedSlot);
     err = cudaMallocManaged((void**)slots, slots_size, cudaMemAttachGlobal);
     if (err != cudaSuccess) {
         printf("Failed to allocate batch slots: %s\n", cudaGetErrorString(err));
         return -1;
     }
 
     // Initialize slots
     memset(*slots, 0, slots_size);
     for (uint32_t i = 0; i < slot_count; i++) {
         (*slots)[i].sequence = UINT64_MAX;
     }
 
     // Allocate metrics
     err = cudaMallocManaged((void**)metrics, sizeof(ProcessingMetrics), cudaMemAttachGlobal);
     if (err != cudaSuccess) {
         printf("Failed to allocate metrics: %s\n", cudaGetErrorString(err));
         cudaFree(*slots);
         return -1;
     }
 
     memset(*metrics, 0, sizeof(ProcessingMetrics));
 
     printf("Batch processing system initialized:\n");
     printf("  Slots: %u (%.2f MB)\n", slot_count, slots_size / (1024.0 * 1024.0));
     printf("  Warp-level optimizations: Enabled\n");
     printf("  Memory coalescing: Optimized\n");
 
     return 0;
 }
 
 /**
  * @brief Get batch processing metrics
  */
 int get_batch_metrics(
     ProcessingMetrics* metrics,
     uint64_t* messages,
     uint64_t* bytes,
     float* bandwidth_gbps,
     float* messages_per_sec
 ) {
     if (!metrics) return -1;
 
     *messages = metrics->messages_processed;
     *bytes = metrics->bytes_processed;
     *bandwidth_gbps = metrics->bandwidth_gbps;
     *messages_per_sec = metrics->messages_per_second;
 
     return 0;
 }
 
 /**
  * @brief Cleanup batch processing system
  */
 int cleanup_batch_processing(
     OptimizedSlot* slots,
     ProcessingMetrics* metrics
 ) {
     cudaDeviceSynchronize();
 
     if (slots) cudaFree(slots);
     if (metrics) cudaFree(metrics);
 
     return 0;
 }
 
 } // extern "C"
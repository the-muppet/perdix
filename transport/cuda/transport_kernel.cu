/**
 * @file transport_kernel.cu
 * @brief Standard message transport kernel for Perdix
 *
 * This kernel provides general-purpose message streaming from GPU to CPU through
 * the ring buffer. It's designed for flexibility where messages come from
 * different sources and may be scattered in memory.
 *
 * ## Use Cases
 * - Real-time message streaming from multiple GPU threads
 * - Messages from different memory locations
 * - Lower setup overhead for dynamic message generation
 * - When you can't pre-aggregate messages into an arena
 *
 * ## Architecture
 * Each StreamContext contains a direct pointer to its message data:
 * ```
 * StreamContext {
 *     const uint8_t* text;  // Direct pointer to message
 *     uint32_t text_len;
 *     ...
 * }
 * ```
 *
 * ## Example Usage
 * ```cuda
 * // Host code
 * StreamContext contexts[100];
 * for(int i = 0; i < 100; i++) {
 *     contexts[i].text = individual_messages[i];  // Scattered pointers
 *     contexts[i].text_len = lengths[i];
 * }
 * launch_transport_kernel(slots, header, contexts, 100, false, stream);
 * ```
 *
 * ## Performance Characteristics
 * - Good for mixed/dynamic workloads
 * - Higher memory access latency due to scattered reads
 * - More flexible than packed kernel
 * - Suitable for < 1M messages/sec throughput
 *
 * @see packed_kernel.cu for the optimized batch processing variant
 */

#include "common.cuh"
#include <cooperative_groups.h>
#include <stdio.h>

namespace cg = cooperative_groups;

/**
 * @brief Allocate slot indices with warp-level batching
 *
 * Reduces atomic contention by having one thread per warp
 * allocate a batch of slots for the entire warp.
 */
__device__ __forceinline__ uint64_t allocate_slot_batch(
    Header* header,
    uint32_t batch_size,
    bool& success
) {
    auto warp = cg::tiled_partition<32>(cg::this_thread_block());
    uint64_t base_idx = 0;
    success = false;

    // Leader thread does atomic allocation
    if (warp.thread_rank() == 0) {
        uint64_t old_idx = atomicAdd((unsigned long long*)&header->producer.write_idx,
                                     (unsigned long long)batch_size);

        // Check for overflow (backpressure)
        uint64_t read_idx = header->consumer.read_idx;
        if (old_idx - read_idx < header->config.n_slots) {
            base_idx = old_idx;
            success = true;
        } else {
            // Rollback on overflow
            atomicAdd((unsigned long long*)&header->producer.write_idx,
                     (unsigned long long)(-batch_size));
            atomicAdd((unsigned long long*)&header->control.overflow_events, 1);
        }
    }

    // Broadcast result to warp
    base_idx = warp.shfl(base_idx, 0);
    success = warp.shfl(success, 0);

    return base_idx;
}

/**
 * @brief Main kernel for streaming messages through the ring buffer
 *
 * Processes multiple messages in parallel with warp-level coordination
 * to minimize atomic contention and maximize throughput.
 */
__global__ void stream_messages_kernel(
    Slot* slots,
    Header* header,
    const StreamContext* contexts,
    uint32_t n_messages,
    KernelMetrics* metrics
) {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t warp_id = tid / WARP_SIZE;
    const uint32_t lane_id = tid % WARP_SIZE;

    // Start metrics
    if (metrics && tid == 0) {
        metrics->start();
    }

    // Each warp processes messages together
    auto warp = cg::tiled_partition<32>(cg::this_thread_block());

    // Process messages in warp-sized batches
    for (uint32_t msg_base = warp_id * WARP_SIZE;
         msg_base < n_messages;
         msg_base += gridDim.x * blockDim.x / WARP_SIZE * WARP_SIZE) {

        uint32_t msg_idx = msg_base + lane_id;
        bool has_message = msg_idx < n_messages;

        // Count active messages in warp
        uint32_t active_mask = warp.ballot(has_message);
        uint32_t batch_size = __popc(active_mask);

        if (batch_size == 0) continue;

        // Allocate slots for entire warp
        bool allocation_success;
        uint64_t base_slot_idx = allocate_slot_batch(header, batch_size, allocation_success);

        if (!allocation_success) {
            if (metrics) {
                atomicAdd(&metrics->backpressure_events, 1);
            }
            continue;
        }

        // Calculate this thread's slot index
        uint32_t thread_slot_offset = __popc(active_mask & ((1u << lane_id) - 1));
        uint64_t my_slot_idx = base_slot_idx + thread_slot_offset;

        // Process message if we have one
        if (has_message) {
            const StreamContext& ctx = contexts[msg_idx];
            Slot* my_slot = &slots[my_slot_idx & header->config.slot_mask];

            // Set sequence number
            my_slot->seq = my_slot_idx;

            // Copy message data
            uint32_t copy_len = min(ctx.text_len, PAYLOAD_SIZE);
            optimized_memcpy(my_slot->payload, ctx.text, copy_len);

            // Set metadata
            my_slot->len = copy_len;
            my_slot->flags = (ctx.flags & 0xFFFFFF) |
                            (static_cast<uint32_t>(ctx.message_type) << 24);

            // Memory fence to ensure CPU visibility
            __threadfence_system();

            if (metrics) {
                atomicAdd(&metrics->messages_processed, 1);
            }
        }
    }

    // End metrics
    if (metrics && tid == 0) {
        metrics->end();
    }
}

// ============================================================================
// Host Interface
// ============================================================================

extern "C" {

/**
 * @brief Launch the main streaming kernel
 */
int launch_transport_kernel(
    Slot* slots,
    Header* header,
    const StreamContext* contexts,
    uint32_t n_messages,
    int enable_metrics,
    cudaStream_t stream
) {
    if (!slots || !header || !contexts || n_messages == 0) {
        return -5;
    }

    // Allocate metrics if requested
    KernelMetrics* d_metrics = nullptr;
    if (enable_metrics) {
        cudaMallocManaged(&d_metrics, sizeof(KernelMetrics));
        memset(d_metrics, 0, sizeof(KernelMetrics));
    }

    // Copy contexts to device
    StreamContext* d_contexts;
    size_t contexts_size = sizeof(StreamContext) * n_messages;
    cudaMallocManaged(&d_contexts, contexts_size);
    memcpy(d_contexts, contexts, contexts_size);

    // Calculate grid dimensions
    int block_size = 256;
    int grid_size = (n_messages + block_size - 1) / block_size;
    grid_size = min(grid_size, 65535);

    // Launch kernel
    stream_messages_kernel<<<grid_size, block_size, 0, stream>>>(
        slots, header, d_contexts, n_messages, d_metrics
    );

    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(d_contexts);
        if (d_metrics) cudaFree(d_metrics);
        return -4;
    }

    // Sync if no stream provided
    if (stream == nullptr) {
        cudaDeviceSynchronize();
    }

    // Print metrics if collected
    if (d_metrics) {
        cudaDeviceSynchronize();
        uint64_t cycles = d_metrics->cycles_end - d_metrics->cycles_start;
        printf("[Metrics] Messages: %u, Cycles: %llu, Backpressure: %u\n",
               d_metrics->messages_processed, cycles, d_metrics->backpressure_events);
        cudaFree(d_metrics);
    }

    cudaFree(d_contexts);
    return 0;
}

} // extern "C"
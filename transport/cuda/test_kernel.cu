/**
 * @file test_kernel.cu
 * @brief Simple test kernel for verifying ring buffer functionality
 *
 * This kernel generates test messages directly on the GPU for testing
 * the ring buffer without needing external data. It's useful for:
 * - Verifying buffer wraparound
 * - Testing backpressure handling
 * - Performance baseline measurements
 * - Debugging sequence number issues
 *
 * ## Comparison with Production Kernels
 *
 * ### Test Kernel (this file)
 * - Generates messages on GPU
 * - No external data needed
 * - For testing/debugging only
 *
 * ### Standard Transport (transport_kernel.cu)
 * - Processes real messages from host
 * - Flexible pointer-based approach
 * - Production use
 *
 * ### Packed Transport (packed_kernel.cu)
 * - Processes pre-aggregated messages
 * - Optimized offset-based approach
 * - High-performance production use
 *
 * ## Example Usage
 * ```cuda
 * // Generate 1000 test messages
 * launch_simple_test(slots, header, 1000);
 * // Messages will be: "GPU Test Message 0", "GPU Test Message 1", ...
 * ```
 */

#include "common.cuh"
#include <stdio.h>

/**
 * @brief Simple test kernel that generates messages directly on GPU
 *
 * Used for testing and debugging the ring buffer without external data.
 * Each thread generates a numbered test message.
 */
__global__ void simple_test_kernel(
    Slot* slots,
    Header* header,
    uint32_t n_messages
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= n_messages) return;

    // Allocate a slot
    uint64_t slot_idx = atomicAdd((unsigned long long*)&header->producer.write_idx, 1ULL);

    // Check for overflow
    uint64_t read_idx = header->consumer.read_idx;
    if (slot_idx - read_idx >= header->config.n_slots) {
        atomicAdd((unsigned long long*)&header->producer.write_idx, (unsigned long long)(-1));
        return;
    }

    // Get the actual slot
    Slot* my_slot = &slots[slot_idx & header->config.slot_mask];

    // Generate test message
    const char* test_msg = "GPU Test Message ";
    uint32_t msg_len = 17;

    // Copy message
    for (uint32_t i = 0; i < msg_len; i++) {
        my_slot->payload[i] = test_msg[i];
    }

    // Add message number
    uint32_t num = tid;
    uint32_t digits = 0;
    uint32_t temp = num;
    do {
        digits++;
        temp /= 10;
    } while (temp > 0);

    for (uint32_t i = 0; i < digits; i++) {
        my_slot->payload[msg_len + digits - 1 - i] = '0' + (num % 10);
        num /= 10;
    }

    my_slot->len = msg_len + digits;
    my_slot->seq = slot_idx;
    my_slot->flags = 0;

    // Ensure CPU visibility
    __threadfence_system();
}

// ============================================================================
// Host Interface
// ============================================================================

extern "C" {

/**
 * @brief Launch simple test kernel
 */
int launch_simple_test(Slot* slots, Header* header, int n_msgs) {
    if (!slots || !header || n_msgs <= 0) {
        return -5;
    }

    int block_size = 256;
    int grid_size = (n_msgs + block_size - 1) / block_size;

    simple_test_kernel<<<grid_size, block_size>>>(slots, header, n_msgs);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("[ERROR] Kernel launch failed: %s\n", cudaGetErrorString(err));
        return -4;
    }

    cudaDeviceSynchronize();

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("[ERROR] Kernel execution failed: %s\n", cudaGetErrorString(err));
        return -4;
    }

    return 0;
}

} // extern "C"
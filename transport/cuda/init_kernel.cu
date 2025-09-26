/**
 * @file init_kernel.cu
 * @brief CUDA initialization and buffer management functions
 *
 * This file provides the initialization infrastructure for the Perdix
 * transport layer. After initialization, you can choose between two
 * kernel variants based on your performance needs:
 *
 * ## Kernel Selection Guide
 *
 * ### Use Standard Transport (launch_transport_kernel) when:
 * - Messages come from multiple sources
 * - Message generation is dynamic/real-time
 * - Flexibility is more important than raw speed
 * - Processing < 1M messages/second
 * - Example: Live log streaming, event processing
 *
 * ### Use Packed Transport (launch_packed_kernel) when:
 * - Can batch messages into contiguous memory
 * - Need maximum throughput (>10M messages/sec)
 * - Processing large batches of messages
 * - Can afford setup overhead for arena packing
 * - Example: Bulk data export, batch analytics
 *
 * ## Initialization Flow
 * ```c
 * // 1. Initialize CUDA device
 * cuda_init_device(0);
 *
 * // 2. Allocate unified buffer
 * Slot* slots; Header* header;
 * init_unified_buffer(&slots, &header, 4096);
 *
 * // 3. Choose your kernel:
 * // Option A: Flexible standard transport
 * StreamContext contexts[100];
 * launch_transport_kernel(slots, header, contexts, 100, false, stream);
 *
 * // Option B: High-performance packed transport
 * PackedStreamContext packed[1000];
 * uint8_t* arena = ...;  // Pre-packed messages
 * launch_packed_kernel(slots, header, packed, arena, 1000, false, stream);
 *
 * // 4. Cleanup when done
 * cleanup_unified_buffer(slots, header);
 * ```
 */

#include "common.cuh"
#include <cuda_runtime.h>
#include <stdio.h>
#include <string.h>

extern "C" {

/**
 * @brief Initialize CUDA device
 *
 * Sets up the specified CUDA device for unified memory operations.
 * Prints device capabilities for debugging.
 */
int cuda_init_device(int device_id) {
    cudaError_t err = cudaSetDevice(device_id);
    if (err != cudaSuccess) {
        printf("[ERROR] Failed to set device %d: %s\n",
               device_id, cudaGetErrorString(err));
        return -1;
    }

    // Set for mapped memory
    err = cudaSetDeviceFlags(cudaDeviceMapHost);
    if (err != cudaSuccess) {
        printf("[ERROR] Failed to set device flags: %s\n",
               cudaGetErrorString(err));
        return -1;
    }

    // Query device properties
    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, device_id);
    if (err != cudaSuccess) {
        printf("[ERROR] Failed to get device properties: %s\n",
               cudaGetErrorString(err));
        return -1;
    }

    printf("[CUDA] Device %d: %s\n", device_id, prop.name);
    printf("[CUDA] Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("[CUDA] Unified Memory: %s\n", prop.unifiedAddressing ? "Yes" : "No");
    printf("[CUDA] Total Memory: %.2f GB\n",
           prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    printf("[CUDA] Multiprocessors: %d\n", prop.multiProcessorCount);
    printf("[CUDA] Warp Size: %d\n", prop.warpSize);

    if (!prop.unifiedAddressing) {
        printf("[WARNING] Device does not support unified addressing\n");
    }

    return 0;
}

/**
 * @brief Allocate and initialize unified memory buffer
 *
 * Creates the ring buffer in CUDA unified memory, accessible from
 * both CPU and GPU without explicit transfers.
 */
int init_unified_buffer(Slot** slots, Header** header, int n_slots) {
    // Validate power of 2
    if (n_slots <= 0 || (n_slots & (n_slots - 1)) != 0) {
        printf("[ERROR] n_slots must be a positive power of 2, got %d\n", n_slots);
        return -3;
    }

    // Allocate header
    cudaError_t err = cudaMallocManaged((void**)header, sizeof(Header));
    if (err != cudaSuccess) {
        printf("[ERROR] Failed to allocate header: %s\n", cudaGetErrorString(err));
        return -1;
    }

    // Allocate slots
    size_t slots_size = sizeof(Slot) * n_slots;
    err = cudaMallocManaged((void**)slots, slots_size);
    if (err != cudaSuccess) {
        printf("[ERROR] Failed to allocate %zu bytes for slots: %s\n",
               slots_size, cudaGetErrorString(err));
        cudaFree(*header);
        *header = nullptr;
        return -1;
    }

    // Initialize header
    memset(*header, 0, sizeof(Header));
    (*header)->config.n_slots = n_slots;
    (*header)->config.slot_mask = n_slots - 1;
    (*header)->producer.write_idx = 0;
    (*header)->producer.messages_produced = 0;
    (*header)->consumer.read_idx = 0;
    (*header)->consumer.messages_consumed = 0;
    (*header)->control.overflow_events = 0;
    (*header)->control.underrun_events = 0;

    // Initialize slots with UINT64_MAX sequence to indicate empty
    for (int i = 0; i < n_slots; i++) {
        (*slots)[i].seq = UINT64_MAX;
        (*slots)[i].len = 0;
        (*slots)[i].flags = 0;
        memset((*slots)[i].payload, 0, sizeof((*slots)[i].payload));
    }

    // Prefetch to GPU for better initial performance
    int device;
    err = cudaGetDevice(&device);
    if (err == cudaSuccess) {
        cudaMemPrefetchAsync(*header, sizeof(Header), device);
        cudaMemPrefetchAsync(*slots, slots_size, device);
    }

    printf("[CUDA] Allocated ring buffer: %d slots, %.2f MB total\n",
           n_slots, (sizeof(Header) + slots_size) / (1024.0 * 1024.0));

    return 0;
}

/**
 * @brief Free unified memory buffer
 *
 * Cleans up all allocated CUDA unified memory.
 */
int cleanup_unified_buffer(Slot* slots, Header* header) {
    cudaError_t err1 = cudaSuccess, err2 = cudaSuccess;

    if (slots) {
        err1 = cudaFree(slots);
        if (err1 != cudaSuccess) {
            printf("[ERROR] Failed to free slots: %s\n", cudaGetErrorString(err1));
        }
    }

    if (header) {
        err2 = cudaFree(header);
        if (err2 != cudaSuccess) {
            printf("[ERROR] Failed to free header: %s\n", cudaGetErrorString(err2));
        }
    }

    return (err1 == cudaSuccess && err2 == cudaSuccess) ? 0 : -1;
}

/**
 * @brief Reset CUDA device
 *
 * Destroys all allocations and resets the device state.
 * Useful for cleanup in error scenarios.
 */
int cuda_reset_device() {
    cudaError_t err = cudaDeviceReset();
    if (err != cudaSuccess) {
        printf("[ERROR] Failed to reset device: %s\n", cudaGetErrorString(err));
        return -1;
    }
    return 0;
}

} // extern "C"
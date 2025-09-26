/**
 * @file common.cuh
 * @brief Common data structures and utilities for Perdix CUDA kernels
 *
 * This header defines the core data structures shared between all Perdix kernels.
 * The transport layer provides two kernel variants that use these structures:
 *
 * ## Kernel Variants
 *
 * ### 1. Standard Transport Kernel (transport_kernel.cu)
 * - Uses `StreamContext` with direct pointers
 * - Flexible, handles scattered messages
 * - Good for dynamic/real-time workloads
 * - Example: Processing individual log entries as they're generated
 *
 * ### 2. Packed Kernel (packed_kernel.cu)
 * - Uses `PackedStreamContext` with offsets into arena
 * - Optimized for batch processing
 * - 2-3x faster for high-throughput scenarios
 * - Example: Bulk processing 1M pre-aggregated messages
 *
 * ## Choosing Between Kernels
 * ```
 * Standard Transport:           Packed Transport:
 * [ptr]->"msg1"                [arena][msg1|msg2|msg3...]
 * [ptr]->"msg2"                [offset=0][offset=5][offset=10]
 * [ptr]->"msg3"
 *
 * Scattered memory             Contiguous memory
 * Flexible                     Fast
 * ```
 */

#ifndef PERDIX_COMMON_CUH
#define PERDIX_COMMON_CUH

#include <cuda_runtime.h>
#include <stdint.h>

// ============================================================================
// Data Structures
// ============================================================================

/**
 * @struct Slot
 * @brief Cache-line aligned message slot in the ring buffer
 *
 * Each slot is 256 bytes to optimize for GPU memory access patterns.
 * The structure MUST match the Rust FFI layout exactly.
 */
struct __align__(64) Slot {
    uint64_t seq;           // 8 bytes - sequence number
    uint32_t len;           // 4 bytes - payload length
    uint32_t flags;         // 4 bytes - flags
    uint32_t _pad1;         // 4 bytes - padding
    uint8_t payload[240];   // 240 bytes - payload data
    uint8_t _pad2[8];       // 8 bytes - padding
    // Total: 256 bytes aligned
};

/**
 * @struct Header
 * @brief Ring buffer control structure with cache-optimized layout
 *
 * Divided into cache lines to prevent false sharing.
 * Total size: 256 bytes (4 cache lines)
 */
struct __align__(64) Header {
    // Producer cache line (hot for GPU)
    struct {
        uint64_t write_idx;
        uint64_t messages_produced;
        uint8_t _pad[48];
    } producer;

    // Consumer cache line (hot for CPU)
    struct {
        uint64_t read_idx;
        uint64_t messages_consumed;
        uint8_t _pad[48];
    } consumer;

    // Configuration (read-only after init)
    struct {
        uint64_t n_slots;
        uint64_t slot_mask;
        uint8_t _pad[48];
    } config;

    // Control and metrics
    struct {
        uint64_t overflow_events;
        uint64_t underrun_events;
        uint8_t _pad[48];
    } control;
};

/**
 * @struct StreamContext
 * @brief Message context for standard transport kernel
 *
 * Used by transport_kernel.cu for flexible message processing.
 * Each context contains a direct pointer to its message data,
 * allowing messages to be scattered across memory.
 *
 * Choose this when:
 * - Messages come from different sources
 * - Can't pre-aggregate into arena
 * - Need flexibility over performance
 */
struct StreamContext {
    const uint8_t* text;
    uint32_t text_len;
    uint8_t message_type;
    uint32_t stream_id;
    uint64_t timestamp;
    uint32_t flags;
    uint8_t _pad[3];
};

/**
 * @struct KernelMetrics
 * @brief Performance metrics for kernel profiling
 */
struct KernelMetrics {
    uint64_t cycles_start;
    uint64_t cycles_end;
    uint32_t messages_processed;
    uint32_t atomic_conflicts;
    uint32_t memory_stalls;
    uint32_t backpressure_events;

    __device__ void start() {
        cycles_start = clock64();
    }

    __device__ void end() {
        cycles_end = clock64();
    }
};

// ============================================================================
// Constants
// ============================================================================

__constant__ uint32_t PAYLOAD_SIZE = 240;
__constant__ uint32_t WARP_SIZE = 32;

// ============================================================================
// Device Utilities
// ============================================================================

/**
 * @brief Optimized memory copy using vectorized operations when possible
 */
__device__ __forceinline__ void optimized_memcpy(
    uint8_t* dst,
    const uint8_t* src,
    uint32_t size
) {
    // Use uint4 (16-byte) transfers when aligned
    if (((uintptr_t)dst & 15) == 0 && ((uintptr_t)src & 15) == 0 && (size & 15) == 0) {
        uint4* dst4 = (uint4*)dst;
        const uint4* src4 = (const uint4*)src;
        for (uint32_t i = 0; i < size/16; i++) {
            dst4[i] = src4[i];
        }
    } else {
        // Byte-by-byte fallback
        for (uint32_t i = 0; i < size; i++) {
            dst[i] = src[i];
        }
    }
}

#endif // PERDIX_COMMON_CUH

#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

namespace cg = cooperative_groups;

// ============================================================================
// Data Structures
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

// Cache-line aligned slot structure (256 bytes total)
struct __align__(64) Slot {
    // First cache line - hot data
    uint64_t seq;           // 8 bytes - uses atomic operations
    uint32_t len;          // 4 bytes
    uint32_t flags;        // 4 bytes
    uint64_t timestamp;    // 8 bytes
    uint8_t _pad1[40];     // Pad to 64 bytes
    
    // Payload in separate cache lines (192 bytes for ANSI-formatted text)
    uint8_t payload[192];
};

// Header structure with separated hot/cold data (256 bytes)
struct __align__(64) Header {
    // Producer cache line (hot for GPU)
    struct {
        uint64_t write_idx;     // Atomic operations via atomicAdd
        uint64_t messages_produced;
        uint8_t _pad[48];
    } producer;
    
    // Consumer cache line (hot for CPU)
    struct {
        uint64_t read_idx;      // For consumer tracking
        uint64_t messages_consumed;
        uint8_t _pad[48];
    } consumer;
    
    // Configuration (read-only after init)
    struct {
        uint64_t wrap_mask;
        uint32_t slot_count;
        uint32_t payload_size;
        uint32_t batch_size;
        uint32_t _reserved;
        uint8_t _pad[40];
    } config;
    
    // Control flags (infrequent access)
    struct {
        uint32_t backpressure;  // Flow control
        uint32_t stop;          // Termination flag
        uint32_t epoch;
        uint32_t error_count;
        uint8_t _pad[48];
    } control;
};

// Stream context for processing AI agent responses
struct StreamContext {
    const uint8_t* text;        // Raw text from AI agent
    uint32_t text_len;          // Length of text
    AgentType agent_type;       // Type of agent
    uint32_t stream_id;         // Stream identifier
    uint64_t timestamp;         // Message timestamp
    bool is_continuation;       // Part of multi-part message
    bool enable_ansi;          // Enable ANSI formatting
    uint8_t _pad[2];
};

// Performance metrics for profiling
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
    
    __host__ __device__ uint64_t elapsed_cycles() const {
        return cycles_end - cycles_start;
    }
};

// ============================================================================
// Device Helper Functions
// ============================================================================

// Get ANSI color code for agent type
__device__ __forceinline__ void get_agent_color(
    AgentType agent_type,
    uint8_t* color_code,
    uint32_t& code_len
) {
    const char* codes[] = {
        "\033[34m",  // SYSTEM - Blue
        "\033[32m",  // USER - Green
        "\033[36m",  // ASSISTANT - Cyan
        "\033[31m",  // ERROR - Red
        "\033[33m",  // WARNING - Yellow
        "\033[37m",  // INFO - White
        "\033[35m",  // DEBUG - Magenta
        "\033[90m"   // TRACE - Bright Black
    };
    
    const char* code = codes[static_cast<uint8_t>(agent_type)];
    code_len = 5;
    for (int i = 0; i < 5; i++) {
        color_code[i] = code[i];
    }
}

// Add ANSI reset sequence
__device__ __forceinline__ void add_reset_sequence(
    uint8_t* buffer,
    uint32_t& offset
) {
    const char reset[] = "\033[0m";
    for (int i = 0; i < 4; i++) {
        buffer[offset++] = reset[i];
    }
}

// Detect keywords for highlighting
__device__ bool detect_keyword(
    const uint8_t* text,
    uint32_t pos,
    uint32_t text_len,
    const char* keyword,
    uint32_t keyword_len
) {
    if (pos + keyword_len > text_len) return false;
    
    for (uint32_t i = 0; i < keyword_len; i++) {
        if (text[pos + i] != keyword[i]) return false;
    }
    
    // Check word boundaries
    if (pos > 0) {
        uint8_t prev = text[pos - 1];
        if ((prev >= 'a' && prev <= 'z') || 
            (prev >= 'A' && prev <= 'Z') ||
            (prev >= '0' && prev <= '9')) {
            return false;
        }
    }
    
    if (pos + keyword_len < text_len) {
        uint8_t next = text[pos + keyword_len];
        if ((next >= 'a' && next <= 'z') || 
            (next >= 'A' && next <= 'Z') ||
            (next >= '0' && next <= '9')) {
            return false;
        }
    }
    
    return true;
}

// Vectorized memory copy
template<typename T>
__device__ __forceinline__ void vectorized_copy(
    void* dst,
    const void* src,
    size_t size
) {
    const size_t vector_size = sizeof(T);
    const size_t num_vectors = size / vector_size;
    
    T* d = reinterpret_cast<T*>(dst);
    const T* s = reinterpret_cast<const T*>(src);
    
    #pragma unroll 4
    for (size_t i = 0; i < num_vectors; i++) {
        d[i] = s[i];
    }
    
    // Handle remaining bytes
    if (size % vector_size != 0) {
        uint8_t* d_bytes = reinterpret_cast<uint8_t*>(dst) + num_vectors * vector_size;
        const uint8_t* s_bytes = reinterpret_cast<const uint8_t*>(src) + num_vectors * vector_size;
        for (size_t i = 0; i < size % vector_size; i++) {
            d_bytes[i] = s_bytes[i];
        }
    }
}

// ============================================================================
// Kernel 
// ============================================================================

// Non-templated kernel to avoid nvcc compilation issues
__global__ void
unified_stream_kernel(
    Slot* __restrict__ slots,
    Header* __restrict__ hdr,
    const StreamContext* __restrict__ contexts,
    const uint32_t n_messages,
    KernelMetrics* __restrict__ metrics,
    int enable_metrics
) {
    const int BLOCK_SIZE = 224;  // Reduced from 256 to fit in 48KB shared memory
    const int PAYLOAD_SIZE = 192;
    // Thread identifiers
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int global_tid = bid * blockDim.x + tid;
    
    // Shared memory for warp-level batching
    extern __shared__ uint8_t shared_mem[];
    uint8_t* shared_buffer = shared_mem;
    uint64_t* shared_seqs = reinterpret_cast<uint64_t*>(
        &shared_mem[BLOCK_SIZE * PAYLOAD_SIZE]
    );
    
    // Start metrics collection if enabled
    KernelMetrics local_metrics = {};
    if (enable_metrics && tid == 0) {
        local_metrics.start();
    }
    
    // Cooperative groups
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    
    // Configuration
    const uint64_t wrap_mask = hdr->config.wrap_mask;
    const uint32_t backpressure_threshold = hdr->config.slot_count - 64;
    
    // Grid-stride loop for processing messages
    for (uint32_t msg_base = bid * BLOCK_SIZE; 
         msg_base < n_messages; 
         msg_base += gridDim.x * BLOCK_SIZE) {
        
        uint32_t msg_id = msg_base + tid;
        bool has_work = msg_id < n_messages;
        
        // Check for stop signal
        if (atomicAdd((unsigned int*)&hdr->control.stop, 0) != 0) {
            break;
        }
        
        // Step 1: Warp-level cooperative sequence reservation
        uint64_t warp_seq_base = 0;
        uint32_t active_mask = __ballot_sync(0xFFFFFFFF, has_work);
        
        if (active_mask != 0) {
            if (lane_id == __ffs(active_mask) - 1) {
                // Leader thread reserves sequences for entire warp
                uint32_t warp_count = __popc(active_mask);
                
                // Check backpressure before reservation
                uint64_t current_write = atomicAdd((unsigned long long*)&hdr->producer.write_idx, 0ULL);
                uint64_t current_read = atomicAdd((unsigned long long*)&hdr->consumer.read_idx, 0ULL);
                
                if ((current_write - current_read) < backpressure_threshold) {
                    warp_seq_base = atomicAdd(
                        (unsigned long long*)&hdr->producer.write_idx,
                        (unsigned long long)warp_count
                    );
                    
                    if (enable_metrics) {
                        atomicAdd(&hdr->producer.messages_produced, warp_count);
                    }
                } else {
                    // Signal backpressure
                    atomicExch((unsigned int*)&hdr->control.backpressure, 1);
                    __threadfence_system();
                    has_work = false;
                    
                    if (enable_metrics) {
                        local_metrics.backpressure_events++;
                    }
                }
            }
            
            // Broadcast base sequence to all threads in warp
            warp_seq_base = __shfl_sync(active_mask, warp_seq_base, __ffs(active_mask) - 1);
            has_work = has_work && (warp_seq_base != 0);
        }
        
        if (!has_work) {
            // Backoff to reduce contention
            __nanosleep(100 + (tid * 10));
            continue;
        }
        
        // Step 2: Calculate per-thread sequence number
        uint32_t thread_offset = __popc(active_mask & ((1U << lane_id) - 1));
        uint64_t my_seq = warp_seq_base + thread_offset;
        shared_seqs[tid] = my_seq;
        
        // Step 3: Load stream context
        const StreamContext& ctx = contexts[msg_id];
        
        // Step 4: Build ANSI-formatted message in shared memory
        uint32_t shared_offset = tid * PAYLOAD_SIZE;
        uint32_t output_len = 0;
        // Bounds check for shared memory access
        if (tid >= BLOCK_SIZE) continue;  // Skip if thread id exceeds block size

        
        if (ctx.enable_ansi) {
            // Add agent type color
            uint8_t color_code[8];
            uint32_t code_len;
            get_agent_color(ctx.agent_type, color_code, code_len);
            
            for (uint32_t i = 0; i < code_len && output_len < PAYLOAD_SIZE; i++) {
                shared_buffer[shared_offset + output_len++] = color_code[i];
            }
            
            // Add agent label
            const char* labels[] = {
                "[SYSTEM] ", "[USER] ", "[ASSISTANT] ", "[ERROR] ",
                "[WARNING] ", "[INFO] ", "[DEBUG] ", "[TRACE] "
            };
            const char* label = labels[static_cast<uint8_t>(ctx.agent_type)];
            for (int i = 0; label[i] != '\0' && output_len < PAYLOAD_SIZE - 4; i++) {
                shared_buffer[shared_offset + output_len++] = label[i];
            }
            
            // Reset color for content
            add_reset_sequence(&shared_buffer[shared_offset], output_len);
        }
        
        // Step 5: Process text with keyword highlighting
        uint32_t text_to_copy = min(ctx.text_len, 
                                    static_cast<uint32_t>(PAYLOAD_SIZE - output_len - 4));
        
        for (uint32_t i = 0; i < text_to_copy; i++) {
            uint32_t pos = i;
            bool highlighted = false;
            
            if (ctx.enable_ansi && output_len < PAYLOAD_SIZE - 10) {  // Leave space for color codes
                // Check for ERROR keyword
                if (detect_keyword(ctx.text, pos, ctx.text_len, "ERROR", 5)) {
                    const char* red = "\033[91m";
                    for (int j = 0; j < 5; j++) {
                        shared_buffer[shared_offset + output_len++] = red[j];
                    }
                    for (int j = 0; j < 5; j++) {
                        shared_buffer[shared_offset + output_len++] = ctx.text[pos + j];
                    }
                    add_reset_sequence(&shared_buffer[shared_offset], output_len);
                    i += 4;
                    highlighted = true;
                }
                // Check for WARNING keyword
                else if (detect_keyword(ctx.text, pos, ctx.text_len, "WARNING", 7)) {
                    const char* yellow = "\033[93m";
                    for (int j = 0; j < 5; j++) {
                        shared_buffer[shared_offset + output_len++] = yellow[j];
                    }
                    for (int j = 0; j < 7; j++) {
                        shared_buffer[shared_offset + output_len++] = ctx.text[pos + j];
                    }
                    add_reset_sequence(&shared_buffer[shared_offset], output_len);
                    i += 6;
                    highlighted = true;
                }
                // Check for SUCCESS keyword
                else if (detect_keyword(ctx.text, pos, ctx.text_len, "SUCCESS", 7)) {
                    const char* green = "\033[92m";
                    for (int j = 0; j < 5; j++) {
                        shared_buffer[shared_offset + output_len++] = green[j];
                    }
                    for (int j = 0; j < 7; j++) {
                        shared_buffer[shared_offset + output_len++] = ctx.text[pos + j];
                    }
                    add_reset_sequence(&shared_buffer[shared_offset], output_len);
                    i += 6;
                    highlighted = true;
                }
            }
            
            if (!highlighted && output_len < PAYLOAD_SIZE) {
                shared_buffer[shared_offset + output_len++] = ctx.text[pos];
            }
            
            if (output_len >= PAYLOAD_SIZE - 4) break;
        }
        
        // Add final reset if ANSI enabled
        if (ctx.enable_ansi && !ctx.is_continuation) {
            add_reset_sequence(&shared_buffer[shared_offset], output_len);
        }
        
        // Step 6: Synchronize shared memory
        __syncwarp(active_mask);
        
        // Step 7: Compute slot index
        uint64_t slot_idx = my_seq & wrap_mask;
        Slot* my_slot = &slots[slot_idx];
        
        // Step 8: Prefetch slot for write (sm_80+)
        #if __CUDA_ARCH__ >= 800
        // Touch memory to bring into cache
        volatile uint32_t dummy = my_slot->len;
        (void)dummy;
        #endif
        
        // Step 9: Write payload to global memory (vectorized)
        if (PAYLOAD_SIZE == 192) {
            // Optimized for 192-byte payloads
            float4* dst = reinterpret_cast<float4*>(my_slot->payload);
            float4* src = reinterpret_cast<float4*>(&shared_buffer[shared_offset]);
            
            #pragma unroll
            for (int i = 0; i < 12; i++) {
                dst[i] = src[i];
            }
        } else {
            // Generic vectorized copy
            vectorized_copy<uint4>(my_slot->payload,
                                  &shared_buffer[shared_offset],
                                  min(output_len, static_cast<uint32_t>(PAYLOAD_SIZE)));
        }
        
        // Step 10: Set metadata
        my_slot->len = min(output_len, static_cast<uint32_t>(PAYLOAD_SIZE));
        my_slot->flags = (ctx.is_continuation ? 0x01 : 0x00) | 
                         (static_cast<uint32_t>(ctx.agent_type) << 8) |
                         (ctx.stream_id << 16);
        
        // Get timestamp if available
        #if __CUDA_ARCH__ >= 700
        my_slot->timestamp = clock64();
        #else
        my_slot->timestamp = ctx.timestamp;
        #endif
        
        // Step 11: Memory fence for host visibility
        __threadfence_system();
        
        // Step 12: Atomically publish sequence number (slot is ready)
        atomicExch((unsigned long long*)&my_slot->seq, my_seq);
        __threadfence_system();
        
        if (enable_metrics) {
            local_metrics.messages_processed++;
        }
    }
    
    // Collect final metrics
    if (enable_metrics && tid == 0) {
        local_metrics.end();
        metrics[bid] = local_metrics;
    }
}


// ============================================================================
// C Interface Functions
// ============================================================================

// Device initialization and query
extern "C" int cuda_init_device(int device_id) {
    cudaError_t err;
    
    // Get device count
    int device_count;
    err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess) {
        printf("Failed to get CUDA device count: %s\n", cudaGetErrorString(err));
        return -1;
    }
    
    if (device_count == 0) {
        printf("No CUDA devices found\n");
        return -1;
    }
    
    // Validate device ID
    if (device_id < 0 || device_id >= device_count) {
        printf("Invalid device ID %d (available: 0-%d)\n", device_id, device_count - 1);
        device_id = 0;  // Use default device
    }
    
    // Set device
    err = cudaSetDevice(device_id);
    if (err != cudaSuccess) {
        printf("Failed to set CUDA device %d: %s\n", device_id, cudaGetErrorString(err));
        return -1;
    }
    
    // Get device properties
    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, device_id);
    if (err != cudaSuccess) {
        printf("Failed to get device properties: %s\n", cudaGetErrorString(err));
        return -1;
    }
    
    printf("CUDA Device Initialized:\n");
    printf("  Device ID: %d\n", device_id);
    printf("  Name: %s\n", prop.name);
    printf("  Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("  Total Memory: %.2f GB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    printf("  SMs: %d\n", prop.multiProcessorCount);
    printf("  Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
    printf("  Max Shared Memory per Block: %zu KB\n", prop.sharedMemPerBlock / 1024);
    printf("  Unified Memory: %s\n", prop.unifiedAddressing ? "Yes" : "No");
    
    return device_id;
}

// Initialize buffer with proper memory allocation
extern "C" int init_unified_buffer(
    Slot** slots,
    Header** hdr,
    int n_slots
) {
    cudaError_t err;
    
    // Ensure power of 2
    if ((n_slots & (n_slots - 1)) != 0) {
        printf("Error: n_slots must be power of 2\n");
        return -1;
    }
    
    // Initialize device if not already done
    int device;
    err = cudaGetDevice(&device);
    if (err != cudaSuccess) {
        device = cuda_init_device(0);
        if (device < 0) return -1;
    }
    
    // Allocate header with pinned memory for zero-copy
    err = cudaMallocHost((void**)hdr, sizeof(Header));
    if (err != cudaSuccess) {
        printf("Failed to allocate header: %s\n", cudaGetErrorString(err));
        return -1;
    }
    
    // Initialize header
    memset(*hdr, 0, sizeof(Header));
    Header* h = *hdr;
    h->config.slot_count = n_slots;
    h->config.wrap_mask = n_slots - 1;
    h->config.payload_size = 192;
    h->config.batch_size = 32;
    h->producer.write_idx = 0;
    h->consumer.read_idx = 0;
    h->control.backpressure = 0;
    h->control.stop = 0;
    
    // Allocate slots with pinned memory
    size_t slots_size = n_slots * sizeof(Slot);
    err = cudaMallocHost((void**)slots, slots_size);
    if (err != cudaSuccess) {
        printf("Failed to allocate slots: %s\n", cudaGetErrorString(err));
        cudaFreeHost(*hdr);
        return -1;
    }
    
    // Initialize slots
    memset(*slots, 0, slots_size);
    for (int i = 0; i < n_slots; i++) {
        (*slots)[i].seq = UINT64_MAX;
    }
    
    printf("Unified buffer initialized:\n");
    printf("  Slots: %d (%.2f MB)\n", n_slots, slots_size / (1024.0 * 1024.0));
    printf("  Header at: %p\n", (void*)*hdr);
    printf("  Slots at: %p\n", (void*)*slots);
    
    return 0;
}

// Main kernel launcher
extern "C" int launch_unified_kernel(
    Slot* slots,
    Header* hdr,
    const StreamContext* contexts,
    uint32_t n_messages,
    int enable_metrics,
    cudaStream_t stream
) {
    // Ensure device is initialized
    int device;
    cudaError_t err = cudaGetDevice(&device);
    if (err != cudaSuccess) {
        device = cuda_init_device(0);
        if (device < 0) return -1;
    }
    
    // Copy contexts to device memory if needed
    StreamContext* d_contexts = nullptr;
    if (n_messages > 0 && contexts) {
        size_t contexts_size = n_messages * sizeof(StreamContext);
        err = cudaMalloc(&d_contexts, contexts_size);
        if (err != cudaSuccess) {
            printf("Failed to allocate device memory for contexts: %s\n", cudaGetErrorString(err));
            return -1;
        }
        
        err = cudaMemcpy(d_contexts, contexts, contexts_size, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            printf("Failed to copy contexts to device: %s\n", cudaGetErrorString(err));
            cudaFree(d_contexts);
            return -1;
        }
    }
    
    // Get device properties for optimal configuration
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    
    // Calculate optimal launch configuration
    const int threads_per_block = 128;  // Reduced to ensure shared memory fits
    const int payload_size = 192;
    
    // Dynamic grid sizing based on workload and SM count
    int blocks;
    if (n_messages <= prop.multiProcessorCount * threads_per_block) {
        blocks = (n_messages + threads_per_block - 1) / threads_per_block;
    } else {
        blocks = prop.multiProcessorCount * 2;  // 2 blocks per SM
    }
    blocks = min(blocks, prop.multiProcessorCount * 4);  // Cap at 4 blocks per SM
    
    // Calculate shared memory requirements
    size_t shared_mem_size = threads_per_block * payload_size +     // Buffer
                            threads_per_block * sizeof(uint64_t);   // Sequences
    
    // Check shared memory limits
    if (shared_mem_size > prop.sharedMemPerBlock) {
        printf("Warning: Required shared memory %zu exceeds limit %zu\n", 
               shared_mem_size, prop.sharedMemPerBlock);
        shared_mem_size = 0;  // Fall back to global memory
    }
    
    // Allocate metrics buffer if enabled
    KernelMetrics* d_metrics = nullptr;
    if (enable_metrics) {
        err = cudaMalloc(&d_metrics, blocks * sizeof(KernelMetrics));
        if (err != cudaSuccess) { printf("Failed to allocate metrics: %s\n", cudaGetErrorString(err)); return -1; }
        cudaMemset(d_metrics, 0, blocks * sizeof(KernelMetrics));
    }
    
    printf("Launching unified kernel:\n");
    printf("  Messages: %u\n", n_messages);
    printf("  Blocks: %d, Threads per block: %d\n", blocks, threads_per_block);
    printf("  Shared memory: %zu bytes\n", shared_mem_size);
    printf("  Metrics: %s\n", enable_metrics ? "Enabled" : "Disabled");
    
    // Prefetch data to GPU for better performance
    size_t header_size = sizeof(Header);
    size_t slots_size = hdr->config.slot_count * sizeof(Slot);
    cudaMemPrefetchAsync(hdr, header_size, device, stream);
    cudaMemPrefetchAsync(slots, slots_size, device, stream);
    
    // Validate parameters before launch
    if (!slots || !hdr) {
        printf("Error: Invalid slots or header pointer\n");
        if (d_metrics) cudaFree(d_metrics);
        return -1;
    }
    
    if (n_messages > 0 && !contexts) {
        printf("Error: Invalid contexts pointer for %u messages\n", n_messages);
        if (d_metrics) cudaFree(d_metrics);
        return -1;
    }
    
    // Launch kernel
    dim3 grid(blocks);
    dim3 block(threads_per_block);
    
    // Use default stream if null pointer passed
    cudaStream_t kernel_stream = stream ? stream : 0;
    
    // Use device contexts for kernel launch
    unified_stream_kernel<<<grid, block, shared_mem_size, kernel_stream>>>(
        slots, hdr, d_contexts ? d_contexts : contexts, n_messages, d_metrics, enable_metrics
    );
    
    // Check for launch errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
        printf("  Grid: %d blocks, Block: %d threads\n", blocks, threads_per_block);
        printf("  Shared memory: %zu bytes\n", shared_mem_size);
        printf("  Pointers - slots: %p, hdr: %p, contexts: %p, metrics: %p\n", 
               slots, hdr, d_contexts ? d_contexts : contexts, d_metrics);
        if (d_metrics) cudaFree(d_metrics);
        if (d_contexts) cudaFree(d_contexts);
        return -1;
    }
    
    // Synchronize and check for execution errors
    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
        printf("Kernel execution failed: %s\n", cudaGetErrorString(err));
        if (d_metrics) cudaFree(d_metrics);
        if (d_contexts) cudaFree(d_contexts);
        return -1;
    }
    
    // Prefetch back to CPU for reading
    cudaMemPrefetchAsync(hdr, header_size, cudaCpuDeviceId, stream);
    cudaMemPrefetchAsync(slots, slots_size, cudaCpuDeviceId, stream);
    cudaStreamSynchronize(stream);
    
    // Process metrics if enabled
    if (enable_metrics && d_metrics) {
        KernelMetrics* h_metrics = new KernelMetrics[blocks];
        cudaMemcpy(h_metrics, d_metrics, blocks * sizeof(KernelMetrics), 
                  cudaMemcpyDeviceToHost);
        
        uint64_t total_cycles = 0;
        uint32_t total_messages = 0;
        uint32_t total_backpressure = 0;
        
        for (int i = 0; i < blocks; i++) {
            total_cycles += h_metrics[i].elapsed_cycles();
            total_messages += h_metrics[i].messages_processed;
            total_backpressure += h_metrics[i].backpressure_events;
        }
        
        printf("\nKernel Metrics:\n");
        printf("  Total messages processed: %u\n", total_messages);
        printf("  Average cycles per block: %llu\n", total_cycles / blocks);
        printf("  Backpressure events: %u\n", total_backpressure);
        
        delete[] h_metrics;
        cudaFree(d_metrics);
    }
    
    // Clean up device contexts
    if (d_contexts) {
        cudaFree(d_contexts);
    }
    
    return 0;
}


// Simple test kernel for basic functionality
__global__ void simple_test_kernel(
    Slot* slots,
    Header* hdr,
    int n_msgs
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    
    for (int m = tid; m < n_msgs; m += stride) {
        // Reserve sequence
        uint64_t seq = atomicAdd((unsigned long long*)&hdr->producer.write_idx, 1ULL);
        
        // Get slot
        uint64_t idx = seq & hdr->config.wrap_mask;
        Slot* s = &slots[idx];
        
        // Write test data
        s->len = 16;
        s->flags = 0;
        
        // Simple test message
        const char* msg = "Test Message ";
        for (int i = 0; i < 13; i++) {
            s->payload[i] = msg[i];
        }
        s->payload[13] = '0' + (seq % 10);
        s->payload[14] = '\n';
        s->payload[15] = '\0';
        
        // Memory fence
        __threadfence_system();
        
        // Publish
        s->seq = seq;
        __threadfence_system();
        
        // Debug output for first few messages
        if (seq < 5) {
            printf("GPU: Wrote seq=%llu to slot %llu\n", seq, idx);
        }
    }
}

// Simple test launcher
extern "C" int launch_simple_test(
    Slot* slots,
    Header* hdr,
    int n_msgs
) {
    printf("Launching simple test kernel with %d messages...\n", n_msgs);
    
    simple_test_kernel<<<32, 256>>>(slots, hdr, n_msgs);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Test kernel launch failed: %s\n", cudaGetErrorString(err));
        return -1;
    }
    
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Test kernel execution failed: %s\n", cudaGetErrorString(err));
        return -1;
    }
    
    printf("Test kernel completed successfully\n");
    return 0;
}

// Cleanup function
extern "C" int cleanup_unified_buffer(Slot* slots, Header* hdr) {
    // Ensure all GPU operations are complete before cleanup
    cudaError_t sync_err = cudaDeviceSynchronize();
    if (sync_err != cudaSuccess) {
        printf("Warning: Device sync failed before cleanup: %s\n", cudaGetErrorString(sync_err));
        // Clear the error and continue with cleanup
        cudaGetLastError();
    }

    if (slots) {
        cudaError_t err = cudaFreeHost(slots);
        if (err != cudaSuccess) {
            printf("Warning: Failed to free slots: %s\n", cudaGetErrorString(err));
        }
    }
    
    if (hdr) {
        cudaError_t err = cudaFreeHost(hdr);
        if (err != cudaSuccess) {
            printf("Warning: Failed to free header: %s\n", cudaGetErrorString(err));
        }
    }
    
    return 0;
}
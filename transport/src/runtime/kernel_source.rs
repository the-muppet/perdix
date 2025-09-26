/// Perdix CUDA kernel source code for runtime compilation
/// stored as a solid string to avoid parsing issues 

pub const PERDIX_KERNEL_SOURCE: &str = r#

// Use CUDA built-in types instead of stdint.h for NVRTC compatibility
typedef unsigned long long uint64_t;
typedef unsigned int uint32_t;
typedef unsigned char uint8_t;
typedef long long int64_t;
typedef int int32_t;
typedef signed char int8_t;

// Define constants that are normally in stdint.h
#define UINT64_MAX 0xFFFFFFFFFFFFFFFFULL
#define UINT32_MAX 0xFFFFFFFFU

// Agent types for AI response categorization
enum AgentType {
    SYSTEM = 0,      // Blue
    USER = 1,        // Green
    ASSISTANT = 2,   // Cyan
    ERROR_TYPE = 3,  // Red
    WARNING = 4,     // Yellow
    INFO = 5,        // White
    DEBUG_TYPE = 6,  // Magenta
    TRACE = 7        // Bright Black
};

// Stream context for processing AI agent responses
struct StreamContext {
    const uint8_t* text;        // Raw text from AI agent
    uint32_t text_len;          // Length of text
    uint8_t agent_type;         // Type of agent (AgentType enum)
    uint8_t _pad1[3];           // Padding for alignment
    uint32_t stream_id;         // Stream identifier
    uint64_t timestamp;         // Message timestamp
    uint8_t is_continuation;    // Part of multi-part message
    uint8_t enable_ansi;        // Enable ANSI formatting
    uint8_t _pad2[6];           // Padding
};

// Performance metrics for profiling
struct KernelMetrics {
    uint64_t cycles_start;
    uint64_t cycles_end;
    uint32_t messages_processed;
    uint32_t atomic_conflicts;
    uint32_t memory_stalls;
    uint32_t backpressure_events;
};

// Simplified slot structure for ring buffer (256 bytes, cache-aligned)
struct __align__(64) Slot {
    uint64_t seq;           // Sequence number for ordering
    uint32_t len;           // Payload length
    uint32_t flags;         // Message flags
    uint64_t timestamp;     // Message timestamp
    uint8_t _pad1[40];      // Padding to 64 bytes
    uint8_t payload[192];   // Message payload
};

// Ring buffer header with producer/consumer indices
struct __align__(64) Header {
    // Producer cache line (written by GPU)
    struct {
        uint64_t write_idx;
        uint64_t messages_produced;
        uint8_t _pad[48];
    } producer;
    
    // Consumer cache line (read by CPU)
    struct {
        uint64_t read_idx;
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
    
    // Control flags
    struct {
        uint32_t backpressure;
        uint32_t stop;
        uint32_t epoch;
        uint32_t error_count;
        uint8_t _pad[48];
    } control;
};


extern "C" __global__ void perdix_test_kernel(
    Slot* __restrict__ slots,
    Header* __restrict__ hdr,
    int n_messages
) {
    // Use the thread index within the block for this logic.
    // This kernel assumes n_messages <= blockDim.x.
    int tid = threadIdx.x;

    // Get the starting sequence number deterministically.
    // To avoid a race where every thread hammers atomicAdd on the write_idx,
    // we have a single thread read the starting index into fast __shared__ memory.
    __shared__ uint64_t base_seq;
    if (tid == 0) {
        // Only thread 0 reads the global memory.
        base_seq = hdr->producer.write_idx;
    }
    
    // Synchronize the block.
    // __syncthreads() acts as a barrier, ensuring that all threads in the block
    // wait here until thread 0 has finished writing to `base_seq`. After this
    // line, all threads are guaranteed to see the correct starting sequence.
    __syncthreads();

    // Each thread processes one message if it has work to do.
    if (tid < n_messages) {
        // Calculate this thread's unique sequence number.
        // This is deterministic and contention-free.
        uint64_t seq = base_seq + tid;

        // Calculate the slot index using the power-of-two wrap_mask.
        uint64_t slot_idx = seq & hdr->config.wrap_mask;
        Slot* slot = &slots[slot_idx];

        // It is critical to write all data *before* publishing the sequence number.
        // Simple integer-to-string for the message number (tid).
        // This is safe to do in a kernel as it uses no external library functions.
        const char* prefix = "GPU Test Msg ";
        for (int i = 0; i < 13; i++) {
            slot->payload[i] = prefix[i];
        }
        
        int num = tid;
        int digits = 0;
        uint8_t num_str[10];
        if (num == 0) {
            num_str[0] = '0';
            digits = 1;
        } else {
            while (num > 0 && digits < 10) {
                num_str[digits++] = '0' + (num % 10);
                num /= 10;
            }
            // Reverse the digits since they were extracted backwards
            for (int i = 0; i < digits / 2; i++) {
                uint8_t tmp = num_str[i];
                num_str[i] = num_str[digits - 1 - i];
                num_str[digits - 1 - i] = tmp;
            }
        }
        
        // Copy the number string into the payload.
        for (int i = 0; i < digits; i++) {
            slot->payload[13 + i] = num_str[i];
        }
        slot->payload[13 + digits] = '\n'; // Add newline for readability

        // Set the final length and other metadata.
        slot->len = 14 + digits; // 13 (prefix) + digits + 1 (newline)
        slot->flags = 0;
        slot->timestamp = clock64();


        // __threadfence_system() guarantees that ALL writes made by this thread
        // before the fence (i.e., the payload, len, flags) are fully visible
        // to the CPU and all other GPU cores BEFORE any write after the fence.
        __threadfence_system();

        // Writing the correct sequence number acts as the
        // "ready" flag for the CPU consumer. We use atomicExch for a strong
        // memory operation, though a simple `slot->seq = seq;` would also work
        // after the system fence.
        atomicExch((unsigned long long*)&slot->seq, seq);
    }
    
    // Update the global write index ONCE.
    // All threads wait here to ensure their individual slot writes are complete.
    __syncthreads();
    
    // Only thread 0 performs the final atomic update to the global index.
    // This avoids 50 separate atomic operations, reducing contention significantly.
    if (tid == 0) {
        atomicAdd((unsigned long long*)&hdr->producer.write_idx, (unsigned long long)n_messages);
    }
}

// Production kernel with ANSI color support
extern "C" __global__ void perdix_ansi_kernel(
    Slot* __restrict__ slots,
    Header* __restrict__ hdr,
    const uint8_t* __restrict__ input_text,
    const uint32_t* __restrict__ text_lengths,
    const uint8_t* __restrict__ agent_types,
    int n_messages
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    
    // ANSI color codes for different agent types
    const char* colors[8] = {
        "\033[34m",  // SYSTEM - Blue
        "\033[32m",  // USER - Green  
        "\033[36m",  // ASSISTANT - Cyan
        "\033[31m",  // ERROR - Red
        "\033[33m",  // WARNING - Yellow
        "\033[37m",  // INFO - White
        "\033[35m",  // DEBUG - Magenta
        "\033[90m"   // TRACE - Bright Black
    };
    
    const char* reset = "\033[0m";
    
    for (int msg_id = tid; msg_id < n_messages; msg_id += stride) {
        // Reserve slot
        uint64_t seq = atomicAdd((unsigned long long*)&hdr->producer.write_idx, 1ULL);
        uint64_t slot_idx = seq & hdr->config.wrap_mask;
        Slot* slot = &slots[slot_idx];
        
        uint32_t output_len = 0;
        uint8_t agent_type = agent_types[msg_id];
        uint32_t text_len = text_lengths[msg_id];
        
        // Add color code
        if (agent_type < 8) {
            const char* color = colors[agent_type];
            for (int i = 0; color[i] != '\0' && output_len < 192; i++) {
                slot->payload[output_len++] = color[i];
            }
        }
        
        // Copy text (up to available space)
        uint32_t text_offset = msg_id * 128;  // Assume max 128 bytes per message
        uint32_t copy_len = min(text_len, (uint32_t)(192 - output_len - 4));
        
        for (uint32_t i = 0; i < copy_len; i++) {
            slot->payload[output_len++] = input_text[text_offset + i];
        }
        
        // Add reset sequence
        for (int i = 0; reset[i] != '\0' && output_len < 192; i++) {
            slot->payload[output_len++] = reset[i];
        }
        
        // Set metadata
        slot->len = output_len;
        slot->flags = agent_type;
        slot->timestamp = clock64();
        
        // Publish
        __threadfence_system();
        atomicExch((unsigned long long*)&slot->seq, seq);
        __threadfence_system();
    }
}

// Optimized kernel using warp-level primitives
extern "C" __global__ void perdix_optimized_kernel(
    Slot* __restrict__ slots,
    Header* __restrict__ hdr,
    const uint8_t* __restrict__ messages,
    const uint32_t* __restrict__ lengths,
    int n_messages
) {
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int lane_id = tid % 32;
    const int warp_id = tid / 32;
    
    // Shared memory for warp-level batching
    extern __shared__ uint64_t shared_seqs[];
    
    // Grid-stride loop
    for (int msg_base = bid * blockDim.x; 
         msg_base < n_messages; 
         msg_base += gridDim.x * blockDim.x) {
        
        int msg_id = msg_base + tid;
        bool has_work = msg_id < n_messages;
        
        // Warp-level cooperative sequence reservation
        uint64_t warp_seq_base = 0;
        uint32_t active_mask = __ballot_sync(0xFFFFFFFF, has_work);
        
        if (active_mask != 0) {
            if (lane_id == __ffs(active_mask) - 1) {
                // Leader thread reserves sequences for entire warp
                uint32_t warp_count = __popc(active_mask);
                warp_seq_base = atomicAdd(
                    (unsigned long long*)&hdr->producer.write_idx,
                    (unsigned long long)warp_count
                );
                atomicAdd((unsigned long long*)&hdr->producer.messages_produced, 
                         (unsigned long long)warp_count);
            }
            
            // Broadcast base sequence to all threads in warp
            warp_seq_base = __shfl_sync(active_mask, warp_seq_base, __ffs(active_mask) - 1);
        }
        
        if (!has_work || warp_seq_base == 0) continue;
        
        // Calculate per-thread sequence
        uint32_t thread_offset = __popc(active_mask & ((1U << lane_id) - 1));
        uint64_t my_seq = warp_seq_base + thread_offset;
        
        // Write to slot
        uint64_t slot_idx = my_seq & hdr->config.wrap_mask;
        Slot* slot = &slots[slot_idx];
        
        // Copy message data
        uint32_t msg_len = lengths[msg_id];
        uint32_t copy_len = min(msg_len, (uint32_t)192);
        const uint8_t* msg_ptr = messages + (msg_id * 192);
        
        // Vectorized copy for better memory throughput
        if (copy_len >= 16) {
            float4* dst = (float4*)slot->payload;
            const float4* src = (const float4*)msg_ptr;
            int vec_count = copy_len / 16;
            
            #pragma unroll 4
            for (int i = 0; i < vec_count; i++) {
                dst[i] = src[i];
            }
            
            // Copy remaining bytes
            for (int i = vec_count * 16; i < copy_len; i++) {
                slot->payload[i] = msg_ptr[i];
            }
        } else {
            for (int i = 0; i < copy_len; i++) {
                slot->payload[i] = msg_ptr[i];
            }
        }
        
        slot->len = copy_len;
        slot->flags = 0;
        slot->timestamp = clock64();
        
        // Publish slot
        __threadfence_system();
        atomicExch((unsigned long long*)&slot->seq, my_seq);
        __threadfence_system();
    }
}

// Unified stream kernel for processing AI agent responses with ANSI formatting
extern "C" __global__ void unified_stream_kernel(
    Slot* __restrict__ slots,
    Header* __restrict__ hdr,
    const PackedStreamContext* __restrict__ contexts,  // Packed contexts with offsets
    const uint8_t* __restrict__ text_arena,            // Contiguous text buffer
    const uint32_t n_messages,
    KernelMetrics* __restrict__ metrics,
    const bool enable_metrics
) {
    const int BLOCK_SIZE = 128;   // Optimized for SM occupancy
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
    
    // Start metrics if enabled
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
        
        // Step 3: Load packed context (now using offsets)
        const PackedStreamContext& ctx = contexts[msg_id];
        
        // Get text pointer from arena using offset
        const uint8_t* text = &text_arena[ctx.text_offset];
        
        // Step 4: Build ANSI-formatted message in shared memory
        uint32_t shared_offset = tid * PAYLOAD_SIZE;
        uint32_t output_len = 0;
        
        if (ctx.enable_ansi) {
            // Add agent type color
            uint8_t color_code[8];
            uint32_t code_len;
            get_agent_color(ctx.agent_type, color_code, code_len);
            
            for (uint32_t i = 0; i < code_len; i++) {
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
            
            if (ctx.enable_ansi) {
                // Check for ERROR keyword - now using text from arena
                if (detect_keyword(text, pos, ctx.text_len, "ERROR", 5)) {
                    const char* red = "\033[91m";
                    for (int j = 0; j < 5; j++) {
                        shared_buffer[shared_offset + output_len++] = red[j];
                    }
                    for (int j = 0; j < 5; j++) {
                        shared_buffer[shared_offset + output_len++] = text[pos + j];
                    }
                    add_reset_sequence(&shared_buffer[shared_offset], output_len);
                    i += 4;
                    highlighted = true;
                }
                // Check for WARNING keyword
                else if (detect_keyword(text, pos, ctx.text_len, "WARNING", 7)) {
                    const char* yellow = "\033[93m";
                    for (int j = 0; j < 5; j++) {
                        shared_buffer[shared_offset + output_len++] = yellow[j];
                    }
                    for (int j = 0; j < 7; j++) {
                        shared_buffer[shared_offset + output_len++] = text[pos + j];
                    }
                    add_reset_sequence(&shared_buffer[shared_offset], output_len);
                    i += 6;
                    highlighted = true;
                }
                // Check for SUCCESS keyword
                else if (detect_keyword(text, pos, ctx.text_len, "SUCCESS", 7)) {
                    const char* green = "\033[92m";
                    for (int j = 0; j < 5; j++) {
                        shared_buffer[shared_offset + output_len++] = green[j];
                    }
                    for (int j = 0; j < 7; j++) {
                        shared_buffer[shared_offset + output_len++] = text[pos + j];
                    }
                    add_reset_sequence(&shared_buffer[shared_offset], output_len);
                    i += 6;
                    highlighted = true;
                }
            }
            
            if (!highlighted) {
                shared_buffer[shared_offset + output_len++] = text[pos];
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

// Kernel for initializing ring buffer
extern "C" __global__ void perdix_init_kernel(
    Slot* __restrict__ slots,
    Header* __restrict__ hdr,
    int n_slots
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    
    // Initialize header (only thread 0)
    if (tid == 0) {
        hdr->producer.write_idx = 0;
        hdr->producer.messages_produced = 0;
        hdr->consumer.read_idx = 0;
        hdr->consumer.messages_consumed = 0;
        hdr->config.slot_count = n_slots;
        hdr->config.wrap_mask = n_slots - 1;
        hdr->config.payload_size = 192;
        hdr->config.batch_size = 32;
        hdr->control.backpressure = 0;
        hdr->control.stop = 0;
        hdr->control.epoch = 0;
        hdr->control.error_count = 0;
    }
    
    // Initialize slots
    for (int i = tid; i < n_slots; i += stride) {
        slots[i].seq = UINT64_MAX;  // Mark as empty
        slots[i].len = 0;
        slots[i].flags = 0;
        slots[i].timestamp = 0;
        
        // Clear payload
        for (int j = 0; j < 192; j++) {
            slots[i].payload[j] = 0;
        }
    }
}

/// Get kernel source with custom configuration
pub fn get_kernel_source(custom_defines: Option<&str>) -> String {
    if let Some(defines) = custom_defines {
        format!("{}\n{}", defines, PERDIX_KERNEL_SOURCE)
    } else {
        PERDIX_KERNEL_SOURCE.to_string()
    }
}

/// Kernel metadata for runtime compilation
pub struct KernelInfo {
    pub name: &'static str,
    pub compute_capability: &'static str,
    pub max_threads: u32,
    pub shared_mem_bytes: u32,
}

impl KernelInfo {
    pub fn test_kernel() -> Self {
        Self {
            name: "perdix_test_kernel",
            compute_capability: "compute_89", // RTX 4070
            max_threads: 256,
            shared_mem_bytes: 0,
        }
    }

    pub fn ansi_kernel() -> Self {
        Self {
            name: "perdix_ansi_kernel",
            compute_capability: "compute_89",
            max_threads: 256,
            shared_mem_bytes: 0,
        }
    }

    pub fn optimized_kernel() -> Self {
        Self {
            name: "perdix_optimized_kernel",
            compute_capability: "compute_89",
            max_threads: 256,
            shared_mem_bytes: 2048, // For sequence batching
        }
    }

    pub fn init_kernel() -> Self {
        Self {
            name: "perdix_init_kernel",
            compute_capability: "compute_89",
            max_threads: 256,
            shared_mem_bytes: 0,
        }
    }

    pub fn unified_stream_kernel() -> Self {
        Self {
            name: "unified_stream_kernel",
            compute_capability: "compute_89",
            max_threads: 224,        // Reduced for shared memory
            shared_mem_bytes: 44800, // 224 * 192 + 224 * 8
        }
    }
}

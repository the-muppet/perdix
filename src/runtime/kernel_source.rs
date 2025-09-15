/// Perdix CUDA kernel source code for runtime compilation
/// This module contains the CUDA C++ source as a string that gets compiled at runtime via NVRTC

pub const PERDIX_KERNEL_SOURCE: &str = r#"
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

// Simple test kernel - writes incrementing messages to ring buffer
extern "C" __global__ void perdix_test_kernel(
    Slot* __restrict__ slots,
    Header* __restrict__ hdr,
    int n_messages
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    
    for (int msg_id = tid; msg_id < n_messages; msg_id += stride) {
        // Reserve sequence number atomically
        uint64_t seq = atomicAdd((unsigned long long*)&hdr->producer.write_idx, 1ULL);
        
        // Calculate slot index using wrap mask
        uint64_t slot_idx = seq & hdr->config.wrap_mask;
        Slot* slot = &slots[slot_idx];
        
        // Write test message
        const char* prefix = "GPU Message ";
        for (int i = 0; i < 12; i++) {
            slot->payload[i] = prefix[i];
        }
        
        // Add message number
        int num = msg_id;
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
            // Reverse digits
            for (int i = 0; i < digits / 2; i++) {
                uint8_t tmp = num_str[i];
                num_str[i] = num_str[digits - 1 - i];
                num_str[digits - 1 - i] = tmp;
            }
        }
        
        for (int i = 0; i < digits; i++) {
            slot->payload[12 + i] = num_str[i];
        }
        slot->payload[12 + digits] = '\n';
        
        // Set metadata
        slot->len = 13 + digits;
        slot->flags = 0;
        slot->timestamp = clock64();
        
        // Memory fence to ensure all writes are visible to host
        __threadfence_system();
        
        // Publish slot by writing sequence number last
        atomicExch((unsigned long long*)&slot->seq, seq);
        __threadfence_system();
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
"#;

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
            compute_capability: "compute_89",  // RTX 4070
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
            shared_mem_bytes: 2048,  // For sequence batching
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
}
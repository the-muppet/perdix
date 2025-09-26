/**
 * @file enhanced_text_kernel.cu
 * @brief Enhanced CUDA Kernels for Text Processing and Streaming AI Responses
 *
 * This kernel implementation focuses on:
 * - Parsing streaming AI responses in GPU memory
 * - Handling multiple concurrent streams from different agents
 * - Building formatted spans for terminal output
 * - Zero-copy transfer using unified memory
 */

 #include <cuda_runtime.h>
 #include <cuda.h>
 #include <cooperative_groups.h>
 #include <stdint.h>
 #include <stdio.h>
 
 namespace cg = cooperative_groups;
 
 // ============================================================================
 // Data Structures
 // ============================================================================
 
 /**
  * @struct TextSpan
  * @brief Represents a formatted text span with ANSI styling
  */
 struct TextSpan {
     uint32_t start_offset;      // Start position in text buffer
     uint32_t length;            // Length of span
     uint8_t fg_r, fg_g, fg_b;  // Foreground color
     uint8_t bg_r, bg_g, bg_b;  // Background color
     uint8_t style_flags;        // Bold, italic, underline, etc.
     uint8_t agent_id;           // Agent that produced this span
 };
 
 /**
  * @struct ParseState
  * @brief Per-stream parsing state for stateful parsing
  */
 struct ParseState {
     uint32_t current_pos;       // Current position in stream
     uint32_t ansi_state;        // ANSI escape sequence state machine
     uint8_t current_fg[3];      // Current foreground color
     uint8_t current_bg[3];      // Current background color
     uint8_t style_flags;        // Current style flags
     bool in_escape_seq;         // Currently parsing escape sequence
     uint8_t escape_buffer[32];  // Buffer for escape sequence
     uint32_t escape_len;        // Length of current escape sequence
 };
 
 /**
  * @struct StreamBuffer
  * @brief Per-agent streaming buffer with metadata
  */
 struct StreamBuffer {
     uint8_t* text_data;         // Raw text data from AI agent
     uint32_t text_length;       // Current text length
     uint32_t buffer_capacity;   // Maximum buffer size
     uint32_t agent_id;          // Agent identifier
     uint64_t timestamp;         // Last update timestamp
     ParseState parse_state;     // Parsing state for this stream
     volatile uint32_t lock;     // Spinlock for concurrent access
 };
 
 /**
  * @struct FormattedOutput
  * @brief Final formatted output ready for terminal
  */
 struct FormattedOutput {
     TextSpan* spans;            // Array of text spans
     uint32_t span_count;        // Number of spans
     uint32_t total_chars;       // Total character count
     uint8_t* formatted_text;    // Final formatted text with ANSI codes
     uint32_t formatted_length;  // Length of formatted text
 };
 
 // Enhanced slot structure for text processing
 struct TextSlot {
     volatile uint64_t sequence;     // Sequence number for ordering
     uint32_t text_offset;          // Offset in global text buffer
     uint32_t text_length;          // Length of text in this slot
     uint32_t agent_id;             // Agent that produced this text
     uint32_t stream_id;            // Stream identifier
     uint8_t agent_type;            // Type of agent (system, user, assistant, etc)
     uint8_t priority;              // Message priority
     uint8_t flags;                 // Various flags
     uint8_t _padding;              // Alignment padding
     TextSpan inline_spans[4];      // Inline spans for small messages
     uint32_t span_count;           // Number of spans in this slot
     uint64_t timestamp;            // Message timestamp
     uint8_t payload[176];          // Additional payload data
 };
 
 static_assert(sizeof(TextSlot) == 320, "TextSlot size must be 320 bytes");
 static_assert(alignof(TextSlot) == 64, "TextSlot must be cache-line aligned");
 
 // ============================================================================
 // Device Functions for Text Processing
 // ============================================================================
 
 /**
  * @brief Parse ANSI escape sequence and update parse state
  */
 __device__ void parse_ansi_escape(
     const uint8_t* text,
     uint32_t& pos,
     uint32_t text_len,
     ParseState& state
 ) {
     // Check for ESC character
     if (pos < text_len && text[pos] == 0x1B) {
         state.in_escape_seq = true;
         state.escape_len = 0;
         state.escape_buffer[state.escape_len++] = text[pos++];
 
         // Parse CSI sequence
         if (pos < text_len && text[pos] == '[') {
             state.escape_buffer[state.escape_len++] = text[pos++];
 
             // Read parameters and command
             while (pos < text_len && state.escape_len < 30) {
                 uint8_t ch = text[pos];
                 state.escape_buffer[state.escape_len++] = ch;
                 pos++;
 
                 // Check for command character (ends sequence)
                 if ((ch >= 0x40 && ch <= 0x7E) || ch == 'm') {
                     // Process the escape sequence
                     if (ch == 'm') {
                         // SGR (Select Graphic Rendition) command
                         process_sgr_command(state);
                     }
                     state.in_escape_seq = false;
                     break;
                 }
             }
         }
     }
 }
 
 /**
  * @brief Process SGR (Select Graphic Rendition) ANSI command
  */
 __device__ void process_sgr_command(ParseState& state) {
     // Parse numeric parameters from escape sequence
     uint32_t params[16];
     uint32_t param_count = 0;
     uint32_t current_param = 0;
     bool has_param = false;
 
     for (uint32_t i = 2; i < state.escape_len - 1 && param_count < 16; i++) {
         uint8_t ch = state.escape_buffer[i];
         if (ch >= '0' && ch <= '9') {
             current_param = current_param * 10 + (ch - '0');
             has_param = true;
         } else if (ch == ';' || ch == 'm') {
             if (has_param) {
                 params[param_count++] = current_param;
             }
             current_param = 0;
             has_param = false;
         }
     }
 
     // Apply parameters to state
     for (uint32_t i = 0; i < param_count; i++) {
         uint32_t param = params[i];
 
         // Reset
         if (param == 0) {
             state.current_fg[0] = 255;
             state.current_fg[1] = 255;
             state.current_fg[2] = 255;
             state.current_bg[0] = 0;
             state.current_bg[1] = 0;
             state.current_bg[2] = 0;
             state.style_flags = 0;
         }
         // Style flags
         else if (param == 1) state.style_flags |= 0x01;  // Bold
         else if (param == 3) state.style_flags |= 0x02;  // Italic
         else if (param == 4) state.style_flags |= 0x04;  // Underline
         // Foreground colors (30-37, 90-97)
         else if (param >= 30 && param <= 37) {
             set_ansi_color(state.current_fg, param - 30, false);
         }
         else if (param >= 90 && param <= 97) {
             set_ansi_color(state.current_fg, param - 90, true);
         }
         // Background colors (40-47, 100-107)
         else if (param >= 40 && param <= 47) {
             set_ansi_color(state.current_bg, param - 40, false);
         }
         else if (param >= 100 && param <= 107) {
             set_ansi_color(state.current_bg, param - 100, true);
         }
         // 256-color mode
         else if (param == 38 && i + 2 < param_count && params[i + 1] == 5) {
             set_256_color(state.current_fg, params[i + 2]);
             i += 2;
         }
         else if (param == 48 && i + 2 < param_count && params[i + 1] == 5) {
             set_256_color(state.current_bg, params[i + 2]);
             i += 2;
         }
     }
 }
 
 /**
  * @brief Set ANSI 16-color palette color
  */
 __device__ void set_ansi_color(uint8_t* color, uint32_t index, bool bright) {
     // Standard ANSI color palette
     const uint8_t normal_colors[8][3] = {
         {0, 0, 0},       // Black
         {170, 0, 0},     // Red
         {0, 170, 0},     // Green
         {170, 85, 0},    // Yellow
         {0, 0, 170},     // Blue
         {170, 0, 170},   // Magenta
         {0, 170, 170},   // Cyan
         {170, 170, 170}  // White
     };
 
     const uint8_t bright_colors[8][3] = {
         {85, 85, 85},    // Bright Black
         {255, 85, 85},   // Bright Red
         {85, 255, 85},   // Bright Green
         {255, 255, 85},  // Bright Yellow
         {85, 85, 255},   // Bright Blue
         {255, 85, 255},  // Bright Magenta
         {85, 255, 255},  // Bright Cyan
         {255, 255, 255}  // Bright White
     };
 
     const uint8_t (*palette)[3] = bright ? bright_colors : normal_colors;
     color[0] = palette[index][0];
     color[1] = palette[index][1];
     color[2] = palette[index][2];
 }
 
 /**
  * @brief Set 256-color palette color
  */
 __device__ void set_256_color(uint8_t* color, uint32_t index) {
     if (index < 16) {
         // Standard 16 colors
         set_ansi_color(color, index % 8, index >= 8);
     } else if (index < 232) {
         // 216-color cube (6x6x6)
         index -= 16;
         uint32_t r = (index / 36) * 51;
         uint32_t g = ((index / 6) % 6) * 51;
         uint32_t b = (index % 6) * 51;
         color[0] = r;
         color[1] = g;
         color[2] = b;
     } else {
         // Grayscale
         uint32_t gray = 8 + (index - 232) * 10;
         color[0] = gray;
         color[1] = gray;
         color[2] = gray;
     }
 }
 
 /**
  * @brief Extract text spans from raw text with ANSI parsing
  */
 __device__ void extract_text_spans(
     const uint8_t* text,
     uint32_t text_len,
     TextSpan* spans,
     uint32_t& span_count,
     uint32_t max_spans,
     ParseState& state
 ) {
     uint32_t pos = 0;
     uint32_t current_span_start = 0;
     span_count = 0;
 
     while (pos < text_len && span_count < max_spans) {
         // Check for ANSI escape sequence
         if (text[pos] == 0x1B) {
             // Save current span if it has content
             if (pos > current_span_start) {
                 spans[span_count].start_offset = current_span_start;
                 spans[span_count].length = pos - current_span_start;
                 spans[span_count].fg_r = state.current_fg[0];
                 spans[span_count].fg_g = state.current_fg[1];
                 spans[span_count].fg_b = state.current_fg[2];
                 spans[span_count].bg_r = state.current_bg[0];
                 spans[span_count].bg_g = state.current_bg[1];
                 spans[span_count].bg_b = state.current_bg[2];
                 spans[span_count].style_flags = state.style_flags;
                 span_count++;
             }
 
             // Parse ANSI escape
             parse_ansi_escape(text, pos, text_len, state);
             current_span_start = pos;
         } else {
             pos++;
         }
     }
 
     // Save final span
     if (pos > current_span_start && span_count < max_spans) {
         spans[span_count].start_offset = current_span_start;
         spans[span_count].length = pos - current_span_start;
         spans[span_count].fg_r = state.current_fg[0];
         spans[span_count].fg_g = state.current_fg[1];
         spans[span_count].fg_b = state.current_fg[2];
         spans[span_count].bg_r = state.current_bg[0];
         spans[span_count].bg_g = state.current_bg[1];
         spans[span_count].bg_b = state.current_bg[2];
         spans[span_count].style_flags = state.style_flags;
         span_count++;
     }
 }
 
 // ============================================================================
 // Main Text Processing Kernel
 // ============================================================================
 
 /**
  * @brief GPU kernel for processing streaming AI text responses
  *
  * This kernel handles multiple concurrent text streams, parses ANSI codes,
  * and generates formatted output spans for terminal display.
  */
 __global__ void text_processing_kernel(
     StreamBuffer* __restrict__ streams,        // Input: per-agent stream buffers
     uint32_t num_streams,                      // Number of active streams
     TextSlot* __restrict__ output_slots,       // Output: ring buffer slots
     volatile uint64_t* __restrict__ write_idx, // Ring buffer write index
     uint32_t slot_count,                       // Total number of slots
     uint8_t* __restrict__ global_text_buffer,  // Global text storage
     uint32_t text_buffer_size,                 // Size of global text buffer
     volatile uint32_t* __restrict__ text_offset, // Current offset in text buffer
     FormattedOutput* __restrict__ formatted_outputs // Formatted output per stream
 ) {
     // Thread/block identifiers
     const int tid = threadIdx.x;
     const int bid = blockIdx.x;
     const int global_tid = bid * blockDim.x + tid;
 
     // Cooperative groups for synchronization
     cg::thread_block block = cg::this_thread_block();
     cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
 
     // Each thread handles one stream
     if (global_tid >= num_streams) return;
 
     StreamBuffer& stream = streams[global_tid];
 
     // Acquire stream lock using atomic CAS
     uint32_t expected = 0;
     while (atomicCAS(&stream.lock, expected, 1) != expected) {
         __nanosleep(100);
         expected = 0;
     }
 
     // Process stream text
     if (stream.text_length > 0) {
         // Allocate space in global text buffer
         uint32_t text_start = atomicAdd((unsigned int*)text_offset, stream.text_length);
 
         if (text_start + stream.text_length <= text_buffer_size) {
             // Copy text to global buffer
             for (uint32_t i = tid; i < stream.text_length; i += blockDim.x) {
                 global_text_buffer[text_start + i] = stream.text_data[i];
             }
             __syncwarp();
 
             // Extract text spans
             TextSpan local_spans[32];
             uint32_t local_span_count = 0;
 
             if (warp.thread_rank() == 0) {
                 extract_text_spans(
                     stream.text_data,
                     stream.text_length,
                     local_spans,
                     local_span_count,
                     32,
                     stream.parse_state
                 );
             }
 
             // Synchronize warp
             warp.sync();
 
             // Broadcast span count to all threads in warp
             local_span_count = __shfl_sync(0xFFFFFFFF, local_span_count, 0);
 
             if (local_span_count > 0 && warp.thread_rank() == 0) {
                 // Reserve slot in ring buffer
                 uint64_t seq = atomicAdd((unsigned long long*)write_idx, 1);
                 uint32_t slot_idx = seq & (slot_count - 1);
                 TextSlot& slot = output_slots[slot_idx];
 
                 // Fill slot with text data
                 slot.text_offset = text_start;
                 slot.text_length = stream.text_length;
                 slot.agent_id = stream.agent_id;
                 slot.stream_id = global_tid;
                 slot.agent_type = global_tid % 8;  // Example agent type assignment
                 slot.priority = 0;
                 slot.flags = 0;
                 slot.timestamp = stream.timestamp;
 
                 // Copy spans to slot (up to 4 inline spans)
                 slot.span_count = min(local_span_count, 4u);
                 for (uint32_t i = 0; i < slot.span_count; i++) {
                     slot.inline_spans[i] = local_spans[i];
                 }
 
                 // Memory fence to ensure all writes are visible
                 __threadfence_system();
 
                 // Publish slot by writing sequence number
                 atomicExch((unsigned long long*)&slot.sequence, seq);
 
                 // Update formatted output
                 FormattedOutput& output = formatted_outputs[global_tid];
                 output.span_count = local_span_count;
                 output.total_chars = stream.text_length;
 
                 // Generate formatted text with ANSI codes
                 generate_formatted_text(
                     stream.text_data,
                     local_spans,
                     local_span_count,
                     output.formatted_text,
                     output.formatted_length
                 );
             }
         }
 
         // Clear processed text from stream
         stream.text_length = 0;
     }
 
     // Release stream lock
     atomicExch(&stream.lock, 0);
 }
 
 /**
  * @brief Generate formatted text with ANSI escape sequences
  */
 __device__ void generate_formatted_text(
     const uint8_t* raw_text,
     const TextSpan* spans,
     uint32_t span_count,
     uint8_t* formatted_text,
     uint32_t& formatted_length
 ) {
     formatted_length = 0;
 
     for (uint32_t i = 0; i < span_count; i++) {
         const TextSpan& span = spans[i];
 
         // Generate ANSI escape sequence for this span
         char escape_seq[64];
         int escape_len = 0;
 
         // Start escape sequence
         escape_seq[escape_len++] = '\033';
         escape_seq[escape_len++] = '[';
 
         // Reset
         escape_seq[escape_len++] = '0';
 
         // Style flags
         if (span.style_flags & 0x01) {
             escape_seq[escape_len++] = ';';
             escape_seq[escape_len++] = '1';  // Bold
         }
         if (span.style_flags & 0x02) {
             escape_seq[escape_len++] = ';';
             escape_seq[escape_len++] = '3';  // Italic
         }
         if (span.style_flags & 0x04) {
             escape_seq[escape_len++] = ';';
             escape_seq[escape_len++] = '4';  // Underline
         }
 
         // Foreground color (24-bit RGB)
         escape_len += sprintf(&escape_seq[escape_len], ";38;2;%d;%d;%d",
                             span.fg_r, span.fg_g, span.fg_b);
 
         // Background color (24-bit RGB)
         escape_len += sprintf(&escape_seq[escape_len], ";48;2;%d;%d;%d",
                             span.bg_r, span.bg_g, span.bg_b);
 
         // End escape sequence
         escape_seq[escape_len++] = 'm';
 
         // Copy escape sequence to output
         for (int j = 0; j < escape_len; j++) {
             formatted_text[formatted_length++] = escape_seq[j];
         }
 
         // Copy span text
         for (uint32_t j = 0; j < span.length; j++) {
             formatted_text[formatted_length++] = raw_text[span.start_offset + j];
         }
     }
 
     // Add reset at the end
     const char* reset = "\033[0m";
     for (int i = 0; reset[i]; i++) {
         formatted_text[formatted_length++] = reset[i];
     }
 }
 
 // ============================================================================
 // Batch Processing Kernel with Warp-Level Optimizations
 // ============================================================================
 
 /**
  * @brief Batch processing kernel for multiple text messages
  *
  * Processes multiple messages in parallel using warp-level primitives
  * to minimize atomic contention and maximize throughput.
  */
 __global__ void batch_text_processing_kernel(
     const uint8_t* __restrict__ text_messages,     // Concatenated text messages
     const uint32_t* __restrict__ message_offsets,  // Offset of each message
     const uint32_t* __restrict__ message_lengths,  // Length of each message
     const uint8_t* __restrict__ agent_types,       // Agent type for each message
     uint32_t num_messages,                         // Total number of messages
     TextSlot* __restrict__ output_slots,          // Output ring buffer
     volatile uint64_t* __restrict__ write_idx,     // Ring buffer write index
     uint32_t slot_count,                          // Number of slots
     uint32_t* __restrict__ processed_count        // Number of messages processed
 ) {
     // Constants
     const int WARP_SIZE = 32;
     const int tid = threadIdx.x;
     const int warp_id = tid / WARP_SIZE;
     const int lane_id = tid % WARP_SIZE;
     const int global_tid = blockIdx.x * blockDim.x + tid;
 
     // Cooperative groups
     cg::thread_block block = cg::this_thread_block();
     cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
 
     // Grid-stride loop for processing messages
     for (uint32_t msg_base = blockIdx.x * blockDim.x / WARP_SIZE;
          msg_base < num_messages;
          msg_base += gridDim.x * blockDim.x / WARP_SIZE) {
 
         // Each warp processes up to WARP_SIZE messages
         uint32_t warp_msg_id = msg_base + warp_id;
         uint32_t msg_id = warp_msg_id * WARP_SIZE + lane_id;
 
         bool has_message = msg_id < num_messages;
 
         // Ballot to find active lanes
         uint32_t active_mask = __ballot_sync(0xFFFFFFFF, has_message);
 
         if (active_mask == 0) continue;
 
         // Warp leader reserves slots for all active messages
         uint64_t warp_seq_base = 0;
         if (lane_id == __ffs(active_mask) - 1) {
             uint32_t active_count = __popc(active_mask);
             warp_seq_base = atomicAdd((unsigned long long*)write_idx, active_count);
             atomicAdd(processed_count, active_count);
         }
 
         // Broadcast base sequence to all lanes
         warp_seq_base = __shfl_sync(active_mask, warp_seq_base, __ffs(active_mask) - 1);
 
         if (has_message) {
             // Calculate this thread's sequence number
             uint32_t thread_offset = __popc(active_mask & ((1U << lane_id) - 1));
             uint64_t my_seq = warp_seq_base + thread_offset;
             uint32_t slot_idx = my_seq & (slot_count - 1);
 
             // Get message data
             uint32_t msg_offset = message_offsets[msg_id];
             uint32_t msg_length = message_lengths[msg_id];
             uint8_t agent_type = agent_types[msg_id];
             const uint8_t* msg_text = &text_messages[msg_offset];
 
             // Process message and fill slot
             TextSlot& slot = output_slots[slot_idx];
 
             // Simple parsing for demonstration
             ParseState state = {};
             state.current_fg[0] = state.current_fg[1] = state.current_fg[2] = 255;
             state.current_bg[0] = state.current_bg[1] = state.current_bg[2] = 0;
 
             // Extract spans
             extract_text_spans(
                 msg_text,
                 msg_length,
                 slot.inline_spans,
                 slot.span_count,
                 4,
                 state
             );
 
             // Fill slot metadata
             slot.text_offset = msg_offset;
             slot.text_length = msg_length;
             slot.agent_id = msg_id % 16;  // Example agent assignment
             slot.stream_id = warp_msg_id;
             slot.agent_type = agent_type;
             slot.priority = 0;
             slot.flags = 0;
             slot.timestamp = clock64();
 
             // Copy small messages directly to payload
             if (msg_length <= sizeof(slot.payload)) {
                 for (uint32_t i = 0; i < msg_length; i++) {
                     slot.payload[i] = msg_text[i];
                 }
             }
 
             // Memory fence for CPU visibility
             __threadfence_system();
 
             // Publish slot
             atomicExch((unsigned long long*)&slot.sequence, my_seq);
         }
     }
 }
 
 // ============================================================================
 // Zero-Copy Producer Kernel Using Unified Memory
 // ============================================================================
 
 /**
  * @brief Zero-copy producer kernel with pinned memory
  *
  * Uses CUDA unified memory and proper memory barriers for
  * CPU-GPU zero-copy communication.
  */
 __global__ void zero_copy_producer_kernel(
     TextSlot* __restrict__ unified_slots,      // Unified memory slots
     volatile uint64_t* __restrict__ write_idx, // Shared write index
     volatile uint64_t* __restrict__ read_idx,  // Shared read index
     uint32_t slot_count,                       // Number of slots
     const uint8_t* __restrict__ input_text,    // Input text data
     const uint32_t* __restrict__ text_lengths, // Length of each text chunk
     const uint8_t* __restrict__ agent_ids,     // Agent ID for each chunk
     uint32_t num_chunks,                       // Number of text chunks
     volatile uint32_t* __restrict__ backpressure_flag // Backpressure indicator
 ) {
     // Thread identification
     const int tid = blockIdx.x * blockDim.x + threadIdx.x;
     const int stride = gridDim.x * blockDim.x;
 
     // Grid-stride loop
     for (uint32_t chunk_id = tid; chunk_id < num_chunks; chunk_id += stride) {
         // Check for backpressure
         uint64_t current_write = atomicAdd((unsigned long long*)write_idx, 0);
         uint64_t current_read = atomicAdd((unsigned long long*)read_idx, 0);
 
         // Calculate available slots
         uint64_t available = slot_count - (current_write - current_read);
 
         if (available < 64) {
             // Signal backpressure
             atomicExch((unsigned int*)backpressure_flag, 1);
 
             // Exponential backoff
             uint32_t backoff = 100 + (tid * 10);
             for (int i = 0; i < 10; i++) {
                 __nanosleep(backoff);
                 backoff *= 2;
 
                 // Recheck
                 current_read = atomicAdd((unsigned long long*)read_idx, 0);
                 available = slot_count - (current_write - current_read);
                 if (available >= 64) {
                     atomicExch((unsigned int*)backpressure_flag, 0);
                     break;
                 }
             }
 
             if (available < 64) continue;
         }
 
         // Reserve slot
         uint64_t seq = atomicAdd((unsigned long long*)write_idx, 1);
         uint32_t slot_idx = seq & (slot_count - 1);
         TextSlot& slot = unified_slots[slot_idx];
 
         // Get chunk data
         uint32_t chunk_offset = 0;
         for (uint32_t i = 0; i < chunk_id; i++) {
             chunk_offset += text_lengths[i];
         }
 
         const uint8_t* chunk_text = &input_text[chunk_offset];
         uint32_t chunk_length = text_lengths[chunk_id];
         uint8_t agent_id = agent_ids[chunk_id];
 
         // Process text and extract spans
         ParseState state = {};
         state.current_fg[0] = state.current_fg[1] = state.current_fg[2] = 255;
         state.current_bg[0] = state.current_bg[1] = state.current_bg[2] = 0;
 
         extract_text_spans(
             chunk_text,
             chunk_length,
             slot.inline_spans,
             slot.span_count,
             4,
             state
         );
 
         // Fill slot
         slot.text_offset = chunk_offset;
         slot.text_length = chunk_length;
         slot.agent_id = agent_id;
         slot.stream_id = chunk_id;
         slot.agent_type = agent_id % 8;
         slot.priority = 0;
         slot.flags = 0;
         slot.timestamp = clock64();
 
         // Copy text to payload if it fits
         if (chunk_length <= sizeof(slot.payload)) {
             for (uint32_t i = 0; i < chunk_length; i++) {
                 slot.payload[i] = chunk_text[i];
             }
         }
 
         // CRITICAL: System-wide memory fence for CPU visibility
         __threadfence_system();
 
         // Publish slot with atomic exchange for strongest ordering
         atomicExch((unsigned long long*)&slot.sequence, seq);
 
         // Additional fence to ensure publication is visible
         __threadfence_system();
     }
 }
 
 // ============================================================================
 // C Interface Functions
 // ============================================================================
 
 extern "C" {
 
 /**
  * @brief Initialize text processing system
  */
 int init_text_processing(
     TextSlot** slots,
     uint32_t slot_count,
     StreamBuffer** streams,
     uint32_t max_streams,
     uint8_t** text_buffer,
     uint32_t text_buffer_size
 ) {
     cudaError_t err;
 
     // Allocate unified memory for slots
     size_t slots_size = slot_count * sizeof(TextSlot);
     err = cudaMallocManaged((void**)slots, slots_size, cudaMemAttachGlobal);
     if (err != cudaSuccess) {
         printf("Failed to allocate text slots: %s\n", cudaGetErrorString(err));
         return -1;
     }
 
     // Initialize slots
     memset(*slots, 0, slots_size);
     for (uint32_t i = 0; i < slot_count; i++) {
         (*slots)[i].sequence = UINT64_MAX;
     }
 
     // Allocate stream buffers
     size_t streams_size = max_streams * sizeof(StreamBuffer);
     err = cudaMallocManaged((void**)streams, streams_size, cudaMemAttachGlobal);
     if (err != cudaSuccess) {
         printf("Failed to allocate stream buffers: %s\n", cudaGetErrorString(err));
         cudaFree(*slots);
         return -1;
     }
 
     // Initialize stream buffers
     memset(*streams, 0, streams_size);
     for (uint32_t i = 0; i < max_streams; i++) {
         (*streams)[i].buffer_capacity = 64 * 1024;  // 64KB per stream
         err = cudaMallocManaged((void**)&(*streams)[i].text_data,
                                 (*streams)[i].buffer_capacity,
                                 cudaMemAttachGlobal);
         if (err != cudaSuccess) {
             printf("Failed to allocate stream text buffer: %s\n", cudaGetErrorString(err));
             // Cleanup
             for (uint32_t j = 0; j < i; j++) {
                 cudaFree((*streams)[j].text_data);
             }
             cudaFree(*streams);
             cudaFree(*slots);
             return -1;
         }
     }
 
     // Allocate global text buffer
     err = cudaMallocManaged((void**)text_buffer, text_buffer_size, cudaMemAttachGlobal);
     if (err != cudaSuccess) {
         printf("Failed to allocate global text buffer: %s\n", cudaGetErrorString(err));
         // Cleanup
         for (uint32_t i = 0; i < max_streams; i++) {
             cudaFree((*streams)[i].text_data);
         }
         cudaFree(*streams);
         cudaFree(*slots);
         return -1;
     }
 
     printf("Text processing system initialized:\n");
     printf("  Slots: %u (%.2f MB)\n", slot_count, slots_size / (1024.0 * 1024.0));
     printf("  Streams: %u\n", max_streams);
     printf("  Text buffer: %.2f MB\n", text_buffer_size / (1024.0 * 1024.0));
 
     return 0;
 }
 
 /**
  * @brief Launch text processing kernel
  */
 int launch_text_processing(
     StreamBuffer* streams,
     uint32_t num_streams,
     TextSlot* output_slots,
     uint64_t* write_idx,
     uint32_t slot_count,
     uint8_t* global_text_buffer,
     uint32_t text_buffer_size,
     uint32_t* text_offset,
     FormattedOutput* formatted_outputs,
     cudaStream_t stream
 ) {
     // Calculate launch configuration
     int threads_per_block = 128;
     int blocks = (num_streams + threads_per_block - 1) / threads_per_block;
     blocks = min(blocks, 64);  // Cap at reasonable number
 
     // Launch kernel
     text_processing_kernel<<<blocks, threads_per_block, 0, stream>>>(
         streams,
         num_streams,
         output_slots,
         write_idx,
         slot_count,
         global_text_buffer,
         text_buffer_size,
         text_offset,
         formatted_outputs
     );
 
     cudaError_t err = cudaGetLastError();
     if (err != cudaSuccess) {
         printf("Text processing kernel launch failed: %s\n", cudaGetErrorString(err));
         return -1;
     }
 
     return 0;
 }
 
 /**
  * @brief Launch batch text processing kernel
  */
 int launch_batch_text_processing(
     const uint8_t* text_messages,
     const uint32_t* message_offsets,
     const uint32_t* message_lengths,
     const uint8_t* agent_types,
     uint32_t num_messages,
     TextSlot* output_slots,
     uint64_t* write_idx,
     uint32_t slot_count,
     uint32_t* processed_count,
     cudaStream_t stream
 ) {
     // Calculate optimal launch configuration
     cudaDeviceProp prop;
     cudaGetDeviceProperties(&prop, 0);
 
     int threads_per_block = 256;  // 8 warps per block
     int max_blocks = prop.multiProcessorCount * 2;
     int blocks = min((num_messages + threads_per_block - 1) / threads_per_block, max_blocks);
 
     // Launch kernel
     batch_text_processing_kernel<<<blocks, threads_per_block, 0, stream>>>(
         text_messages,
         message_offsets,
         message_lengths,
         agent_types,
         num_messages,
         output_slots,
         write_idx,
         slot_count,
         processed_count
     );
 
     cudaError_t err = cudaGetLastError();
     if (err != cudaSuccess) {
         printf("Batch text processing kernel launch failed: %s\n", cudaGetErrorString(err));
         return -1;
     }
 
     return 0;
 }
 
 /**
  * @brief Launch zero-copy producer kernel
  */
 int launch_zero_copy_producer(
     TextSlot* unified_slots,
     uint64_t* write_idx,
     uint64_t* read_idx,
     uint32_t slot_count,
     const uint8_t* input_text,
     const uint32_t* text_lengths,
     const uint8_t* agent_ids,
     uint32_t num_chunks,
     uint32_t* backpressure_flag,
     cudaStream_t stream
 ) {
     // Calculate launch configuration
     int threads_per_block = 256;
     int blocks = min((num_chunks + threads_per_block - 1) / threads_per_block, 128);
 
     // Launch kernel
     zero_copy_producer_kernel<<<blocks, threads_per_block, 0, stream>>>(
         unified_slots,
         write_idx,
         read_idx,
         slot_count,
         input_text,
         text_lengths,
         agent_ids,
         num_chunks,
         backpressure_flag
     );
 
     cudaError_t err = cudaGetLastError();
     if (err != cudaSuccess) {
         printf("Zero-copy producer kernel launch failed: %s\n", cudaGetErrorString(err));
         return -1;
     }
 
     return 0;
 }
 
 /**
  * @brief Cleanup text processing system
  */
 int cleanup_text_processing(
     TextSlot* slots,
     StreamBuffer* streams,
     uint32_t num_streams,
     uint8_t* text_buffer
 ) {
     cudaDeviceSynchronize();
 
     if (slots) cudaFree(slots);
     if (text_buffer) cudaFree(text_buffer);
 
     if (streams) {
         for (uint32_t i = 0; i < num_streams; i++) {
             if (streams[i].text_data) {
                 cudaFree(streams[i].text_data);
             }
         }
         cudaFree(streams);
     }
 
     return 0;
 }
 
 } // extern "C"
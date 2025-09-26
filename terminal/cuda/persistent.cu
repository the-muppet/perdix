// Persistent GPU Rendering Kernel for Real-Time TUI Effects
// This kernel runs continuously on the GPU, rendering frames at maximum speed

#include <cuda_runtime.h>
#include <cuda.h>
#include <stdint.h>
#include <math.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

// Terminal cell structure (matching Rust side)
struct TerminalCell {
    uint32_t codepoint;
    uint8_t fg_r, fg_g, fg_b, fg_a;
    uint8_t bg_r, bg_g, bg_b, bg_a;
    uint8_t attributes;
    uint8_t padding[3];
};

// Control block for GPU-CPU communication
struct RenderControl {
    // Control flags
    volatile int stop_flag;           // Set to 1 to stop the kernel
    volatile int effects_dirty;       // Set to 1 when effects change
    volatile int source_dirty;        // Set to 1 when source cells update

    // Timing
    float time;                        // Current animation time
    float delta_time;                  // Time since last frame
    uint64_t frame_count;              // Total frames rendered

    // Effect parameters (updated by CPU)
    float blur_intensity;
    float glow_intensity;
    float glow_threshold;
    float wave_amplitude;
    float wave_frequency;
    float fade_opacity;
    uint8_t fade_color[3];
    float scanline_intensity;
    int scanline_spacing;
    float chromatic_r_offset;
    float chromatic_g_offset;
    float chromatic_b_offset;

    // Smooth scrolling physics
    float scroll_position;
    float scroll_velocity;
    float scroll_target;
    float scroll_spring_strength;
    float scroll_damping;

    // Performance metrics
    float last_frame_time_ms;
    float avg_frame_time_ms;
    uint32_t dropped_frames;

    // Active effect flags
    uint32_t active_effects;  // Bitmask of enabled effects
};

// Effect type flags
#define EFFECT_BLUR           (1 << 0)
#define EFFECT_GLOW           (1 << 1)
#define EFFECT_WAVE           (1 << 2)
#define EFFECT_SCANLINES      (1 << 3)
#define EFFECT_FADE           (1 << 4)
#define EFFECT_CHROMATIC      (1 << 5)
#define EFFECT_MATRIX_RAIN    (1 << 6)
#define EFFECT_SMOOTH_SCROLL  (1 << 7)

// Frame notification structure for ring buffer
struct FrameNotification {
    uint64_t frame_id;
    float render_time_ms;
    uint32_t effects_applied;
    uint8_t ready_flag;
    uint8_t padding[3];
};

// Ring buffer slot for frame notifications
struct RingSlot {
    volatile uint64_t sequence;
    uint32_t length;
    uint32_t type;
    FrameNotification notification;
    uint8_t padding[256 - sizeof(uint64_t) - sizeof(uint32_t) * 2 - sizeof(FrameNotification)];
};

// Ring buffer header
struct RingHeader {
    volatile uint64_t producer_seq;
    uint8_t cache_pad1[64 - sizeof(uint64_t)];

    volatile uint64_t consumer_seq;
    uint8_t cache_pad2[64 - sizeof(uint64_t)];

    uint32_t slot_count;
    uint32_t slot_size;
    uint8_t cache_pad3[128 - 2 * sizeof(uint32_t)];
};

// Shared memory for cooperative rendering
__shared__ TerminalCell shared_cells[32][32];
__shared__ float shared_effects[32];

// Device function: Apply smooth scrolling physics
__device__ void update_scroll_physics(RenderControl* control, float dt) {
    if (control->active_effects & EFFECT_SMOOTH_SCROLL) {
        // Spring physics for smooth scrolling
        float distance = control->scroll_target - control->scroll_position;
        float spring_force = distance * control->scroll_spring_strength;

        // Apply damping
        control->scroll_velocity += spring_force * dt;
        control->scroll_velocity *= (1.0f - control->scroll_damping * dt);

        // Update position
        control->scroll_position += control->scroll_velocity * dt;

        // Snap to target if very close
        if (fabsf(distance) < 0.01f && fabsf(control->scroll_velocity) < 0.01f) {
            control->scroll_position = control->scroll_target;
            control->scroll_velocity = 0.0f;
        }
    }
}

// Device function: Apply blur effect inline
__device__ void apply_blur_inline(
    TerminalCell* input,
    TerminalCell* output,
    int x, int y,
    int width, int height,
    float intensity
) {
    const float kernel[5][5] = {
        {0.003765f, 0.015019f, 0.023792f, 0.015019f, 0.003765f},
        {0.015019f, 0.059912f, 0.094907f, 0.059912f, 0.015019f},
        {0.023792f, 0.094907f, 0.150342f, 0.094907f, 0.023792f},
        {0.015019f, 0.059912f, 0.094907f, 0.059912f, 0.015019f},
        {0.003765f, 0.015019f, 0.023792f, 0.015019f, 0.003765f}
    };

    float acc_fg_r = 0, acc_fg_g = 0, acc_fg_b = 0;
    float acc_bg_r = 0, acc_bg_g = 0, acc_bg_b = 0;
    float weight_sum = 0;

    for (int dy = -2; dy <= 2; dy++) {
        for (int dx = -2; dx <= 2; dx++) {
            int sx = x + dx;
            int sy = y + dy;

            if (sx >= 0 && sx < width && sy >= 0 && sy < height) {
                int idx = sy * width + sx;
                float weight = kernel[dy + 2][dx + 2];

                acc_fg_r += input[idx].fg_r * weight;
                acc_fg_g += input[idx].fg_g * weight;
                acc_fg_b += input[idx].fg_b * weight;
                acc_bg_r += input[idx].bg_r * weight;
                acc_bg_g += input[idx].bg_g * weight;
                acc_bg_b += input[idx].bg_b * weight;
                weight_sum += weight;
            }
        }
    }

    int idx = y * width + x;
    TerminalCell result = input[idx];

    if (weight_sum > 0) {
        float inv_weight = 1.0f / weight_sum;
        result.fg_r = (uint8_t)(result.fg_r * (1.0f - intensity) + (acc_fg_r * inv_weight) * intensity);
        result.fg_g = (uint8_t)(result.fg_g * (1.0f - intensity) + (acc_fg_g * inv_weight) * intensity);
        result.fg_b = (uint8_t)(result.fg_b * (1.0f - intensity) + (acc_fg_b * inv_weight) * intensity);
        result.bg_r = (uint8_t)(result.bg_r * (1.0f - intensity) + (acc_bg_r * inv_weight) * intensity);
        result.bg_g = (uint8_t)(result.bg_g * (1.0f - intensity) + (acc_bg_g * inv_weight) * intensity);
        result.bg_b = (uint8_t)(result.bg_b * (1.0f - intensity) + (acc_bg_b * inv_weight) * intensity);
    }

    output[idx] = result;
}

// Device function: Apply wave distortion inline
__device__ void apply_wave_inline(
    TerminalCell* input,
    TerminalCell* output,
    int x, int y,
    int width, int height,
    float time,
    float amplitude,
    float frequency
) {
    // Calculate wave offset
    float wave_x = amplitude * sinf(frequency * y + time);
    float wave_y = amplitude * cosf(frequency * x + time * 0.7f);

    // Sample from distorted position
    int src_x = x + (int)wave_x;
    int src_y = y + (int)wave_y;

    // Clamp to boundaries
    src_x = max(0, min(width - 1, src_x));
    src_y = max(0, min(height - 1, src_y));

    output[y * width + x] = input[src_y * width + src_x];
}

// Device function: Apply glow effect inline
__device__ void apply_glow_inline(
    TerminalCell* cell,
    float intensity,
    float threshold
) {
    // Calculate luminance
    float lum = 0.299f * cell->fg_r + 0.587f * cell->fg_g + 0.114f * cell->fg_b;
    float bloom = fmaxf(0.0f, (lum / 255.0f) - threshold);

    if (bloom > 0) {
        float glow_factor = 1.0f + intensity * bloom;
        cell->fg_r = min(255, (int)(cell->fg_r * glow_factor));
        cell->fg_g = min(255, (int)(cell->fg_g * glow_factor));
        cell->fg_b = min(255, (int)(cell->fg_b * glow_factor));
    }
}

// Device function: Publish frame ready notification
__device__ void publish_frame_notification(
    RingHeader* header,
    RingSlot* slots,
    uint64_t frame_id,
    float render_time,
    uint32_t effects
) {
    // Get next sequence number
    uint64_t seq = atomicAdd((unsigned long long*)&header->producer_seq, 1);
    uint32_t slot_idx = seq & (header->slot_count - 1);

    RingSlot* slot = &slots[slot_idx];

    // Wait for slot to be available
    while (slot->sequence != seq) {
        __threadfence();
    }

    // Write notification
    slot->notification.frame_id = frame_id;
    slot->notification.render_time_ms = render_time;
    slot->notification.effects_applied = effects;
    slot->notification.ready_flag = 1;
    slot->length = sizeof(FrameNotification);
    slot->type = 0xFF; // Frame notification type

    // Make visible with release semantics
    __threadfence();
    slot->sequence = seq + header->slot_count;
}

// Main persistent rendering kernel
extern "C" __global__ void persistent_render_kernel(
    TerminalCell* d_source_cells,      // Source TUI state (updated by CPU)
    TerminalCell* d_rendered_cells,    // Output buffer for display
    TerminalCell* d_temp_buffer,       // Temporary buffer for multi-pass
    RenderControl* control,            // Control block
    RingHeader* ring_header,           // Ring buffer header
    RingSlot* ring_slots,              // Ring buffer slots
    int width,
    int height,
    int total_cells
) {
    // Grid-wide synchronization group for cooperative effects
    cg::grid_group grid = cg::this_grid();

    // Calculate thread's cell responsibility
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int cells_per_thread = (total_cells + blockDim.x * gridDim.x - 1) / (blockDim.x * gridDim.x);
    int start_cell = tid * cells_per_thread;
    int end_cell = min(start_cell + cells_per_thread, total_cells);

    // Local timing
    clock_t last_frame_clock = clock();

    // Main rendering loop - runs until stop flag is set
    while (!control->stop_flag) {
        clock_t frame_start = clock();

        // Update physics (only thread 0)
        if (tid == 0) {
            update_scroll_physics(control, control->delta_time);
            control->frame_count++;
        }
        grid.sync();

        // Phase 1: Copy source to working buffer with scroll offset
        for (int i = start_cell; i < end_cell; i++) {
            int x = i % width;
            int y = i / width;

            // Apply scroll offset if enabled
            if (control->active_effects & EFFECT_SMOOTH_SCROLL) {
                int src_y = y + (int)control->scroll_position;
                src_y = max(0, min(height - 1, src_y));
                d_temp_buffer[i] = d_source_cells[src_y * width + x];
            } else {
                d_temp_buffer[i] = d_source_cells[i];
            }
        }
        grid.sync();

        // Phase 2: Apply effects based on active flags
        TerminalCell* current_input = d_temp_buffer;
        TerminalCell* current_output = d_rendered_cells;

        // Wave effect
        if (control->active_effects & EFFECT_WAVE) {
            for (int i = start_cell; i < end_cell; i++) {
                int x = i % width;
                int y = i / width;
                apply_wave_inline(current_input, current_output, x, y, width, height,
                                control->time, control->wave_amplitude, control->wave_frequency);
            }
            grid.sync();
            // Swap buffers
            TerminalCell* temp = current_input;
            current_input = current_output;
            current_output = temp;
        }

        // Blur effect
        if (control->active_effects & EFFECT_BLUR) {
            for (int i = start_cell; i < end_cell; i++) {
                int x = i % width;
                int y = i / width;
                apply_blur_inline(current_input, current_output, x, y, width, height,
                                control->blur_intensity);
            }
            grid.sync();
            TerminalCell* temp = current_input;
            current_input = current_output;
            current_output = temp;
        }

        // Glow effect (in-place)
        if (control->active_effects & EFFECT_GLOW) {
            for (int i = start_cell; i < end_cell; i++) {
                apply_glow_inline(&current_input[i], control->glow_intensity, control->glow_threshold);
            }
            grid.sync();
        }

        // Scanlines effect (in-place)
        if (control->active_effects & EFFECT_SCANLINES) {
            for (int i = start_cell; i < end_cell; i++) {
                int y = i / width;
                if (y % control->scanline_spacing == 0) {
                    float darken = 1.0f - control->scanline_intensity;
                    current_input[i].fg_r = (uint8_t)(current_input[i].fg_r * darken);
                    current_input[i].fg_g = (uint8_t)(current_input[i].fg_g * darken);
                    current_input[i].fg_b = (uint8_t)(current_input[i].fg_b * darken);
                }
            }
            grid.sync();
        }

        // Fade effect (in-place)
        if (control->active_effects & EFFECT_FADE) {
            for (int i = start_cell; i < end_cell; i++) {
                float inv_opacity = 1.0f - control->fade_opacity;
                current_input[i].fg_r = (uint8_t)(current_input[i].fg_r * control->fade_opacity +
                                                 control->fade_color[0] * inv_opacity);
                current_input[i].fg_g = (uint8_t)(current_input[i].fg_g * control->fade_opacity +
                                                 control->fade_color[1] * inv_opacity);
                current_input[i].fg_b = (uint8_t)(current_input[i].fg_b * control->fade_opacity +
                                                 control->fade_color[2] * inv_opacity);
            }
            grid.sync();
        }

        // Ensure final output is in d_rendered_cells
        if (current_input != d_rendered_cells) {
            for (int i = start_cell; i < end_cell; i++) {
                d_rendered_cells[i] = current_input[i];
            }
            grid.sync();
        }

        // Phase 3: Calculate frame timing and publish notification
        if (tid == 0) {
            clock_t frame_end = clock();
            float frame_time_ms = (float)(frame_end - frame_start) / (float)CLOCKS_PER_SEC * 1000.0f;

            // Update timing in control block
            control->last_frame_time_ms = frame_time_ms;
            control->avg_frame_time_ms = control->avg_frame_time_ms * 0.95f + frame_time_ms * 0.05f;
            control->time += control->delta_time;

            // Publish frame ready notification to ring buffer
            publish_frame_notification(
                ring_header,
                ring_slots,
                control->frame_count,
                frame_time_ms,
                control->active_effects
            );
        }

        // Small delay to prevent overheating (can be tuned)
        // This ensures we don't spin-wait too aggressively
        __nanosleep(100000); // 0.1ms
    }
}

// Helper kernel to initialize control block
extern "C" __global__ void init_render_control(RenderControl* control) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        control->stop_flag = 0;
        control->effects_dirty = 0;
        control->source_dirty = 0;
        control->time = 0.0f;
        control->delta_time = 0.016f; // 60 FPS default
        control->frame_count = 0;

        // Default effect parameters
        control->blur_intensity = 0.0f;
        control->glow_intensity = 0.0f;
        control->glow_threshold = 0.8f;
        control->wave_amplitude = 0.0f;
        control->wave_frequency = 0.1f;
        control->fade_opacity = 1.0f;
        control->fade_color[0] = 0;
        control->fade_color[1] = 0;
        control->fade_color[2] = 0;
        control->scanline_intensity = 0.0f;
        control->scanline_spacing = 2;

        // Scroll physics defaults
        control->scroll_position = 0.0f;
        control->scroll_velocity = 0.0f;
        control->scroll_target = 0.0f;
        control->scroll_spring_strength = 10.0f;
        control->scroll_damping = 0.8f;

        // Performance metrics
        control->last_frame_time_ms = 0.0f;
        control->avg_frame_time_ms = 16.67f;
        control->dropped_frames = 0;

        control->active_effects = 0;
    }
}

// Helper kernel to stop the renderer
extern "C" __global__ void stop_renderer(RenderControl* control) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        control->stop_flag = 1;
    }
}
//! Persistent GPU Renderer Integration
//!
//! This module manages the persistent GPU rendering kernel that continuously
//! renders frames at maximum speed while the CPU only displays the results.

use std::sync::{Arc, Mutex, RwLock};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::thread;
use std::time::{Duration, Instant};
use std::os::raw::c_void;
use std::ptr;
use std::ffi::CString;
use tokio::task::JoinHandle;

use crate::buffer::{Buffer as SharedBuffer, ConsumerHandle, Message};
use crate::fx::cuda_ffi::TerminalCell;

#[cfg(feature = "cuda")]
use crate::cuda::{
    cuInit, cuDeviceGet, cuCtxCreate_v2, cuModuleLoad, cuModuleGetFunction,
    cuStreamCreate, cuStreamDestroy_v2, cuStreamSynchronize, cuLaunchCooperativeKernel,
    cudaMalloc, cudaFree, cudaMemcpy, cudaMemcpyAsync,
    cudaMemcpyKind, CUresult, CUdevice, CUcontext, CUmodule, CUfunction, CUstream,
    cudaStream_t,
    CU_STREAM_NON_BLOCKING, CU_STREAM_DEFAULT,
};
/// Control block for GPU-CPU communication
#[repr(C)]
#[derive(Debug)]
pub struct RenderControl {
    // Control flags
    pub stop_flag: i32,
    pub effects_dirty: i32,
    pub source_dirty: i32,

    // Timing
    pub time: f32,
    pub delta_time: f32,
    pub frame_count: u64,

    // Effect parameters
    pub blur_intensity: f32,
    pub glow_intensity: f32,
    pub glow_threshold: f32,
    pub wave_amplitude: f32,
    pub wave_frequency: f32,
    pub fade_opacity: f32,
    pub fade_color: [u8; 3],
    pub scanline_intensity: f32,
    pub scanline_spacing: i32,
    pub chromatic_r_offset: f32,
    pub chromatic_g_offset: f32,
    pub chromatic_b_offset: f32,

    // Smooth scrolling physics
    pub scroll_position: f32,
    pub scroll_velocity: f32,
    pub scroll_target: f32,
    pub scroll_spring_strength: f32,
    pub scroll_damping: f32,

    // Performance metrics
    pub last_frame_time_ms: f32,
    pub avg_frame_time_ms: f32,
    pub dropped_frames: u32,

    // Active effect flags
    pub active_effects: u32,
}

impl Default for RenderControl {
    fn default() -> Self {
        Self {
            stop_flag: 0,
            effects_dirty: 0,
            source_dirty: 0,
            time: 0.0,
            delta_time: 0.016,
            frame_count: 0,
            blur_intensity: 0.0,
            glow_intensity: 0.0,
            glow_threshold: 0.8,
            wave_amplitude: 0.0,
            wave_frequency: 0.1,
            fade_opacity: 1.0,
            fade_color: [0, 0, 0],
            scanline_intensity: 0.0,
            scanline_spacing: 2,
            chromatic_r_offset: 0.0,
            chromatic_g_offset: 0.0,
            chromatic_b_offset: 0.0,
            scroll_position: 0.0,
            scroll_velocity: 0.0,
            scroll_target: 0.0,
            scroll_spring_strength: 10.0,
            scroll_damping: 0.8,
            last_frame_time_ms: 0.0,
            avg_frame_time_ms: 16.67,
            dropped_frames: 0,
            active_effects: 0,
        }
    }
}

// Effect type flags matching CUDA
pub const EFFECT_BLUR: u32 = 1 << 0;
pub const EFFECT_GLOW: u32 = 1 << 1;
pub const EFFECT_WAVE: u32 = 1 << 2;
pub const EFFECT_SCANLINES: u32 = 1 << 3;
pub const EFFECT_FADE: u32 = 1 << 4;
pub const EFFECT_CHROMATIC: u32 = 1 << 5;
pub const EFFECT_MATRIX_RAIN: u32 = 1 << 6;
pub const EFFECT_SMOOTH_SCROLL: u32 = 1 << 7;

/// Frame notification from GPU
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FrameNotification {
    pub frame_id: u64,
    pub render_time_ms: f32,
    pub effects_applied: u32,
    pub ready_flag: u8,
    pub padding: [u8; 3],
}

/// Persistent GPU renderer that runs continuously
pub struct PersistentGpuRenderer {
    // Device buffers
    #[cfg(feature = "cuda")]
    d_source_cells: *mut TerminalCell,
    #[cfg(feature = "cuda")]
    d_rendered_cells: *mut TerminalCell,
    #[cfg(feature = "cuda")]
    d_temp_buffer: *mut TerminalCell,
    #[cfg(feature = "cuda")]
    d_control: *mut RenderControl,

    // Host buffers
    host_source_cells: Vec<TerminalCell>,
    host_display_cells: Vec<TerminalCell>,

    // Control
    control: Arc<RwLock<RenderControl>>,
    running: Arc<AtomicBool>,
    frame_counter: Arc<AtomicU64>,

    // Dimensions
    width: usize,
    height: usize,
    total_cells: usize,

    // CUDA resources
    #[cfg(feature = "cuda")]
    stream: CUstream,
    #[cfg(feature = "cuda")]
    kernel_handle: Option<thread::JoinHandle<()>>,

    // Frame synchronization
    frame_consumer: Option<ConsumerHandle>, 
    last_frame_id: u64,

    // Performance tracking
    fps_counter: Arc<RwLock<FpsCounter>>,
}

pub struct FpsCounter {
    pub current_fps: f32,
    pub avg_fps: f32,
    pub frame_times: Vec<f32>,
    pub last_update: Instant,
}

impl PersistentGpuRenderer {
    #[cfg(feature = "cuda")]
    pub fn new(
        width: usize,
        height: usize,
        frame_buffer: SharedBuffer, // Takes ownership to create handles
    ) -> Result<Self, String> {
        let total_cells = width * height;

        unsafe {
            // Initialize CUDA
            let result = cuInit(0);
            if result != CUresult::CUDA_SUCCESS {
                return Err(format!("Failed to initialize CUDA: {:?}", result));
            }

            // Get device and create context
            let mut device: CUdevice = 0;
            cuDeviceGet(&mut device, 0);

            let mut context: CUcontext = ptr::null_mut();
            cuCtxCreate_v2(&mut context, 0, device);

            // Create stream for async operations
            let mut stream: CUstream = ptr::null_mut();
            cuStreamCreate(&mut stream, CU_STREAM_DEFAULT as std::os::raw::c_uint);

            // Allocate device memory
            let cell_size = std::mem::size_of::<TerminalCell>();
            let buffer_size = total_cells * cell_size;

            let mut d_source_cells: *mut c_void = ptr::null_mut();
            let mut d_rendered_cells: *mut c_void = ptr::null_mut();
            let mut d_temp_buffer: *mut c_void = ptr::null_mut();
            let mut d_control: *mut c_void = ptr::null_mut();

            cudaMalloc(&mut d_source_cells, buffer_size);
            cudaMalloc(&mut d_rendered_cells, buffer_size);
            cudaMalloc(&mut d_temp_buffer, buffer_size);
            cudaMalloc(&mut d_control, std::mem::size_of::<RenderControl>());

            // Initialize control block
            let control = RenderControl::default();
            cudaMemcpy(
                d_control,
                &control as *const _ as *const c_void,
                std::mem::size_of::<RenderControl>(),
                cudaMemcpyKind::cudaMemcpyHostToDevice,
            );

            // frame consumer 
            let (_, _, frame_consumer) = frame_buffer.create_handles();

            Ok(Self {
                d_source_cells: d_source_cells as *mut TerminalCell,
                d_rendered_cells: d_rendered_cells as *mut TerminalCell,
                d_temp_buffer: d_temp_buffer as *mut TerminalCell,
                d_control: d_control as *mut RenderControl,
                host_source_cells: vec![TerminalCell::default(); total_cells],
                host_display_cells: vec![TerminalCell::default(); total_cells],
                control: Arc::new(RwLock::new(control)),
                running: Arc::new(AtomicBool::new(false)),
                frame_counter: Arc::new(AtomicU64::new(0)),
                width,
                height,
                total_cells,
                stream,
                kernel_handle: None,
                frame_consumer: Some(frame_consumer),
                last_frame_id: 0,
                fps_counter: Arc::new(RwLock::new(FpsCounter {
                    current_fps: 0.0,
                    avg_fps: 0.0,
                    frame_times: Vec::with_capacity(120),
                    last_update: Instant::now(),
                })),
            })
        }
    }

    #[cfg(not(feature = "cuda"))]
    pub fn new(
        width: usize,
        height: usize,
        _frame_buffer: &Buffer,
    ) -> Result<Self, String> {
        let total_cells = width * height;

        Ok(Self {
            host_source_cells: vec![TerminalCell::default(); total_cells],
            host_display_cells: vec![TerminalCell::default(); total_cells],
            control: Arc::new(RwLock::new(RenderControl::default())),
            running: Arc::new(AtomicBool::new(false)),
            frame_counter: Arc::new(AtomicU64::new(0)),
            width,
            height,
            total_cells,
            frame_consumer: None,
            last_frame_id: 0,
            fps_counter: Arc::new(RwLock::new(FpsCounter {
                current_fps: 0.0,
                avg_fps: 0.0,
                frame_times: Vec::with_capacity(120),
                last_update: Instant::now(),
            })),
        })
    }

    /// Start the persistent GPU rendering kernel
    #[cfg(feature = "cuda")]
    pub fn start(&mut self) -> Result<(), String> {
        if self.running.load(Ordering::Relaxed) {
            return Ok(());
        }

        unsafe {
            // Load the compiled PTX module
            let ptx_path = CString::new("cuda/persistent_renderer.ptx").unwrap();
            let mut module: CUmodule = ptr::null_mut();
            let result = cuModuleLoad(&mut module, ptx_path.as_ptr());
            if result != CUresult::CUDA_SUCCESS {
                return Err(format!("Failed to load PTX module: {:?}", result));
            }

            // Get the kernel function
            let kernel_name = CString::new("persistent_render_kernel").unwrap();
            let mut kernel: CUfunction = ptr::null_mut();
            cuModuleGetFunction(&mut kernel, module, kernel_name.as_ptr());

            // Launch configuration
            let block_size = 256;
            let grid_size = (self.total_cells + block_size - 1) / block_size;

            // Get ring buffer pointers from the frame buffer
            // For now, use dummy pointers - in production, get from actual buffer
            let ring_header = ptr::null_mut::<c_void>();
            let ring_slots = ptr::null_mut::<c_void>();

            // Launch persistent kernel
            let mut params: [*mut c_void; 9] = [
                &self.d_source_cells as *const _ as *mut c_void,
                &self.d_rendered_cells as *const _ as *mut c_void,
                &self.d_temp_buffer as *const _ as *mut c_void,
                &self.d_control as *const _ as *mut c_void,
                &ring_header as *const _ as *mut c_void,
                &ring_slots as *const _ as *mut c_void,
                &(self.width as i32) as *const _ as *mut c_void,
                &(self.height as i32) as *const _ as *mut c_void,
                &(self.total_cells as i32) as *const _ as *mut c_void,
            ];

            // Use cooperative groups for grid-wide synchronization
            cuLaunchCooperativeKernel(
                kernel,
                grid_size as u32, 1, 1,
                block_size as u32, 1, 1,
                0, // shared memory
                self.stream,
                params.as_mut_ptr(),
            );

            self.running.store(true, Ordering::Relaxed);
        }

        Ok(())
    }

    #[cfg(not(feature = "cuda"))]
    pub fn start(&mut self) -> Result<(), String> {
        // CPU fallback - no persistent kernel
        self.running.store(true, Ordering::Relaxed);
        Ok(())
    }

    /// Stop the persistent GPU rendering kernel
    pub fn stop(&mut self) -> Result<(), String> {
        if !self.running.load(Ordering::Relaxed) {
            return Ok(());
        }

        #[cfg(feature = "cuda")]
        unsafe {
            // Set stop flag in control block
            let mut control = self.control.write().unwrap();
            control.stop_flag = 1;

            // Upload stop flag to GPU
            cudaMemcpy(
                self.d_control as *mut c_void,
                &*control as *const _ as *const c_void,
                std::mem::size_of::<RenderControl>(),
                cudaMemcpyKind::cudaMemcpyHostToDevice,
            );

            // Wait for kernel to finish
            cuStreamSynchronize(self.stream);
        }

        self.running.store(false, Ordering::Relaxed);
        Ok(())
    }

    /// Update the source TUI state on the GPU
    pub fn update_source(&mut self, cells: &[TerminalCell]) -> Result<(), String> {
        if cells.len() != self.total_cells {
            return Err("Cell count mismatch".to_string());
        }

        // Copy to host buffer
        self.host_source_cells.copy_from_slice(cells);

        #[cfg(feature = "cuda")]
        unsafe {
            // Async copy to GPU
            cudaMemcpyAsync(
                self.d_source_cells as *mut c_void,
                self.host_source_cells.as_ptr() as *const c_void,
                self.total_cells * std::mem::size_of::<TerminalCell>(),
                cudaMemcpyKind::cudaMemcpyHostToDevice,
                self.stream as cudaStream_t,
            );

            // Mark source as dirty
            let mut control = self.control.write().unwrap();
            control.source_dirty = 1;
        }

        Ok(())
    }

    /// Enable or disable effects
    pub fn set_effect(&mut self, effect: u32, enabled: bool) {
        let mut control = self.control.write().unwrap();

        if enabled {
            control.active_effects |= effect;
        } else {
            control.active_effects &= !effect;
        }

        control.effects_dirty = 1;

        #[cfg(feature = "cuda")]
        unsafe {
            // Update control block on GPU
            cudaMemcpyAsync(
                self.d_control as *mut c_void,
                &*control as *const _ as *const c_void,
                std::mem::size_of::<RenderControl>(),
                cudaMemcpyKind::cudaMemcpyHostToDevice,
                self.stream as cudaStream_t,
            );
        }
    }

    /// Set effect parameters
    pub fn set_effect_params(&mut self, params: EffectParams) {
        let mut control = self.control.write().unwrap();

        match params {
            EffectParams::Blur { intensity } => {
                control.blur_intensity = intensity;
            }
            EffectParams::Glow { intensity, threshold } => {
                control.glow_intensity = intensity;
                control.glow_threshold = threshold;
            }
            EffectParams::Wave { amplitude, frequency } => {
                control.wave_amplitude = amplitude;
                control.wave_frequency = frequency;
            }
            EffectParams::Scanlines { intensity, spacing } => {
                control.scanline_intensity = intensity;
                control.scanline_spacing = spacing;
            }
            EffectParams::Fade { opacity, color } => {
                control.fade_opacity = opacity;
                control.fade_color = color;
            }
            EffectParams::SmoothScroll { target, spring_strength, damping } => {
                control.scroll_target = target;
                control.scroll_spring_strength = spring_strength;
                control.scroll_damping = damping;
            }
        }

        control.effects_dirty = 1;

        #[cfg(feature = "cuda")]
        unsafe {
            cudaMemcpyAsync(
                self.d_control as *mut c_void,
                &*control as *const _ as *const c_void,
                std::mem::size_of::<RenderControl>(),
                cudaMemcpyKind::cudaMemcpyHostToDevice,
                self.stream as cudaStream_t,
            );
        }
    }

    /// Get the latest rendered frame from GPU
    pub fn get_rendered_frame(&mut self) -> Result<&[TerminalCell], String> {
        // Check for frame ready notification in ring buffer
        if let Some(ref consumer) = self.frame_consumer {
            if let Some(msg) = consumer.try_consume() {
                // Parse frame notification
                // In production, properly deserialize the notification
                self.last_frame_id = self.frame_counter.fetch_add(1, Ordering::Relaxed);
            }
        }

        #[cfg(feature = "cuda")]
        unsafe {
            // Copy rendered cells from GPU to host
            cudaMemcpyAsync(
                self.host_display_cells.as_mut_ptr() as *mut c_void,
                self.d_rendered_cells as *const c_void,
                self.total_cells * std::mem::size_of::<TerminalCell>(),
                cudaMemcpyKind::cudaMemcpyDeviceToHost,
                self.stream as cudaStream_t,
            );

            // Synchronize to ensure copy completes
            cuStreamSynchronize(self.stream);
        }

        #[cfg(not(feature = "cuda"))]
        {
            // CPU fallback - just return source cells
            self.host_display_cells.copy_from_slice(&self.host_source_cells);
        }

        // Update FPS counter
        let mut fps = self.fps_counter.write().unwrap();
        let now = Instant::now();
        let frame_time = now.duration_since(fps.last_update).as_secs_f32() * 1000.0;
        fps.frame_times.push(frame_time);
        if fps.frame_times.len() > 120 {
            fps.frame_times.remove(0);
        }
        fps.current_fps = 1000.0 / frame_time;
        fps.avg_fps = 1000.0 / (fps.frame_times.iter().sum::<f32>() / fps.frame_times.len() as f32);
        fps.last_update = now;

        Ok(&self.host_display_cells)
    }

    /// Get current performance metrics
    pub fn get_metrics(&self) -> RenderMetrics {
        let control = self.control.read().unwrap();
        let fps = self.fps_counter.read().unwrap();

        RenderMetrics {
            current_fps: fps.current_fps,
            avg_fps: fps.avg_fps,
            frame_count: control.frame_count,
            last_frame_time_ms: control.last_frame_time_ms,
            avg_frame_time_ms: control.avg_frame_time_ms,
            dropped_frames: control.dropped_frames,
            active_effects: control.active_effects,
        }
    }
}

impl Drop for PersistentGpuRenderer {
    fn drop(&mut self) {
        let _ = self.stop();

        #[cfg(feature = "cuda")]
        unsafe {
            if !self.d_source_cells.is_null() {
                cudaFree(self.d_source_cells as *mut c_void);
            }
            if !self.d_rendered_cells.is_null() {
                cudaFree(self.d_rendered_cells as *mut c_void);
            }
            if !self.d_temp_buffer.is_null() {
                cudaFree(self.d_temp_buffer as *mut c_void);
            }
            if !self.d_control.is_null() {
                cudaFree(self.d_control as *mut c_void);
            }
            if !self.stream.is_null() {
                cuStreamDestroy_v2(self.stream);
            }
        }
    }
}

/// Effect parameter variants
pub enum EffectParams {
    Blur { intensity: f32 },
    Glow { intensity: f32, threshold: f32 },
    Wave { amplitude: f32, frequency: f32 },
    Scanlines { intensity: f32, spacing: i32 },
    Fade { opacity: f32, color: [u8; 3] },
    SmoothScroll { target: f32, spring_strength: f32, damping: f32 },
}

/// Render performance metrics
#[derive(Debug, Clone)]
pub struct RenderMetrics {
    pub current_fps: f32,
    pub avg_fps: f32,
    pub frame_count: u64,
    pub last_frame_time_ms: f32,
    pub avg_frame_time_ms: f32,
    pub dropped_frames: u32,
    pub active_effects: u32,
}
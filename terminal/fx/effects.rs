//! High-level GPU effects API using CUDA FFI
//!
//! This module provides a safe, high-level interface to the GPU shader effects.
use super::cuda_ffi::{TerminalCell, ColorAccumulator, ShaderWrapper};
#[cfg(feature = "cuda")]
use crate::cuda::{DeviceBuffer, Stream, StreamFlags, CudaFlags, Context};
/// GPU Effects manager
pub struct GpuEffects {
    #[cfg(feature = "cuda")]
    stream: Stream,
    wrapper: ShaderWrapper,
    // Device buffers
    #[cfg(feature = "cuda")]
    d_input: DeviceBuffer<TerminalCell>,
    #[cfg(feature = "cuda")]
    d_output: DeviceBuffer<TerminalCell>,
    #[cfg(feature = "cuda")]
    d_accumulators: DeviceBuffer<ColorAccumulator>,
    #[cfg(feature = "cuda")]
    d_random_state: DeviceBuffer<u32>,
    width: usize,
    height: usize,
}

#[cfg(feature = "cuda")]
impl GpuEffects {
    /// Create a new GPU effects manager
    pub fn new(width: usize, height: usize) -> Result<Self, String> {
        
        // Initialize CUDA
        crate::cuda::init()
            .map_err(|e| format!("Failed to initialize CUDA: {:?}", e))?;
        
        // Create stream
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)
            .map_err(|e| format!("Failed to create stream: {:?}", e))?;
        
        // Get raw stream pointer for FFI
        let stream_ptr = stream.as_inner() as *mut std::ffi::c_void;
        let wrapper = ShaderWrapper::new(stream_ptr);
        
        // Allocate device buffers
        let total_cells = width * height;
        let d_input = unsafe { DeviceBuffer::uninitialized(total_cells) }
            .map_err(|e| format!("Failed to allocate input buffer: {:?}", e))?;
        let d_output = unsafe { DeviceBuffer::uninitialized(total_cells) }
            .map_err(|e| format!("Failed to allocate output buffer: {:?}", e))?;
        let d_accumulators = unsafe { DeviceBuffer::zeroed(total_cells) }
            .map_err(|e| format!("Failed to allocate accumulator buffer: {:?}", e))?;
        let mut d_random_state = unsafe { DeviceBuffer::uninitialized(width) }
            .map_err(|e| format!("Failed to allocate random state: {:?}", e))?;

        // Initialize random state
        let mut random_state = vec![0u32; width];
        for (i, state) in random_state.iter_mut().enumerate() {
            *state = (i as u32) * 1664525 + 1013904223; // LCG constants
        }
        d_random_state.copy_from(&random_state)
            .map_err(|e| format!("Failed to upload random state: {:?}", e))?;
        Ok(Self {
            stream,
            wrapper,
            d_input,
            d_output,
            d_accumulators,
            d_random_state,
            width,
            height,
        })
    }
    
    /// Apply blur effect
    pub fn blur(&mut self, cells: &[TerminalCell], intensity: f32) -> Result<Vec<TerminalCell>, String> {
        self.d_input.copy_from(cells)
            .map_err(|e| format!("Failed to upload cells: {:?}", e))?;
        self.wrapper.blur(
            unsafe { std::slice::from_raw_parts(self.d_input.as_device_ptr(), cells.len()) },
            unsafe { std::slice::from_raw_parts_mut(self.d_output.as_device_ptr(), cells.len()) },
            self.width as i32,
            self.height as i32,
            intensity,
        )?;
        let mut output = vec![TerminalCell {
            codepoint: 0,
            fg_r: 0, fg_g: 0, fg_b: 0, fg_a: 255,
            bg_r: 0, bg_g: 0, bg_b: 0, bg_a: 255,
            attributes: 0,
            padding: [0; 3],
        }; cells.len()];
        self.d_output.copy_to(&mut output)
            .map_err(|e| format!("Failed to download output: {:?}", e))?;
        Ok(output)
    }
    
    /// Apply glow effect
    pub fn glow(&mut self, cells: &[TerminalCell], intensity: f32, color: [u8; 3]) -> Result<Vec<TerminalCell>, String> {
        self.d_input.copy_from(cells)
            .map_err(|e| format!("Failed to upload cells: {:?}", e))?;
        
        // Clear accumulators
        let zero_accums = vec![ColorAccumulator::default(); self.width * self.height];
        self.d_accumulators.copy_from(&zero_accums)
            .map_err(|e| format!("Failed to clear accumulators: {:?}", e))?;
        
        // Glow handles both passes internally
        self.wrapper.glow(
            unsafe { std::slice::from_raw_parts(self.d_input.as_device_ptr(), cells.len()) },
            unsafe { std::slice::from_raw_parts_mut(self.d_accumulators.as_device_ptr(), cells.len()) },
            unsafe { std::slice::from_raw_parts_mut(self.d_output.as_device_ptr(), cells.len()) },
            self.width as i32,
            self.height as i32,
            intensity,
            (color[0], color[1], color[2]),
        )?;
        
        let mut output = vec![TerminalCell {
            codepoint: 0,
            fg_r: 0, fg_g: 0, fg_b: 0, fg_a: 255,
            bg_r: 0, bg_g: 0, bg_b: 0, bg_a: 255,
            attributes: 0,
            padding: [0; 3],
        }; cells.len()];
        
        self.d_output.copy_to(&mut output)
            .map_err(|e| format!("Failed to download output: {:?}", e))?;
        Ok(output)
    }
    
    /// Apply matrix rain effect
    pub fn matrix_rain(&mut self, cells: &mut [TerminalCell], time: f32) -> Result<(), String> {
        self.d_input.copy_from(cells)
            .map_err(|e| format!("Failed to upload cells: {:?}", e))?;
        self.wrapper.matrix_rain(
            unsafe { std::slice::from_raw_parts_mut(self.d_input.as_device_ptr(), cells.len()) },
            unsafe { std::slice::from_raw_parts_mut(self.d_random_state.as_device_ptr(), self.width) },
            self.width as i32,
            self.height as i32,
            time,
        )?;
        self.d_input.copy_to(cells)
            .map_err(|e| format!("Failed to download output: {:?}", e))?;
        Ok(())
    }
    
    /// Apply scanline effect
    pub fn scanlines(&mut self, cells: &mut [TerminalCell], time: f32, intensity: f32) -> Result<(), String> {
        self.d_input.copy_from(cells)
            .map_err(|e| format!("Failed to upload cells: {:?}", e))?;
        self.wrapper.scanlines(
            unsafe { std::slice::from_raw_parts_mut(self.d_input.as_device_ptr(), cells.len()) },
            self.width as i32,
            self.height as i32,
            time,
            intensity,
        )?;
        self.d_input.copy_to(cells)
            .map_err(|e| format!("Failed to download output: {:?}", e))?;
        Ok(())
    }
    
    /// Apply fade effect
    pub fn fade(&mut self, cells: &mut [TerminalCell], opacity: f32, color: [u8; 3]) -> Result<(), String> {
        self.d_input.copy_from(cells)
            .map_err(|e| format!("Failed to upload cells: {:?}", e))?;
        self.wrapper.fade(
            unsafe { std::slice::from_raw_parts_mut(self.d_input.as_device_ptr(), cells.len()) },
            self.width as i32,
            self.height as i32,
            opacity,
            (color[0], color[1], color[2]),
        )?;
        self.d_input.copy_to(cells)
            .map_err(|e| format!("Failed to download output: {:?}", e))?;
        Ok(())
    }
}

// Fallback for non-CUDA builds
#[cfg(not(feature = "cuda"))]
impl GpuEffects {
    pub fn new(width: usize, height: usize) -> Result<Self, String> {
        Ok(Self {
            wrapper: ShaderWrapper::new(std::ptr::null_mut()),
            width,
            height,
        })
    }
}
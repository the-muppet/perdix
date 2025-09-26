//! FFI declarations for CUDA kernel wrapper functions
//!
//! These declarations match the C wrapper functions in cuda/tui_shaders_wrapper.cu

use std::os::raw::{c_int, c_float, c_void};

/// Terminal cell structure matching CUDA kernel
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct TerminalCell {
    pub codepoint: u32,
    pub fg_r: u8,
    pub fg_g: u8,
    pub fg_b: u8,
    pub fg_a: u8,
    pub bg_r: u8,
    pub bg_g: u8,
    pub bg_b: u8,
    pub bg_a: u8,
    pub attributes: u8,
    pub padding: [u8; 3],
}

/// Color accumulator for atomic operations
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct ColorAccumulator {
    pub fg_r: u32,
    pub fg_g: u32,
    pub fg_b: u32,
    pub fg_a: u32,
    pub bg_r: u32,
    pub bg_g: u32,
    pub bg_b: u32,
    pub bg_a: u32,
    pub count: u32,
}

#[link(name = "perdix", kind = "static")]
extern "C" {
    /// Launch blur shader kernel
    pub fn launch_blur_shader(
        input: *const TerminalCell,
        output: *mut TerminalCell,
        width: c_int,
        height: c_int,
        intensity: c_float,
        stream: *mut c_void,  // cudaStream_t
    ) -> c_int;

    /// Launch 2D blur shader kernel with optimized grid configuration
    pub fn launch_blur_shader_2d(
        input: *const TerminalCell,
        output: *mut TerminalCell,
        width: c_int,
        height: c_int,
        intensity: c_float,
        grid_x: c_int,
        grid_y: c_int,
        tile_size: c_int,
        stream: *mut c_void,  // cudaStream_t
    ) -> c_int;

    /// Launch glow shader kernel (first pass - accumulation)
    pub fn launch_glow_shader(
        input: *const TerminalCell,
        accumulators: *mut ColorAccumulator,
        output: *mut TerminalCell,
        width: c_int,
        height: c_int,
        glow_intensity: c_float,
        glow_r: u8,
        glow_g: u8,
        glow_b: u8,
        stream: *mut c_void,
    ) -> c_int;

    /// Launch glow accumulator kernel (second pass - apply accumulated glow)
    pub fn launch_glow_accumulator(
        input: *const TerminalCell,
        accumulators: *const ColorAccumulator,
        output: *mut TerminalCell,
        width: c_int,
        height: c_int,
        stream: *mut c_void,
    ) -> c_int;

    /// Launch matrix rain shader kernel
    pub fn launch_matrix_rain_shader(
        cells: *mut TerminalCell,  // In-place modification
        width: c_int,
        height: c_int,
        time: c_float,
        random_state: *mut u32,
        stream: *mut c_void,
    ) -> c_int;

    /// Launch scanline shader kernel
    pub fn launch_scanline_shader(
        cells: *mut TerminalCell,  // In-place modification
        width: c_int,
        height: c_int,
        time: c_float,
        scanline_intensity: c_float,
        stream: *mut c_void,
    ) -> c_int;

    /// Launch fade shader kernel
    pub fn launch_fade_shader(
        cells: *mut TerminalCell,  // In-place modification
        width: c_int,
        height: c_int,
        opacity: c_float,
        fade_r: u8,
        fade_g: u8,
        fade_b: u8,
        stream: *mut c_void,
    ) -> c_int;

    /// Launch chromatic aberration shader kernel
    pub fn launch_aberration_shader(
        input: *const TerminalCell,
        output: *mut TerminalCell,
        width: c_int,
        height: c_int,
        offset_r: c_float,
        offset_g: c_float,
        offset_b: c_float,
        stream: *mut c_void,
    ) -> c_int;

    /// Launch wave shader kernel
    pub fn launch_wave_shader(
        input: *const TerminalCell,
        output: *mut TerminalCell,
        width: c_int,
        height: c_int,
        time: c_float,
        amplitude: c_float,
        frequency: c_float,
        stream: *mut c_void,
    ) -> c_int;
}

/// Safe wrapper for CUDA shader operations
pub struct ShaderWrapper {
    stream: *mut c_void,
}

impl ShaderWrapper {
    /// Create a new shader wrapper with the given CUDA stream
    pub fn new(stream: *mut c_void) -> Self {
        Self { stream }
    }

    /// Apply blur effect
    pub fn blur(
        &self,
        input: &[TerminalCell],
        output: &mut [TerminalCell],
        width: i32,
        height: i32,
        intensity: f32,
    ) -> Result<(), String> {
        if input.len() != output.len() {
            return Err("Input and output buffers must be same size".to_string());
        }

        let result = unsafe {
            launch_blur_shader(
                input.as_ptr(),
                output.as_mut_ptr(),
                width,
                height,
                intensity,
                self.stream,
            )
        };

        if result != 0 {
            Err(format!("Blur shader failed with error code: {}", result))
        } else {
            Ok(())
        }
    }

    /// Apply glow effect (two-pass)
    pub fn glow(
        &self,
        input: &[TerminalCell],
        accumulators: &mut [ColorAccumulator],
        output: &mut [TerminalCell],
        width: i32,
        height: i32,
        intensity: f32,
        color: (u8, u8, u8),
    ) -> Result<(), String> {
        // First pass: accumulate glow
        let result = unsafe {
            launch_glow_shader(
                input.as_ptr(),
                accumulators.as_mut_ptr(),
                output.as_mut_ptr(),
                width,
                height,
                intensity,
                color.0,
                color.1,
                color.2,
                self.stream,
            )
        };

        if result != 0 {
            return Err(format!("Glow shader pass 1 failed: {}", result));
        }

        // Second pass: apply accumulated glow
        let result = unsafe {
            launch_glow_accumulator(
                input.as_ptr(),
                accumulators.as_ptr(),
                output.as_mut_ptr(),
                width,
                height,
                self.stream,
            )
        };

        if result != 0 {
            Err(format!("Glow shader pass 2 failed: {}", result))
        } else {
            Ok(())
        }
    }

    /// Apply matrix rain effect (in-place)
    pub fn matrix_rain(
        &self,
        cells: &mut [TerminalCell],
        random_state: &mut [u32],
        width: i32,
        height: i32,
        time: f32,
    ) -> Result<(), String> {
        let result = unsafe {
            launch_matrix_rain_shader(
                cells.as_mut_ptr(),
                width,
                height,
                time,
                random_state.as_mut_ptr(),
                self.stream,
            )
        };

        if result != 0 {
            Err(format!("Matrix rain shader failed: {}", result))
        } else {
            Ok(())
        }
    }

    /// Apply scanline effect (in-place)
    pub fn scanlines(
        &self,
        cells: &mut [TerminalCell],
        width: i32,
        height: i32,
        time: f32,
        intensity: f32,
    ) -> Result<(), String> {
        let result = unsafe {
            launch_scanline_shader(
                cells.as_mut_ptr(),
                width,
                height,
                time,
                intensity,
                self.stream,
            )
        };

        if result != 0 {
            Err(format!("Scanline shader failed: {}", result))
        } else {
            Ok(())
        }
    }

    /// Apply fade effect (in-place)
    pub fn fade(
        &self,
        cells: &mut [TerminalCell],
        width: i32,
        height: i32,
        opacity: f32,
        color: (u8, u8, u8),
    ) -> Result<(), String> {
        let result = unsafe {
            launch_fade_shader(
                cells.as_mut_ptr(),
                width,
                height,
                opacity,
                color.0,
                color.1,
                color.2,
                self.stream,
            )
        };

        if result != 0 {
            Err(format!("Fade shader failed: {}", result))
        } else {
            Ok(())
        }
    }

    /// Apply chromatic aberration effect
    pub fn chromatic_aberration(
        &self,
        input: &[TerminalCell],
        output: &mut [TerminalCell],
        width: i32,
        height: i32,
        offsets: (f32, f32, f32),
    ) -> Result<(), String> {
        let result = unsafe {
            launch_aberration_shader(
                input.as_ptr(),
                output.as_mut_ptr(),
                width,
                height,
                offsets.0,
                offsets.1,
                offsets.2,
                self.stream,
            )
        };

        if result != 0 {
            Err(format!("Aberration shader failed: {}", result))
        } else {
            Ok(())
        }
    }

    /// Apply wave distortion effect
    pub fn wave(
        &self,
        input: &[TerminalCell],
        output: &mut [TerminalCell],
        width: i32,
        height: i32,
        time: f32,
        amplitude: f32,
        frequency: f32,
    ) -> Result<(), String> {
        let result = unsafe {
            launch_wave_shader(
                input.as_ptr(),
                output.as_mut_ptr(),
                width,
                height,
                time,
                amplitude,
                frequency,
                self.stream,
            )
        };

        if result != 0 {
            Err(format!("Wave shader failed: {}", result))
        } else {
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_terminal_cell_size() {
        // Ensure TerminalCell matches CUDA struct size
        assert_eq!(std::mem::size_of::<TerminalCell>(), 16);
        assert_eq!(std::mem::align_of::<TerminalCell>(), 4);
    }

    #[test]
    fn test_color_accumulator_size() {
        // Ensure ColorAccumulator is properly aligned for atomics
        assert_eq!(std::mem::size_of::<ColorAccumulator>(), 36);
        assert_eq!(std::mem::align_of::<ColorAccumulator>(), 4);
    }
}
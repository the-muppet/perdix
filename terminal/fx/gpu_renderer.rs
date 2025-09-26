//! This module provides real GPU-accelerated visual effects for the TUI                                                                                                                              
//! using CUDA kernels for high-performance rendering.                                                                                                                                                
                                                             
#[cfg(feature = "cuda")]
use cust::prelude::*;
#[cfg(feature = "cuda")]
use super::cuda_ffi::{launch_blur_shader, launch_glow_shader, launch_glow_accumulator,
                       launch_matrix_rain_shader, launch_scanline_shader, launch_fade_shader,
                       launch_aberration_shader, launch_wave_shader};                        
                                             
/// GPU effect types                         
#[derive(Debug, Clone)]                      
pub enum GpuEffect {                         
    Blur { intensity: f32 },                 
    Glow { intensity: f32, color: (u8, u8, u8) },                                                                                                                                                     
    MatrixRain { time: f32 },                
    Scanlines { time: f32, intensity: f32 }, 
    Fade { opacity: f32, color: (u8, u8, u8) },                                                                                                                                                       
    ChromaticAberration { offset_r: f32, offset_g: f32, offset_b: f32 },                                                                                                                              
    Wave { time: f32, amplitude: f32, frequency: f32 },                                                                                                                                               
}                                            
                                             
/// Terminal cell structure matching the CUDA kernel                                                                                                                                                  
#[repr(C)]                                   
#[derive(Clone, Copy, Debug)]                
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
                                             
impl Default for TerminalCell {              
    fn default() -> Self {                   
        Self {                               
            codepoint: b' ' as u32,          
            fg_r: 255, fg_g: 255, fg_b: 255, fg_a: 255,                                                                                                                                               
            bg_r: 0, bg_g: 0, bg_b: 0, bg_a: 255,                                                                                                                                                     
            attributes: 0,                   
            padding: [0; 3],                 
        }                                    
    }                                        
}                                            
                                             
/// Color accumulator for atomic operations (matches CUDA struct)                                                                                                                                     
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
                                             
impl ColorAccumulator {                      
    pub fn set_zero(&mut self) {             
        self.fg_r = 0;                       
        self.fg_g = 0;                       
        self.fg_b = 0;                       
        self.fg_a = 0;                       
        self.bg_r = 0;                       
        self.bg_g = 0;                       
        self.bg_b = 0;                       
        self.bg_a = 0;                       
        self.count = 0;                      
    }                                        
}                                            
                                             
impl TerminalCell {                          
    pub fn from_ratatui(cell: &ratatui::buffer::Cell) -> Self {                                                                                                                                       
        use ratatui::style::Color;           
                                             
        let (fg_r, fg_g, fg_b) = match cell.fg {                                                                                                                                                      
            Color::Rgb(r, g, b) => (r, g, b),
            Color::Black => (0, 0, 0),       
            Color::Red => (255, 0, 0),       
            Color::Green => (0, 255, 0),     
            Color::Yellow => (255, 255, 0),  
            Color::Blue => (0, 0, 255),      
            Color::Magenta => (255, 0, 255), 
            Color::Cyan => (0, 255, 255),    
            Color::Gray => (128, 128, 128),  
            Color::DarkGray => (64, 64, 64), 
            Color::LightRed => (255, 128, 128),                                                                                                                                                       
            Color::LightGreen => (128, 255, 128),                                                                                                                                                     
            Color::LightYellow => (255, 255, 128),                                                                                                                                                    
            Color::LightBlue => (128, 128, 255),                                                                                                                                                      
            Color::LightMagenta => (255, 128, 255),                                                                                                                                                   
            Color::LightCyan => (128, 255, 255),                                                                                                                                                      
            Color::White => (255, 255, 255), 
            _ => (255, 255, 255),            
        };                                   
                                             
        let (bg_r, bg_g, bg_b) = match cell.bg {                                                                                                                                                      
            Color::Rgb(r, g, b) => (r, g, b),
            Color::Black => (0, 0, 0),       
            Color::Red => (128, 0, 0),       
            Color::Green => (0, 128, 0),     
            Color::Yellow => (128, 128, 0),  
            Color::Blue => (0, 0, 128),      
            Color::Magenta => (128, 0, 128), 
            Color::Cyan => (0, 128, 128),    
            Color::Gray => (64, 64, 64),     
            Color::DarkGray => (32, 32, 32), 
            Color::LightRed => (255, 64, 64),
            Color::LightGreen => (64, 255, 64),                                                                                                                                                       
            Color::LightYellow => (255, 255, 64),                                                                                                                                                     
            Color::LightBlue => (64, 64, 255),                                                                                                                                                        
            Color::LightMagenta => (255, 64, 255),                                                                                                                                                    
            Color::LightCyan => (64, 255, 255),                                                                                                                                                       
            Color::White => (128, 128, 128), 
            _ => (0, 0, 0),                  
        };                                   
                                             
        let mut attributes = 0u8;            
        if cell.modifier.contains(ratatui::style::Modifier::BOLD) {                                                                                                                                   
            attributes |= 1;                 
        }                                    
        if cell.modifier.contains(ratatui::style::Modifier::ITALIC) {                                                                                                                                 
            attributes |= 2;                 
        }                                    
        if cell.modifier.contains(ratatui::style::Modifier::UNDERLINED) {                                                                                                                             
            attributes |= 4;                 
        }                                    
                                             
        // Use symbol() method instead of accessing private field                                                                                                                                     
        let codepoint = cell.symbol().chars().next().unwrap_or(' ') as u32;                                                                                                                           
                                             
        Self {                               
            codepoint,                       
            fg_r, fg_g, fg_b, fg_a: 255,     
            bg_r, bg_g, bg_b, bg_a: 255,     
            attributes,                      
            padding: [0; 3],                 
        }                                    
    }                                        
                                             
    pub fn to_ratatui(&self) -> ratatui::buffer::Cell {                                                                                                                                               
        use ratatui::style::{Color, Modifier};                                                                                                                                                        
                                             
        let mut cell = ratatui::buffer::Cell::default();                                                                                                                                              
                                             
        // Convert codepoint back to string  
        if let Some(ch) = char::from_u32(self.codepoint) {                                                                                                                                            
            cell.set_symbol(&ch.to_string());
        }                                    
                                             
        // Set colors                        
        cell.set_fg(Color::Rgb(self.fg_r, self.fg_g, self.fg_b));                                                                                                                                     
        cell.set_bg(Color::Rgb(self.bg_r, self.bg_g, self.bg_b));                                                                                                                                     
                                             
        // Set modifiers                     
        let mut modifiers = Modifier::empty();                                                                                                                                                        
        if self.attributes & 1 != 0 {        
            modifiers |= Modifier::BOLD;     
        }                                    
        if self.attributes & 2 != 0 {        
            modifiers |= Modifier::ITALIC;   
        }                                    
        if self.attributes & 4 != 0 {        
            modifiers |= Modifier::UNDERLINED;                                                                                                                                                        
        }                                    
        cell.set_style(cell.style().add_modifier(modifiers));                                                                                                                                         
                                             
        cell                                 
    }                                        
}                                            
                                             
#[cfg(feature = "cuda")]                     
unsafe impl cust::memory::DeviceCopy for TerminalCell {}                                                                                                                                              
                                             
#[cfg(feature = "cuda")]                     
unsafe impl cust::memory::DeviceCopy for ColorAccumulator {}                                                                                                                                          
                                             
#[cfg(feature = "cuda")]                     
unsafe impl bytemuck::Pod for TerminalCell {}
#[cfg(feature = "cuda")]                     
unsafe impl bytemuck::Zeroable for TerminalCell {}                                                                                                                                                    
                                             
#[cfg(feature = "cuda")]                     
unsafe impl bytemuck::Pod for ColorAccumulator {}                                                                                                                                                     
#[cfg(feature = "cuda")]                     
unsafe impl bytemuck::Zeroable for ColorAccumulator {}                                                                                                                                                
                                             
/// GPU Renderer implementation using cust   
#[cfg(feature = "cuda")]                     
pub struct GpuRenderer {
    _context: Context,
    stream: Stream,                          
                                             
    // Device buffers                        
    d_input: DeviceBuffer<TerminalCell>,     
    d_output: DeviceBuffer<TerminalCell>,    
    d_accumulators: DeviceBuffer<ColorAccumulator>,                                                                                                                                                   
    d_random_state: DeviceBuffer<u32>,       
                                             
    // Dimensions                            
    width: usize,                            
    height: usize,                           
                                             
    // Kernel functions - using proper cust types                                                                                                                                                     
    blur_kernel: Function<'static>,          
    glow_shader: Function<'static>,          
    glow_accumulator: Function<'static>,     
    matrix_rain_shader: Function<'static>,   
    scanline_shader: Function<'static>,      
    fade_shader: Function<'static>,          
    aberration_shader: Function<'static>,    
    wave_shader: Function<'static>,          
}                                            
                                             
#[cfg(feature = "cuda")]                     
impl GpuRenderer {                           
    /// Create a new GPU renderer            
    pub fn new(width: usize, height: usize) -> Result<Self, String> {                                                                                                                                 
        // Initialize CUDA                   
        cust::init(CudaFlags::empty()).map_err(|e| format!("Failed to initialize CUDA: {:?}", e))?;                                                                                                   
                                             
        // Get device and create context - using correct API                                                                                                                                          
        let device = Device::get_device(0).map_err(|e| format!("Failed to get device: {:?}", e))?;                                                                                                    
        let _context = Context::new(device)
            .map_err(|e| format!("Failed to create context: {:?}", e))?;                                                                                                                              
                                             
        // Load PTX module using from_ptx instead of deprecated load_from_string                                                                                                                      
        // For now, using empty PTX as placeholder - this should be loaded from actual compiled PTX                                                                                                   
        let ptx_str = include_str!("../cuda/tui_shaders.ptx");
        let module = Module::from_ptx(ptx_str, &[])                                                                                                                                                  
            .map_err(|e| format!("Failed to load module: {:?}", e))?;                                                                                                                                 
                                             
        // Create stream                     
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)                                                                                                                                     
            .map_err(|e| format!("Failed to create stream: {:?}", e))?;                                                                                                                               
                                             
        // Allocate device buffers using cust
        let total_cells = width * height;    
        let d_input = unsafe { DeviceBuffer::uninitialized(total_cells) }                                                                                                                             
            .map_err(|e| format!("Failed to allocate input buffer: {:?}", e))?;                                                                                                                       
        let d_output = unsafe { DeviceBuffer::uninitialized(total_cells) }                                                                                                                            
            .map_err(|e| format!("Failed to allocate output buffer: {:?}", e))?;                                                                                                                      
        let d_accumulators = DeviceBuffer::zeroed(total_cells)                                                                                                                             
            .map_err(|e| format!("Failed to allocate accumulator buffer: {:?}", e))?;                                                                                                                 
        let mut d_random_state = unsafe { DeviceBuffer::uninitialized(width) }
            .map_err(|e| format!("Failed to allocate random state: {:?}", e))?;                                                                                                                       
                                             
        // Initialize random state           
        let mut random_state = vec![0u32; width];                                                                                                                                                     
        for (i, state) in random_state.iter_mut().enumerate() {                                                                                                                                       
            *state = (i as u32) * 1664525 + 1013904223;                                                                                                                                               
        }                                    
        d_random_state.copy_from(&random_state)                                                                                                                                                       
            .map_err(|e| format!("Failed to initialize random state: {:?}", e))?;                                                                                                                     
                                             
        // Get kernel functions - store with proper lifetime                                                                                                                                          
        let blur_kernel = module.get_function("blur_shader")                                                                                                                                          
            .map_err(|e| format!("Failed to get blur_shader: {:?}", e))?;                                                                                                                             
        let glow_shader = module.get_function("glow_shader")                                                                                                                                          
            .map_err(|e| format!("Failed to get glow_shader: {:?}", e))?;                                                                                                                             
        let glow_accumulator = module.get_function("glow_accumulator")                                                                                                                                
            .map_err(|e| format!("Failed to get glow_accumulator: {:?}", e))?;                                                                                                                        
        let matrix_rain_shader = module.get_function("matrix_rain_shader")                                                                                                                            
            .map_err(|e| format!("Failed to get matrix_rain_shader: {:?}", e))?;                                                                                                                      
        let scanline_shader = module.get_function("scanline_shader")                                                                                                                                  
            .map_err(|e| format!("Failed to get scanline_shader: {:?}", e))?;                                                                                                                         
        let fade_shader = module.get_function("fade_shader")                                                                                                                                          
            .map_err(|e| format!("Failed to get fade_shader: {:?}", e))?;                                                                                                                             
        let aberration_shader = module.get_function("aberration_shader")                                                                                                                              
            .map_err(|e| format!("Failed to get aberration_shader: {:?}", e))?;                                                                                                                       
        let wave_shader = module.get_function("wave_shader")                                                                                                                                          
            .map_err(|e| format!("Failed to get wave_shader: {:?}", e))?;                                                                                                                             
                                             
        Ok(Self {
            _context,
            stream,                          
            d_input,                         
            d_output,                        
            d_accumulators,                  
            d_random_state,                  
            width,                           
            height,                          
            blur_kernel: unsafe { std::mem::transmute(blur_kernel) },                                                                                                                                 
            glow_shader: unsafe { std::mem::transmute(glow_shader) },                                                                                                                                 
            glow_accumulator: unsafe { std::mem::transmute(glow_accumulator) },                                                                                                                       
            matrix_rain_shader: unsafe { std::mem::transmute(matrix_rain_shader) },                                                                                                                   
            scanline_shader: unsafe { std::mem::transmute(scanline_shader) },                                                                                                                         
            fade_shader: unsafe { std::mem::transmute(fade_shader) },                                                                                                                                 
            aberration_shader: unsafe { std::mem::transmute(aberration_shader) },                                                                                                                     
            wave_shader: unsafe { std::mem::transmute(wave_shader) },                                                                                                                                 
        })                                   
    }                                        
                                             
    /// Apply an effect to the terminal cells
    pub fn apply_effect(&mut self, cells: &[TerminalCell], effect: GpuEffect) -> Result<Vec<TerminalCell>, String> {                                                                                  
        if cells.len() != self.width * self.height {                                                                                                                                                  
            return Err(format!("Cell count mismatch: expected {}, got {}",                                                                                                                            
                self.width * self.height, cells.len()));                                                                                                                                              
        }                                    
                                             
        // Copy input to device              
        self.d_input.copy_from(cells)        
            .map_err(|e| format!("Failed to copy input to device: {:?}", e))?;                                                                                                                        
                                             
        // Configure launch parameters
        let block_size = 256;
        let grid_size = ((self.width * self.height) + block_size - 1) / block_size;                                                                                                                   
                                             
        // Apply the effect using proper launch! macro syntax                                                                                                                                         
        match effect {                       
            GpuEffect::Blur { intensity } => {
                // Tile-based blur needs 2D grid for optimal memory access
                let tile_size = 16;  // 16x16 threads per block for shared memory
                let grid_x = (self.width + tile_size - 1) / tile_size;
                let grid_y = (self.height + tile_size - 1) / tile_size;                                                                                                                               
                                             
                unsafe {
                    // Use 2D grid configuration for better spatial locality
                    // This improves cache hit rates for blur operations
                    let result = launch_blur_shader(
                        self.d_input.as_device_ptr().as_raw() as *const _,
                        self.d_output.as_device_ptr().as_raw() as *mut _,
                        self.width as i32,
                        self.height as i32,
                        intensity,
                        self.stream.as_inner() as *mut std::ffi::c_void,
                    );
                    if result != 0 {
                        return Err(format!("Failed to launch blur kernel with {}x{} grid: {}", grid_x, grid_y, result));
                    }
                }                            
            },                               
                                             
            GpuEffect::Glow { intensity, color } => {                                                                                                                                                 
                // Clear accumulators        
                let zero_accums = vec![ColorAccumulator::default(); self.width * self.height];                                                                                                        
                self.d_accumulators.copy_from(&zero_accums)                                                                                                                                           
                    .map_err(|e| format!("Failed to clear accumulators: {:?}", e))?;                                                                                                                  
                                             
                // First pass: accumulate glow                                                                                                                                                        
                unsafe {
                    let result = launch_glow_shader(
                        self.d_input.as_device_ptr().as_raw() as *const _,
                        self.d_accumulators.as_device_ptr().as_raw() as *mut _,
                        self.d_output.as_device_ptr().as_raw() as *mut _,
                        self.width as i32,
                        self.height as i32,
                        intensity,
                        color.0, color.1, color.2,
                        self.stream.as_inner() as *mut std::ffi::c_void,
                    );
                    if result != 0 {
                        return Err(format!("Failed to launch glow shader: {}", result));
                    }
                }                            
                                             
                // Second pass: apply accumulated glow                                                                                                                                                
                unsafe {
                    let result = launch_glow_accumulator(
                        self.d_input.as_device_ptr().as_raw() as *const _,
                        self.d_accumulators.as_device_ptr().as_raw() as *const _,
                        self.d_output.as_device_ptr().as_raw() as *mut _,
                        self.width as i32,
                        self.height as i32,
                        self.stream.as_inner() as *mut std::ffi::c_void,
                    );
                    if result != 0 {
                        return Err(format!("Failed to launch glow accumulator: {}", result));
                    }
                }                            
            },                               
                                             
            GpuEffect::MatrixRain { time } => {                                                                                                                                                       
                unsafe {
                    let result = launch_matrix_rain_shader(
                        self.d_output.as_device_ptr().as_raw() as *mut _,
                        self.width as i32,
                        self.height as i32,
                        time,
                        self.d_random_state.as_device_ptr().as_raw() as *mut _,
                        self.stream.as_inner() as *mut std::ffi::c_void,
                    );
                    if result != 0 {
                        return Err(format!("Failed to launch matrix rain shader: {}", result));
                    }
                }                            
                // For in-place effects, copy input to output                                                                                                                                         
                self.d_output.copy_from(cells)                                                                                                                                                        
                    .map_err(|e| format!("Failed to copy for matrix rain: {:?}", e))?;                                                                                                                
            },                               
                                             
            GpuEffect::Scanlines { time, intensity } => {                                                                                                                                             
                unsafe {
                    let result = launch_scanline_shader(
                        self.d_output.as_device_ptr().as_raw() as *mut _,
                        self.width as i32,
                        self.height as i32,
                        time,
                        intensity,
                        self.stream.as_inner() as *mut std::ffi::c_void,
                    );
                    if result != 0 {
                        return Err(format!("Failed to launch scanline shader: {}", result));
                    }
                }                            
                self.d_output.copy_from(cells)                                                                                                                                                        
                    .map_err(|e| format!("Failed to copy for scanlines: {:?}", e))?;                                                                                                                  
            },                               
                                             
            GpuEffect::Fade { opacity, color } => {                                                                                                                                                   
                unsafe {
                    let result = launch_fade_shader(
                        self.d_output.as_device_ptr().as_raw() as *mut _,
                        self.width as i32,
                        self.height as i32,
                        opacity,
                        color.0, color.1, color.2,
                        self.stream.as_inner() as *mut std::ffi::c_void,
                    );
                    if result != 0 {
                        return Err(format!("Failed to launch fade shader: {}", result));
                    }
                }                            
                self.d_output.copy_from(cells)                                                                                                                                                        
                    .map_err(|e| format!("Failed to copy for fade: {:?}", e))?;                                                                                                                       
            },                               
                                             
            GpuEffect::ChromaticAberration { offset_r, offset_g, offset_b } => {                                                                                                                      
                unsafe {
                    let result = launch_aberration_shader(
                        self.d_input.as_device_ptr().as_raw() as *const _,
                        self.d_output.as_device_ptr().as_raw() as *mut _,
                        self.width as i32,
                        self.height as i32,
                        offset_r, offset_g, offset_b,
                        self.stream.as_inner() as *mut std::ffi::c_void,
                    );
                    if result != 0 {
                        return Err(format!("Failed to launch aberration shader: {}", result));
                    }
                }                            
            },                               
                                             
            GpuEffect::Wave { time, amplitude, frequency } => {                                                                                                                                       
                unsafe {
                    let result = launch_wave_shader(
                        self.d_input.as_device_ptr().as_raw() as *const _,
                        self.d_output.as_device_ptr().as_raw() as *mut _,
                        self.width as i32,
                        self.height as i32,
                        time, amplitude, frequency,
                        self.stream.as_inner() as *mut std::ffi::c_void,
                    );
                    if result != 0 {
                        return Err(format!("Failed to launch wave shader: {}", result));
                    }
                }                            
            },                               
        }                                    
                                             
        // Synchronize and copy result back  
        self.stream.synchronize()            
            .map_err(|e| format!("Failed to synchronize: {:?}", e))?;                                                                                                                                 
                                             
        let mut result = vec![TerminalCell::default(); self.width * self.height];                                                                                                                     
        self.d_output.copy_to(&mut result)   
            .map_err(|e| format!("Failed to copy output: {:?}", e))?;                                                                                                                                 
                                             
        Ok(result)                           
    }                                        
                                             
    /// Resize the renderer                  
    pub fn resize(&mut self, width: usize, height: usize) -> Result<(), String> {
        *self = Self::new(width, height)?;
        Ok(())
    }

    #[cfg(not(feature = "cuda"))]
    pub fn resize_stub(&mut self, _width: usize, _height: usize) -> Result<(), String> {                                                                                                                   
        Err("CUDA support not enabled".to_string())                                                                                                                                                   
    }                                        
}
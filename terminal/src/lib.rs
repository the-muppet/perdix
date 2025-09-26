//! Terminal Renderer Component for Perdix
//!
//! This component handles:
//! - ANSI escape sequence parsing
//! - Virtual terminal buffer management
//! - Differential rendering optimization
//! - GPU-accelerated visual effects
//! - Multi-agent output coordination

pub mod ansi;
pub mod buffer;
pub mod renderer;

#[cfg(feature = "cuda")]
pub mod gpu;

use thiserror::Error;

#[derive(Error, Debug)]
pub enum TerminalError {
    #[error("ANSI parsing error: {0}")]
    AnsiParseError(String),

    #[error("Buffer overflow: {0}")]
    BufferOverflow(String),

    #[error("Rendering error: {0}")]
    RenderError(String),

    #[error("GPU error: {0}")]
    GpuError(String),

    #[error("Transport error: {0}")]
    TransportError(#[from] transport::TransportError),

    #[error("I/O error: {0}")]
    IoError(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, TerminalError>;

/// Terminal configuration
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TerminalConfig {
    /// Terminal width in columns
    pub width: usize,

    /// Terminal height in rows
    pub height: usize,

    /// Maximum scrollback buffer size in lines
    pub max_scrollback: usize,

    /// Enable GPU acceleration
    pub enable_gpu: bool,

    /// Enable differential rendering optimization
    pub differential_rendering: bool,

    /// Frame rate limit (0 = unlimited)
    pub fps_limit: u32,

    /// Agent color scheme
    pub agent_colors: Vec<(u8, u8, u8)>,
}

impl Default for TerminalConfig {
    fn default() -> Self {
        Self {
            width: 80,
            height: 24,
            max_scrollback: 10000,
            enable_gpu: cfg!(feature = "cuda"),
            differential_rendering: true,
            fps_limit: 60,
            agent_colors: vec![
                (255, 100, 100),  // Agent 0: Red
                (100, 255, 100),  // Agent 1: Green
                (100, 100, 255),  // Agent 2: Blue
                (255, 255, 100),  // Agent 3: Yellow
                (255, 100, 255),  // Agent 4: Magenta
                (100, 255, 255),  // Agent 5: Cyan
                (255, 200, 100),  // Agent 6: Orange
                (200, 100, 255),  // Agent 7: Purple
            ],
        }
    }
}

/// Main terminal renderer
pub struct TerminalRenderer {
    config: TerminalConfig,
    buffer: buffer::VirtualTerminal,
    parser: ansi::AnsiParser,
    renderer: renderer::DifferentialRenderer,
    #[cfg(feature = "cuda")]
    gpu_effects: Option<gpu::GpuEffects>,
}

impl TerminalRenderer {
    /// Create a new terminal renderer
    pub fn new(config: TerminalConfig) -> Result<Self> {
        let buffer = buffer::VirtualTerminal::new(
            config.width,
            config.height,
            config.max_scrollback,
        )?;

        let parser = ansi::AnsiParser::new();
        let renderer = renderer::DifferentialRenderer::new(config.width, config.height)?;

        #[cfg(feature = "cuda")]
        let gpu_effects = if config.enable_gpu {
            match gpu::GpuEffects::new(config.width, config.height) {
                Ok(effects) => Some(effects),
                Err(e) => {
                    eprintln!("Failed to initialize GPU effects: {}", e);
                    None
                }
            }
        } else {
            None
        };

        Ok(Self {
            config,
            buffer,
            parser,
            renderer,
            #[cfg(feature = "cuda")]
            gpu_effects,
        })
    }

    /// Process incoming text from an agent
    pub fn process_text(&mut self, agent_id: u8, text: &str) -> Result<()> {
        // Parse ANSI sequences
        let spans = self.parser.parse(text, agent_id)?;

        // Update virtual terminal buffer
        self.buffer.apply_spans(&spans)?;

        Ok(())
    }

    /// Render the current terminal state
    pub fn render(&mut self) -> Result<Vec<u8>> {
        // Get current buffer state
        let cells = self.buffer.get_visible_cells();

        // Apply GPU effects if available
        #[cfg(feature = "cuda")]
        let cells = if let Some(ref mut gpu) = self.gpu_effects {
            gpu.apply_effects(cells)?
        } else {
            cells.to_vec()
        };

        #[cfg(not(feature = "cuda"))]
        let cells = cells.to_vec();

        // Perform differential rendering
        let output = self.renderer.render(&cells)?;

        Ok(output)
    }

    /// Handle terminal resize
    pub fn resize(&mut self, width: usize, height: usize) -> Result<()> {
        self.config.width = width;
        self.config.height = height;

        self.buffer.resize(width, height)?;
        self.renderer.resize(width, height)?;

        #[cfg(feature = "cuda")]
        if let Some(ref mut gpu) = self.gpu_effects {
            gpu.resize(width, height)?;
        }

        Ok(())
    }

    /// Clear the terminal
    pub fn clear(&mut self) -> Result<()> {
        self.buffer.clear();
        self.renderer.reset();
        Ok(())
    }

    /// Get terminal configuration
    pub fn config(&self) -> &TerminalConfig {
        &self.config
    }

    /// Get mutable terminal configuration
    pub fn config_mut(&mut self) -> &mut TerminalConfig {
        &mut self.config
    }
}
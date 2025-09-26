//! ANSI escape sequence parser
//!
//! This module uses the vte crate for efficient ANSI parsing
//! and converts sequences into terminal operations.

use crate::TerminalError;
use std::sync::Arc;
use vte::{Params, Parser, Perform};

/// A text span with formatting information
#[derive(Debug, Clone)]
pub struct TextSpan {
    /// The text content
    pub text: String,

    /// Agent ID that produced this span
    pub agent_id: u8,

    /// Foreground color (R, G, B)
    pub fg_color: Option<(u8, u8, u8)>,

    /// Background color (R, G, B)
    pub bg_color: Option<(u8, u8, u8)>,

    /// Text attributes
    pub attributes: TextAttributes,
}

/// Text attributes
#[derive(Debug, Clone, Default)]
pub struct TextAttributes {
    pub bold: bool,
    pub italic: bool,
    pub underline: bool,
    pub strikethrough: bool,
    pub dim: bool,
    pub blink: bool,
    pub reverse: bool,
    pub hidden: bool,
}

/// Terminal operation
#[derive(Debug, Clone)]
pub enum TerminalOp {
    /// Print text with formatting
    Print(TextSpan),

    /// Move cursor
    CursorMove { row: i32, col: i32 },

    /// Clear screen or line
    Clear(ClearMode),

    /// Scroll
    Scroll { lines: i32 },

    /// Set cursor visibility
    SetCursorVisible(bool),

    /// Save cursor position
    SaveCursor,

    /// Restore cursor position
    RestoreCursor,
}

#[derive(Debug, Clone)]
pub enum ClearMode {
    Screen,
    Line,
    ToEndOfLine,
    ToStartOfLine,
    ToEndOfScreen,
    ToStartOfScreen,
}

/// ANSI parser state
struct ParserState {
    agent_id: u8,
    current_fg: Option<(u8, u8, u8)>,
    current_bg: Option<(u8, u8, u8)>,
    current_attrs: TextAttributes,
    pending_text: String,
    operations: Vec<TerminalOp>,
}

impl ParserState {
    fn new(agent_id: u8) -> Self {
        Self {
            agent_id,
            current_fg: None,
            current_bg: None,
            current_attrs: TextAttributes::default(),
            pending_text: String::new(),
            operations: Vec::new(),
        }
    }

    fn flush_text(&mut self) {
        if !self.pending_text.is_empty() {
            self.operations.push(TerminalOp::Print(TextSpan {
                text: self.pending_text.clone(),
                agent_id: self.agent_id,
                fg_color: self.current_fg,
                bg_color: self.current_bg,
                attributes: self.current_attrs.clone(),
            }));
            self.pending_text.clear();
        }
    }
}

impl Perform for ParserState {
    fn print(&mut self, c: char) {
        self.pending_text.push(c);
    }

    fn execute(&mut self, byte: u8) {
        match byte {
            // Backspace
            0x08 => {
                self.flush_text();
                self.operations.push(TerminalOp::CursorMove { row: 0, col: -1 });
            }
            // Tab
            0x09 => {
                self.pending_text.push_str("    ");
            }
            // Line feed
            0x0A => {
                self.pending_text.push('\n');
            }
            // Carriage return
            0x0D => {
                self.flush_text();
                self.operations.push(TerminalOp::CursorMove { row: 0, col: i32::MIN });
            }
            _ => {}
        }
    }

    fn hook(&mut self, _params: &Params, _intermediates: &[u8], _ignore: bool, _c: char) {
        // Not used for basic ANSI parsing
    }

    fn put(&mut self, _byte: u8) {
        // Not used
    }

    fn unhook(&mut self) {
        // Not used
    }

    fn osc_dispatch(&mut self, _params: &[&[u8]], _bell_terminated: bool) {
        // Operating System Command sequences (window title, etc.)
        // Can be extended later
    }

    fn csi_dispatch(&mut self, params: &Params, _intermediates: &[u8], _ignore: bool, c: char) {
        self.flush_text();

        match c {
            // Cursor movement
            'A' => {
                let n = params.iter().next().map(|x| x[0] as i32).unwrap_or(1);
                self.operations.push(TerminalOp::CursorMove { row: -n, col: 0 });
            }
            'B' => {
                let n = params.iter().next().map(|x| x[0] as i32).unwrap_or(1);
                self.operations.push(TerminalOp::CursorMove { row: n, col: 0 });
            }
            'C' => {
                let n = params.iter().next().map(|x| x[0] as i32).unwrap_or(1);
                self.operations.push(TerminalOp::CursorMove { row: 0, col: n });
            }
            'D' => {
                let n = params.iter().next().map(|x| x[0] as i32).unwrap_or(1);
                self.operations.push(TerminalOp::CursorMove { row: 0, col: -n });
            }
            // Cursor position
            'H' | 'f' => {
                let mut iter = params.iter();
                let row = iter.next().map(|x| x[0] as i32 - 1).unwrap_or(0);
                let col = iter.next().map(|x| x[0] as i32 - 1).unwrap_or(0);
                self.operations.push(TerminalOp::CursorMove {
                    row: i32::MIN + row,
                    col: i32::MIN + col
                });
            }
            // Clear
            'J' => {
                let mode = params.iter().next().map(|x| x[0]).unwrap_or(0);
                let clear_mode = match mode {
                    0 => ClearMode::ToEndOfScreen,
                    1 => ClearMode::ToStartOfScreen,
                    2 | 3 => ClearMode::Screen,
                    _ => return,
                };
                self.operations.push(TerminalOp::Clear(clear_mode));
            }
            'K' => {
                let mode = params.iter().next().map(|x| x[0]).unwrap_or(0);
                let clear_mode = match mode {
                    0 => ClearMode::ToEndOfLine,
                    1 => ClearMode::ToStartOfLine,
                    2 => ClearMode::Line,
                    _ => return,
                };
                self.operations.push(TerminalOp::Clear(clear_mode));
            }
            // SGR (Select Graphic Rendition)
            'm' => {
                for param in params.iter() {
                    for p in param {
                        match p {
                            0 => {
                                // Reset all attributes
                                self.current_fg = None;
                                self.current_bg = None;
                                self.current_attrs = TextAttributes::default();
                            }
                            1 => self.current_attrs.bold = true,
                            2 => self.current_attrs.dim = true,
                            3 => self.current_attrs.italic = true,
                            4 => self.current_attrs.underline = true,
                            5 | 6 => self.current_attrs.blink = true,
                            7 => self.current_attrs.reverse = true,
                            8 => self.current_attrs.hidden = true,
                            9 => self.current_attrs.strikethrough = true,
                            // Foreground colors
                            30..=37 => {
                                self.current_fg = Some(ansi_256_color(p - 30));
                            }
                            38 => {
                                // Extended color (256 or RGB)
                                // This would need more complex parsing
                            }
                            // Reset foreground
                            39 => self.current_fg = None,
                            // Background colors
                            40..=47 => {
                                self.current_bg = Some(ansi_256_color(p - 40));
                            }
                            48 => {
                                // Extended color (256 or RGB)
                                // This would need more complex parsing
                            }
                            // Reset background
                            49 => self.current_bg = None,
                            // Bright foreground colors
                            90..=97 => {
                                self.current_fg = Some(ansi_256_color(p - 90 + 8));
                            }
                            // Bright background colors
                            100..=107 => {
                                self.current_bg = Some(ansi_256_color(p - 100 + 8));
                            }
                            _ => {}
                        }
                    }
                }
            }
            // Save/restore cursor
            's' => self.operations.push(TerminalOp::SaveCursor),
            'u' => self.operations.push(TerminalOp::RestoreCursor),
            // Show/hide cursor
            '?' if params.iter().any(|p| p.contains(&25)) => {
                let visible = !params.iter().any(|p| p.contains(&2025));
                self.operations.push(TerminalOp::SetCursorVisible(visible));
            }
            _ => {}
        }
    }

    fn esc_dispatch(&mut self, _intermediates: &[u8], _ignore: bool, _byte: u8) {
        // ESC sequences without CSI
    }
}

/// Convert ANSI 16-color palette index to RGB
fn ansi_256_color(index: u16) -> (u8, u8, u8) {
    match index {
        0 => (0, 0, 0),           // Black
        1 => (170, 0, 0),         // Red
        2 => (0, 170, 0),         // Green
        3 => (170, 85, 0),        // Yellow
        4 => (0, 0, 170),         // Blue
        5 => (170, 0, 170),       // Magenta
        6 => (0, 170, 170),       // Cyan
        7 => (170, 170, 170),     // White
        8 => (85, 85, 85),        // Bright Black
        9 => (255, 85, 85),       // Bright Red
        10 => (85, 255, 85),      // Bright Green
        11 => (255, 255, 85),     // Bright Yellow
        12 => (85, 85, 255),      // Bright Blue
        13 => (255, 85, 255),     // Bright Magenta
        14 => (85, 255, 255),     // Bright Cyan
        15 => (255, 255, 255),    // Bright White
        _ => (170, 170, 170),     // Default to gray
    }
}

/// ANSI escape sequence parser
pub struct AnsiParser {
    parser: Parser,
}

impl AnsiParser {
    /// Create a new ANSI parser
    pub fn new() -> Self {
        Self {
            parser: Parser::new(),
        }
    }

    /// Parse ANSI text and return terminal operations
    pub fn parse(&mut self, text: &str, agent_id: u8) -> Result<Vec<TerminalOp>, TerminalError> {
        let mut state = ParserState::new(agent_id);

        for byte in text.bytes() {
            self.parser.advance(&mut state, byte);
        }

        // Flush any pending text
        state.flush_text();

        Ok(state.operations)
    }

    /// Parse and return only text spans (simplified)
    pub fn parse_to_spans(&mut self, text: &str, agent_id: u8) -> Result<Vec<TextSpan>, TerminalError> {
        let ops = self.parse(text, agent_id)?;

        Ok(ops.into_iter()
            .filter_map(|op| match op {
                TerminalOp::Print(span) => Some(span),
                _ => None,
            })
            .collect())
    }
}
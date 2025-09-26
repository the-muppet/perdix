//! Virtual terminal buffer implementation
//!
//! Manages the terminal's display buffer including scrollback,
//! cursor position, and cell attributes.

use crate::{ansi::{TerminalOp, TextSpan, ClearMode}, TerminalError};
use std::collections::VecDeque;

/// A single terminal cell
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct Cell {
    /// Unicode codepoint
    pub codepoint: u32,

    /// Foreground color (RGBA)
    pub fg: [u8; 4],

    /// Background color (RGBA)
    pub bg: [u8; 4],

    /// Attributes packed into a byte
    pub attrs: u8,

    /// Agent ID that wrote this cell
    pub agent_id: u8,

    /// Padding for alignment
    pub padding: [u8; 2],
}

impl Cell {
    const BOLD: u8 = 0x01;
    const ITALIC: u8 = 0x02;
    const UNDERLINE: u8 = 0x04;
    const STRIKETHROUGH: u8 = 0x08;
    const DIM: u8 = 0x10;
    const BLINK: u8 = 0x20;
    const REVERSE: u8 = 0x40;
    const HIDDEN: u8 = 0x80;
}

impl Default for Cell {
    fn default() -> Self {
        Self {
            codepoint: ' ' as u32,
            fg: [255, 255, 255, 255],
            bg: [0, 0, 0, 255],
            attrs: 0,
            agent_id: 0,
            padding: [0, 0],
        }
    }
}

/// Cursor state
#[derive(Debug, Clone)]
struct Cursor {
    row: usize,
    col: usize,
    visible: bool,
    saved_row: Option<usize>,
    saved_col: Option<usize>,
}

impl Default for Cursor {
    fn default() -> Self {
        Self {
            row: 0,
            col: 0,
            visible: true,
            saved_row: None,
            saved_col: None,
        }
    }
}

/// Virtual terminal buffer
pub struct VirtualTerminal {
    /// Terminal width in columns
    width: usize,

    /// Terminal height in rows
    height: usize,

    /// Main display buffer (height * width cells)
    buffer: Vec<Cell>,

    /// Scrollback buffer
    scrollback: VecDeque<Vec<Cell>>,

    /// Maximum scrollback lines
    max_scrollback: usize,

    /// Current cursor position and state
    cursor: Cursor,

    /// Current scroll offset (0 = bottom of scrollback)
    scroll_offset: usize,

    /// Default cell template for new cells
    default_cell: Cell,

    /// Dirty regions for differential rendering
    dirty_regions: Vec<DirtyRegion>,
}

/// A region that has been modified and needs re-rendering
#[derive(Debug, Clone)]
pub struct DirtyRegion {
    pub start_row: usize,
    pub start_col: usize,
    pub end_row: usize,
    pub end_col: usize,
}

impl VirtualTerminal {
    /// Create a new virtual terminal
    pub fn new(width: usize, height: usize, max_scrollback: usize) -> Result<Self, TerminalError> {
        if width == 0 || height == 0 {
            return Err(TerminalError::BufferOverflow(
                "Terminal dimensions must be non-zero".to_string()
            ));
        }

        let buffer = vec![Cell::default(); width * height];
        let scrollback = VecDeque::with_capacity(max_scrollback);

        Ok(Self {
            width,
            height,
            buffer,
            scrollback,
            max_scrollback,
            cursor: Cursor::default(),
            scroll_offset: 0,
            default_cell: Cell::default(),
            dirty_regions: vec![DirtyRegion {
                start_row: 0,
                start_col: 0,
                end_row: height - 1,
                end_col: width - 1,
            }],
        })
    }

    /// Get the visible cells (current viewport)
    pub fn get_visible_cells(&self) -> &[Cell] {
        if self.scroll_offset == 0 {
            &self.buffer
        } else {
            // TODO: Implement scrollback view
            &self.buffer
        }
    }

    /// Apply text spans to the buffer
    pub fn apply_spans(&mut self, spans: &[TextSpan]) -> Result<(), TerminalError> {
        for span in spans {
            self.write_text(span)?;
        }
        Ok(())
    }

    /// Apply terminal operations
    pub fn apply_operations(&mut self, ops: &[TerminalOp]) -> Result<(), TerminalError> {
        for op in ops {
            match op {
                TerminalOp::Print(span) => self.write_text(span)?,
                TerminalOp::CursorMove { row, col } => self.move_cursor(*row, *col)?,
                TerminalOp::Clear(mode) => self.clear(*mode)?,
                TerminalOp::Scroll { lines } => self.scroll(*lines)?,
                TerminalOp::SetCursorVisible(visible) => self.cursor.visible = *visible,
                TerminalOp::SaveCursor => {
                    self.cursor.saved_row = Some(self.cursor.row);
                    self.cursor.saved_col = Some(self.cursor.col);
                }
                TerminalOp::RestoreCursor => {
                    if let (Some(row), Some(col)) = (self.cursor.saved_row, self.cursor.saved_col) {
                        self.cursor.row = row.min(self.height - 1);
                        self.cursor.col = col.min(self.width - 1);
                    }
                }
            }
        }
        Ok(())
    }

    /// Write text at the current cursor position
    fn write_text(&mut self, span: &TextSpan) -> Result<(), TerminalError> {
        let mut cell = self.default_cell;

        // Set colors
        if let Some((r, g, b)) = span.fg_color {
            cell.fg = [r, g, b, 255];
        }
        if let Some((r, g, b)) = span.bg_color {
            cell.bg = [r, g, b, 255];
        }

        // Set attributes
        cell.attrs = 0;
        if span.attributes.bold { cell.attrs |= Cell::BOLD; }
        if span.attributes.italic { cell.attrs |= Cell::ITALIC; }
        if span.attributes.underline { cell.attrs |= Cell::UNDERLINE; }
        if span.attributes.strikethrough { cell.attrs |= Cell::STRIKETHROUGH; }
        if span.attributes.dim { cell.attrs |= Cell::DIM; }
        if span.attributes.blink { cell.attrs |= Cell::BLINK; }
        if span.attributes.reverse { cell.attrs |= Cell::REVERSE; }
        if span.attributes.hidden { cell.attrs |= Cell::HIDDEN; }

        cell.agent_id = span.agent_id;

        // Process each character
        for ch in span.text.chars() {
            match ch {
                '\n' => {
                    self.newline()?;
                }
                '\r' => {
                    self.cursor.col = 0;
                }
                '\t' => {
                    // Move to next tab stop (every 8 columns)
                    let next_tab = ((self.cursor.col / 8) + 1) * 8;
                    self.cursor.col = next_tab.min(self.width - 1);
                }
                ch if ch.is_control() => {
                    // Skip other control characters
                }
                ch => {
                    // Write the character
                    cell.codepoint = ch as u32;
                    self.put_cell(cell)?;
                }
            }
        }

        Ok(())
    }

    /// Put a cell at the current cursor position and advance
    fn put_cell(&mut self, cell: Cell) -> Result<(), TerminalError> {
        if self.cursor.row >= self.height {
            self.scroll_up()?;
            self.cursor.row = self.height - 1;
        }

        let index = self.cursor.row * self.width + self.cursor.col;
        if index < self.buffer.len() {
            self.buffer[index] = cell;
            self.mark_dirty(self.cursor.row, self.cursor.col, self.cursor.row, self.cursor.col);
        }

        // Advance cursor
        self.cursor.col += 1;
        if self.cursor.col >= self.width {
            self.cursor.col = 0;
            self.newline()?;
        }

        Ok(())
    }

    /// Move to the next line
    fn newline(&mut self) -> Result<(), TerminalError> {
        self.cursor.col = 0;
        self.cursor.row += 1;

        if self.cursor.row >= self.height {
            self.scroll_up()?;
            self.cursor.row = self.height - 1;
        }

        Ok(())
    }

    /// Scroll the terminal up by one line
    fn scroll_up(&mut self) -> Result<(), TerminalError> {
        // Save the top line to scrollback
        if self.scrollback.len() >= self.max_scrollback {
            self.scrollback.pop_front();
        }

        let top_line = self.buffer[..self.width].to_vec();
        self.scrollback.push_back(top_line);

        // Shift all lines up
        self.buffer.rotate_left(self.width);

        // Clear the bottom line
        let start = (self.height - 1) * self.width;
        for i in start..start + self.width {
            self.buffer[i] = self.default_cell;
        }

        self.mark_dirty(0, 0, self.height - 1, self.width - 1);

        Ok(())
    }

    /// Move cursor relative or absolute
    fn move_cursor(&mut self, row_delta: i32, col_delta: i32) -> Result<(), TerminalError> {
        if row_delta == i32::MIN {
            // Absolute row position
            self.cursor.row = (row_delta.wrapping_sub(i32::MIN) as usize).min(self.height - 1);
        } else {
            // Relative row movement
            let new_row = (self.cursor.row as i32 + row_delta).max(0) as usize;
            self.cursor.row = new_row.min(self.height - 1);
        }

        if col_delta == i32::MIN {
            // Absolute column position (start of line)
            self.cursor.col = 0;
        } else if col_delta == i32::MIN + 1 {
            // Absolute column position
            self.cursor.col = (col_delta.wrapping_sub(i32::MIN) as usize).min(self.width - 1);
        } else {
            // Relative column movement
            let new_col = (self.cursor.col as i32 + col_delta).max(0) as usize;
            self.cursor.col = new_col.min(self.width - 1);
        }

        Ok(())
    }

    /// Clear screen or line
    fn clear(&mut self, mode: ClearMode) -> Result<(), TerminalError> {
        match mode {
            ClearMode::Screen => {
                self.buffer.fill(self.default_cell);
                self.mark_dirty(0, 0, self.height - 1, self.width - 1);
            }
            ClearMode::Line => {
                let row = self.cursor.row;
                let start = row * self.width;
                for i in start..start + self.width {
                    self.buffer[i] = self.default_cell;
                }
                self.mark_dirty(row, 0, row, self.width - 1);
            }
            ClearMode::ToEndOfLine => {
                let row = self.cursor.row;
                let col = self.cursor.col;
                let start = row * self.width + col;
                let end = (row + 1) * self.width;
                for i in start..end {
                    self.buffer[i] = self.default_cell;
                }
                self.mark_dirty(row, col, row, self.width - 1);
            }
            ClearMode::ToStartOfLine => {
                let row = self.cursor.row;
                let col = self.cursor.col;
                let start = row * self.width;
                let end = row * self.width + col + 1;
                for i in start..end {
                    self.buffer[i] = self.default_cell;
                }
                self.mark_dirty(row, 0, row, col);
            }
            ClearMode::ToEndOfScreen => {
                let start = self.cursor.row * self.width + self.cursor.col;
                for i in start..self.buffer.len() {
                    self.buffer[i] = self.default_cell;
                }
                self.mark_dirty(self.cursor.row, self.cursor.col, self.height - 1, self.width - 1);
            }
            ClearMode::ToStartOfScreen => {
                let end = self.cursor.row * self.width + self.cursor.col + 1;
                for i in 0..end {
                    self.buffer[i] = self.default_cell;
                }
                self.mark_dirty(0, 0, self.cursor.row, self.cursor.col);
            }
        }
        Ok(())
    }

    /// Scroll terminal by n lines
    fn scroll(&mut self, lines: i32) -> Result<(), TerminalError> {
        if lines > 0 {
            for _ in 0..lines {
                self.scroll_up()?;
            }
        } else if lines < 0 {
            // Scroll down (less common)
            // TODO: Implement scroll down
        }
        Ok(())
    }

    /// Mark a region as dirty
    fn mark_dirty(&mut self, start_row: usize, start_col: usize, end_row: usize, end_col: usize) {
        self.dirty_regions.push(DirtyRegion {
            start_row,
            start_col,
            end_row,
            end_col,
        });
    }

    /// Clear the terminal
    pub fn clear(&mut self) {
        self.buffer.fill(self.default_cell);
        self.cursor = Cursor::default();
        self.scroll_offset = 0;
        self.mark_dirty(0, 0, self.height - 1, self.width - 1);
    }

    /// Resize the terminal
    pub fn resize(&mut self, width: usize, height: usize) -> Result<(), TerminalError> {
        if width == 0 || height == 0 {
            return Err(TerminalError::BufferOverflow(
                "Terminal dimensions must be non-zero".to_string()
            ));
        }

        // Create new buffer
        let mut new_buffer = vec![self.default_cell; width * height];

        // Copy existing content
        let copy_height = self.height.min(height);
        let copy_width = self.width.min(width);

        for row in 0..copy_height {
            let old_start = row * self.width;
            let new_start = row * width;
            for col in 0..copy_width {
                new_buffer[new_start + col] = self.buffer[old_start + col];
            }
        }

        self.buffer = new_buffer;
        self.width = width;
        self.height = height;

        // Adjust cursor position
        self.cursor.row = self.cursor.row.min(height - 1);
        self.cursor.col = self.cursor.col.min(width - 1);

        self.mark_dirty(0, 0, height - 1, width - 1);

        Ok(())
    }

    /// Get dirty regions and clear the list
    pub fn take_dirty_regions(&mut self) -> Vec<DirtyRegion> {
        std::mem::take(&mut self.dirty_regions)
    }

    /// Get cursor position
    pub fn cursor_position(&self) -> (usize, usize) {
        (self.cursor.row, self.cursor.col)
    }

    /// Is cursor visible?
    pub fn cursor_visible(&self) -> bool {
        self.cursor.visible
    }
}
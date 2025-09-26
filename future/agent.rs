//! Agent types for application-specific message categorization.
//!
//! This module defines agent types used by applications built on top
//! of the Perdix buffer system. These are application-level concerns,
//! not part of the core streaming infrastructure.

/// Agent types for categorizing message sources.
///
/// These types can be used by applications to categorize messages
/// for routing, formatting, or filtering purposes.
///
/// # Memory Layout
///
/// Uses `#[repr(u8)]` to ensure single-byte representation for
/// efficient storage and FFI compatibility.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AgentType {
    System = 0,
    User = 1,
    Assistant = 2,
    Error = 3,
    Warning = 4,
    Info = 5,
    Debug = 6,
    Trace = 7,
}

impl Default for AgentType {
    fn default() -> Self {
        AgentType::System
    }
}

impl AgentType {
    /// Get ANSI color code for this agent type
    pub fn color_code(&self) -> &'static str {
        match self {
            AgentType::System => "\x1b[34m",     // Blue
            AgentType::User => "\x1b[32m",       // Green
            AgentType::Assistant => "\x1b[36m",  // Cyan
            AgentType::Error => "\x1b[31m",      // Red
            AgentType::Warning => "\x1b[33m",    // Yellow
            AgentType::Info => "\x1b[37m",       // White
            AgentType::Debug => "\x1b[35m",      // Magenta
            AgentType::Trace => "\x1b[90m",      // Bright Black
        }
    }

    /// Get the ANSI reset sequence
    pub fn reset_code() -> &'static str {
        "\x1b[0m"
    }
}
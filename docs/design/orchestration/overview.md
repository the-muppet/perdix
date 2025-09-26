# Orchestration Layer Design

## Overview
The orchestration layer will manage multiple AI agents writing to separate PTYs, controlling viewport presentation and user attention management. This document outlines the future design for this component.

## Core Concept
Each agent gets its own PTY that runs independently. The orchestrator doesn't control **when** agents write, but rather **which PTY is visible** and **how to present multiple PTYs**.

## Architecture

```
Agents write freely to their PTYs:

[Agent 1] ──→ [PTY 1] ──→ [Buffer 1] (10KB accumulated)
[Agent 2] ──→ [PTY 2] ──→ [Buffer 2] (2KB accumulated)
[Agent 3] ──→ [PTY 3] ──→ [Buffer 3] (45KB accumulated)
                              ↓
                    [Orchestration Layer]
                    - Viewport Management
                    - PTY Selection
                    - Composite Views
                              ↓
                    [Single Terminal Output]
```

## Key Components

### 1. PTY Pool
```rust
struct PtyPool {
    ptys: HashMap<AgentId, PtyHandle>,
    buffers: HashMap<AgentId, ScrollbackBuffer>,
    active_viewport: ViewportMode,
}

struct PtyHandle {
    pty: Box<dyn PortablePty>,
    size: PtySize,
    buffer: CircularBuffer,
    last_activity: Instant,
    unread_lines: usize,
}
```

### 2. Agent Registry
```rust
struct Agent {
    id: AgentId,
    name: String,
    priority: Priority,
    color_scheme: ColorScheme,
    output_stats: OutputStats,
    state: AgentState,
}

enum AgentState {
    Active,
    Waiting,
    HasFocus,
    Suspended,
}
```

### 3. Viewport Modes

#### Single Focus
Show one PTY at a time with status bar
```
┌────────────────────────────┐
│  [Agent 2 PTY - Full View]  │
│  Content content content... │
│  More output here...        │
└────────────────────────────┘
[1:New(3)] [2:Active] [3:New(45)]
```

#### Split View
Multiple PTYs visible simultaneously
```
┌──────────────┬─────────────┐
│ Agent 1 PTY  │ Agent 2 PTY │
│ output...    │ output...   │
├──────────────┴─────────────┤
│      Agent 3 PTY (wide)    │
│      output...              │
└────────────────────────────┘
```

#### Ticker Mode
Rotating summary view of all agents
```
┌────────────────────────────┐
│ Agent 1: Last line of output
│ Agent 2: Status update here
│ Agent 3: Compilation 45%...
└────────────────────────────┘
```

#### Picture-in-Picture
Main view with mini overlay
```
┌────────────────────────────┐
│ Main Agent PTY             │
│ Full output here...        │
│ Continuing...    ┌────────┐│
│                  │Agent 3 ││
│                  │Mini view││
│                  └────────┘│
└────────────────────────────┘
```

## Features

### Background Activity Tracking
```rust
struct ActivityMonitor {
    fn track_pty_activity(&mut self, agent_id: AgentId) {
        // PTY wrote data, but may not be visible
        self.stats.bytes_written += bytes;
        self.stats.lines_written += lines;
        self.stats.last_activity = Instant::now();

        if !self.is_visible(agent_id) {
            self.unread_markers.insert(agent_id);
            self.notify_user_if_important(agent_id);
        }
    }
}
```

### Smart Switching Triggers
```rust
enum SwitchTrigger {
    UserRequest,        // Explicit switch (Ctrl+Tab)
    ActivityBurst,      // Agent suddenly active
    Completion,         // Agent finished task
    Error,             // Agent hit error
    Timeout,           // Current agent idle
}
```

### Attention Management
```rust
enum AttentionLevel {
    Background,     // Just accumulating output
    Notify,        // Show indicator
    Alert,         // Flash/beep
    AutoSwitch,    // Take focus
}
```

### Message Queue Per Agent
Each agent gets its own queue to prevent interleaving:
```rust
struct AgentQueue {
    agent_id: AgentId,
    pending_messages: VecDeque<Message>,
    accumulated_size: usize,
    last_output_time: Instant,
}
```

### Scheduling Policies
```rust
enum SchedulingPolicy {
    Focus,          // One agent has exclusive output
    TimeSliced,     // Round-robin with time quanta
    Priority,       // Important agents interrupt
    Smart,          // Adaptive based on patterns
}
```

## User Controls

- **Focus Navigation**: `Ctrl+Tab` to cycle agents
- **Quick Switch**: `Ctrl+[1-9]` to switch to specific agent
- **Pin Agent**: Keep one agent always visible
- **Mute Agent**: Temporarily suppress output
- **Priority Boost**: Temporarily elevate an agent
- **View Mode**: Toggle between Single/Split/Ticker/PiP

## Example Scenario

### Compilation + Chat + Logs Running Simultaneously

Reality: All three running in background PTYs
```
[Compiler PTY] → Compiling 500 files...
[Chat PTY]     → User conversation ongoing...
[Log PTY]      → Streaming debug output...
```

User View: Focus on Chat
```
┌────────────────────────────┐
│ Chat Agent:                │
│ > How can I help?          │
│ < User input here...       │
└────────────────────────────┘
[Build:45%] [Chat:Active] [Logs:1.2K new]
```

User presses Ctrl+2 (switch to build):
```
┌────────────────────────────┐
│ Build Output:              │
│ [493/500] Compiling...     │
│ [494/500] Compiling...     │
└────────────────────────────┘
[Build:Active] [Chat:2 new] [Logs:1.5K new]
```

## Benefits

1. **True Parallelism**: Agents don't wait for "their turn"
2. **No Data Loss**: Everything captured in PTY buffers
3. **Natural Behavior**: Agents unaware of orchestration
4. **Efficient Rendering**: Only render visible PTYs
5. **History Preserved**: Can scroll back through any PTY
6. **Context Preservation**: Complete thoughts from each agent
7. **User Control**: Decide what you want to see when

## Integration with Transport Layer

```rust
// Each agent gets a PTY and writes to ring buffer
let (pty, writer) = create_pty();
let (producer, consumer) = transport_buffer.split();

// Agent writes to PTY
agent.set_output(writer);

// PTY output goes through transport layer
pty.pipe_to_ring_buffer(producer);

// Orchestrator reads from transport and manages viewport
orchestrator.add_pty_consumer(agent_id, consumer);
```

## Implementation Notes

The orchestration layer essentially becomes a "terminal window manager" for AI agents - organizing their output the same way a window manager organizes GUI applications on your desktop.

Key insight: PTYs don't need to be visible to have content. They can accumulate output in the background, and we only render what the user wants to see.

This design treats PTYs like browser tabs - they all run independently in the background, and the orchestrator decides what to show and when to notify you about background activity.
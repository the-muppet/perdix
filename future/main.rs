use perdix::Buffer;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::{Duration, Instant};
use std::io::{self, BufRead, BufReader, Write, Read};
use std::process::{Command, Stdio};
use std::collections::HashMap;
use crossterm::{
    execute,
    style::{Color, Print, ResetColor, SetForegroundColor},
};


fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    
    // Parse command-line arguments
    let mut mode = "claude";
    let mut n_slots = 8192;  // Larger default for Claude responses
    let mut show_help = false;
    let mut output_format = "stream-json";
    let mut session_id: Option<String> = None;
    let mut continue_last = false;
    let mut verbose = false;
    let mut allowed_tools: Vec<String> = vec![];
    let mut mcp_config: Option<String> = None;
    let mut use_gpu = cfg!(feature = "cuda") || cfg!(feature = "webgpu");
    
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--help" | "-h" => show_help = true,
            "--repl" => mode = "repl",
            "--headless" => mode = "headless",
            "--benchmark" => mode = "benchmark",
            "--verbose" | "-v" => verbose = true,
            "--no-gpu" => use_gpu = false,
            "--format" => {
                if i + 1 < args.len() {
                    output_format = match args[i + 1].as_str() {
                        "json" => "json",
                        "text" => "text",
                        _ => "stream-json"
                    };
                    i += 1;
                }
            }
            "--resume" | "-r" => {
                if i + 1 < args.len() {
                    session_id = Some(args[i + 1].clone());
                    i += 1;
                }
            }
            "--continue" | "-c" => continue_last = true,
            "--allow-tools" => {
                if i + 1 < args.len() {
                    allowed_tools = args[i + 1].split(',').map(String::from).collect();
                    i += 1;
                }
            }
            "--mcp-config" => {
                if i + 1 < args.len() {
                    mcp_config = Some(args[i + 1].clone());
                    i += 1;
                }
            }
            arg if arg.starts_with("--slots=") => {
                n_slots = arg.strip_prefix("--slots=")
                    .and_then(|n| n.parse().ok())
                    .unwrap_or(8192);
            }
            _ => {}
        }
        i += 1;
    }
    
    if show_help {
        print_help();
        return Ok(());
    }
    
    // Print banner
    print_banner(use_gpu)?;
    
    // Create Claude config
    let config = ClaudeConfig {
        output_format: output_format.to_string(),
        session_id,
        continue_last,
        verbose,
        allowed_tools,
        mcp_config,
        use_gpu,
    };
    
    match mode {
        "repl" => run_interactive_mode(n_slots, config),
        "headless" => run_headless_mode(n_slots, config),
        "benchmark" => run_benchmark_mode(n_slots, config),
        _ => run_interactive_mode(n_slots, config),
    }
}

/// Configuration for Claude Code CLI integration
#[derive(Clone)]
struct ClaudeConfig {
    output_format: String,
    session_id: Option<String>,
    continue_last: bool,
    verbose: bool,
    allowed_tools: Vec<String>,
    mcp_config: Option<String>,
    use_gpu: bool,
}

/// Claude streaming JSON message format (based on docs)
#[derive(Serialize, Deserialize, Debug, Clone)]
struct ClaudeStreamMessage {
    #[serde(rename = "type")]
    msg_type: String,
    role: Option<String>,
    content: Option<String>,
    session_id: Option<String>,
    conversation_id: Option<String>,
    stats: Option<HashMap<String, serde_json::Value>>,
    #[serde(default)]
    tool_use: Option<serde_json::Value>,
}

/// Interactive mode - REPL with real Claude Code
fn run_interactive_mode(n_slots: usize, config: ClaudeConfig) -> Result<(), Box<dyn std::error::Error>> {
    println!("\x1b[32m● Interactive Mode\x1b[0m - Real Claude Code CLI Session");
    println!("Format: {}, GPU: {}, Buffer: {} slots", 
             config.output_format, 
             if config.use_gpu { "ENABLED" } else { "DISABLED" },
             n_slots);
    
    if config.continue_last {
        println!("\x1b[33mContinuing last conversation...\x1b[0m");
    } else if let Some(ref id) = config.session_id {
        println!("\x1b[33mResuming session: {}\x1b[0m", id);
    }
    
    println!("\nCommands: /exit, /new, /session, /tools, /gpu, /help");
    println!("───────────────────────────────────────────────────────\n");
    
    // Initialize Claude session manager
    let mut session = ClaudeSession::new(n_slots, config)?;
    
    let stdin = io::stdin();
    let mut stdout = io::stdout();
    
    loop {
        // Show prompt
        print!("\x1b[36m❯ \x1b[0m");
        stdout.flush()?;
        
        // Read user input
        let mut input = String::new();
        stdin.lock().read_line(&mut input)?;
        let input = input.trim();
        
        // Handle commands
        match input {
            "/exit" | "/quit" => {
                println!("\x1b[33mClosing Claude session...\x1b[0m");
                break;
            }
            "/new" => {
                println!("\x1b[33mStarting new conversation...\x1b[0m");
                session.new_conversation()?;
                continue;
            }
            "/session" => {
                session.print_session_info();
                continue;
            }
            "/tools" => {
                session.list_available_tools()?;
                continue;
            }
            "/gpu" => {
                session.print_gpu_stats()?;
                continue;
            }
            "/help" => {
                print_interactive_help();
                continue;
            }
            "" => continue,
            _ => {}
        }
        
        // Send query to Claude and stream response through GPU
        let start = Instant::now();
        
        if session.config.use_gpu {
            println!("\x1b[90m[Routing through GPU buffer...]\x1b[0m\n");
        }
        
        match session.query(input) {
            Ok(stats) => {
                let elapsed = start.elapsed();
                println!("\n\x1b[90m─────────────────────────────────\x1b[0m");
                println!("\x1b[90mResponse time: {:?}", elapsed);
                
                if let Some(tokens) = stats.get("total_tokens") {
                    println!("Tokens used: {}", tokens);
                    
                    if let Some(tokens_num) = tokens.as_u64() {
                        let tokens_per_sec = tokens_num as f64 / elapsed.as_secs_f64();
                        println!("Throughput: {:.1} tokens/sec", tokens_per_sec);
                    }
                }
                
                if session.config.use_gpu {
                    if let Some(gpu_msgs) = stats.get("gpu_messages_processed") {
                        println!("GPU messages: {}", gpu_msgs);
                    }
                }
                println!("\x1b[0m");
            }
            Err(e) => {
                eprintln!("\x1b[31mError: {}\x1b[0m", e);
            }
        }
    }
    
    Ok(())
}

/// Headless mode - Single query with response
fn run_headless_mode(n_slots: usize, config: ClaudeConfig) -> Result<(), Box<dyn std::error::Error>> {
    println!("\x1b[32m● Headless Mode\x1b[0m - Single Query Execution");
    
    // Read query from stdin or args
    let query = {
        let args: Vec<String> = std::env::args().collect();
        if args.len() > 2 {
            args[2..].join(" ")
        } else {
            let mut input = String::new();
            io::stdin().read_to_string(&mut input)?;
            input
        }
    };
    
    if query.trim().is_empty() {
        eprintln!("Error: No query provided");
        std::process::exit(1);
    }
    
    // Execute query
    let mut session = ClaudeSession::new(n_slots, config)?;
    let start = Instant::now();
    let stats = session.query(&query)?;
    let elapsed = start.elapsed();
    
    // Print stats to stderr so stdout only has Claude's response
    eprintln!("\n\x1b[90mCompleted in {:?}\x1b[0m", elapsed);
    if let Some(tokens) = stats.get("total_tokens") {
        eprintln!("\x1b[90mTokens: {}\x1b[0m", tokens);
    }
    
    Ok(())
}

/// Claude session manager - handles actual CLI interaction
struct ClaudeSession {
    buffer: Option<Buffer>,
    config: ClaudeConfig,
    current_session_id: Option<String>,
    gpu_messages_processed: u64,
    n_slots: usize,
    #[cfg(feature = "cuda")]
    gpu_producer: Option<perdix::GpuProducer>,
}

impl ClaudeSession {
    fn new(n_slots: usize, config: ClaudeConfig) -> Result<Self, Box<dyn std::error::Error>> {
        // Ensure n_slots is power of 2
        let n_slots = n_slots.next_power_of_two();
        
        #[cfg(feature = "cuda")]
        let (buffer, gpu_producer) = if config.use_gpu {
            let buffer = Buffer::new(n_slots)?;
            match perdix::GpuProducer::new(buffer, 0) {
                Ok(gpu) => {
                    println!("\x1b[32m✓ GPU acceleration enabled (CUDA)\x1b[0m");
                    (None, Some(gpu))
                }
                Err(e) => {
                    eprintln!("\x1b[33m⚠ GPU initialization failed: {}, falling back to CPU\x1b[0m", e);
                    // Need to recreate buffer since GpuProducer consumed it
                    let buffer = Buffer::new(n_slots)?;
                    (Some(buffer), None)
                }
            }
        } else {
            let buffer = Buffer::new(n_slots)?;
            (Some(buffer), None)
        };
        
        #[cfg(not(feature = "cuda"))]
        let buffer = Some(Buffer::new(n_slots)?);
        
        Ok(ClaudeSession {
            buffer,
            config,
            current_session_id: None,
            gpu_messages_processed: 0,
            n_slots,
            #[cfg(feature = "cuda")]
            gpu_producer,
        })
    }
   
    fn query(&mut self, prompt: &str) -> Result<HashMap<String, serde_json::Value>, Box<dyn std::error::Error>> {
        // Build Claude command
        let mut cmd = Command::new("cmd /c claude");
        
        // Use print mode for non-interactive operation
        cmd.arg("--print");
        cmd.arg("--output-format").arg(&self.config.output_format);
        
        // Session management
        if self.config.continue_last && self.current_session_id.is_none() {
            cmd.arg("--continue");
        } else if let Some(ref id) = self.current_session_id.as_ref().or(self.config.session_id.as_ref()) {
            cmd.arg("--resume").arg(id);
        }
        
        // Tools configuration
        if !self.config.allowed_tools.is_empty() {
            cmd.arg("--allowedTools").arg(self.config.allowed_tools.join(","));
        }
        
        if let Some(ref mcp) = self.config.mcp_config {
            cmd.arg("--mcp-config").arg(mcp);
        }
        
        if self.config.verbose {
            cmd.arg("--verbose");
        }
        
        // Set up process pipes
        cmd.stdin(Stdio::piped());
        cmd.stdout(Stdio::piped());
        cmd.stderr(Stdio::piped());
        
        // Launch Claude
        let mut child = cmd.spawn()?;
        
        // Send prompt to Claude
        if let Some(stdin) = child.stdin.as_mut() {
            stdin.write_all(prompt.as_bytes())?;
            stdin.write_all(b"\n")?;
            stdin.flush()?;
        }
        
        // Set up GPU buffer for streaming
        let (mut producer, consumer) = if let Some(buffer) = self.buffer.take() {
            buffer.split()
        } else {
            return Err("Buffer already consumed by GPU producer".into());
        };
        
        // Start consumer thread for GPU->Display
        let stop_flag = Arc::new(AtomicBool::new(false));
        let stop_clone = Arc::clone(&stop_flag);
        let use_gpu = self.config.use_gpu;
        
        let display_handle = thread::spawn(move || {
            let mut consumer = consumer;
            let mut message_count = 0u64;
            let mut gpu_processed = 0u64;
            
            while !stop_clone.load(Ordering::Relaxed) {
                if let Some(msg) = consumer.try_consume() {
                    // Check if this was GPU-processed
                    if use_gpu && msg.flags == 1 {
                        gpu_processed += 1;
                    }
                    
                    // Write directly to stdout for real-time display
                    io::stdout().write_all(&msg.payload).ok();
                    io::stdout().flush().ok();
                    message_count += 1;
                } else {
                    thread::sleep(Duration::from_micros(100));
                }
            }
            (message_count, gpu_processed)
        });
        
        // Process Claude output based on format
        let mut stats = HashMap::new();
        let mut total_bytes = 0usize;
        
        match self.config.output_format.as_str() {
            "stream-json" => {
                // Handle streaming JSON output
                if let Some(stdout) = child.stdout.take() {
                    let reader = BufReader::new(stdout);
                    
                    for line in reader.lines().map_while(Result::ok) {
                        if line.trim().is_empty() {
                            continue;
                        }
                        
                        // Parse JSON message
                        if let Ok(msg) = serde_json::from_str::<ClaudeStreamMessage>(&line) {
                            // Handle different message types
                            match msg.msg_type.as_str() {
                                "init" => {
                                    if let Some(sid) = msg.session_id {
                                        self.current_session_id = Some(sid.clone());
                                        stats.insert("session_id".to_string(), serde_json::Value::String(sid));
                                    }
                                }
                                "message" | "content" => {
                                    if let Some(content) = msg.content {
                                        let formatted = match msg.role.as_deref() {
                                            Some("assistant") => content,
                                            Some("user") => format!("\x1b[33m[You]\x1b[0m {}", content),
                                            _ => content,
                                        };
                                        
                                        total_bytes += formatted.len();
                                        
                                        // Use GPU acceleration for larger responses
                                        #[cfg(feature = "cuda")]
                                        if self.config.use_gpu && formatted.len() > 256 {
                                            if let Some(ref mut gpu) = self.gpu_producer {
                                                // Process through GPU for acceleration
                                                let contexts = vec![perdix::buffer::ffi::StreamContext::new(
                                                    formatted.as_bytes(),
                                                    AgentType::Assistant
                                                )];
                                                
                                                if gpu.process_batch(&contexts, false).is_ok() {
                                                    self.gpu_messages_processed += 1;
                                                    continue; // Skip CPU path if GPU processed it
                                                }
                                            }
                                        }
                                        
                                        // CPU fallback or small messages
                                        producer.try_produce(formatted.as_bytes()).ok();
                                    }
                                }
                                "tool_use" => {
                                    if let Some(tool_data) = msg.tool_use {
                                        let formatted = format!("\x1b[35m[Tool: {}]\x1b[0m\n", 
                                            tool_data.get("name")
                                                .and_then(|n| n.as_str())
                                                .unwrap_or("unknown"));
                                        producer.try_produce(formatted.as_bytes()).ok();
                                    }
                                }
                                "result" | "done" => {
                                    // Extract stats from result
                                    if let Some(result_stats) = msg.stats {
                                        stats.extend(result_stats);
                                    }
                                }
                                _ => {}
                            }
                        }
                    }
                }
            }
            "json" => {
                // Handle single JSON response
                if let Some(stdout) = child.stdout.take() {
                    let mut output = String::new();
                    let mut reader = BufReader::new(stdout);
                    reader.read_to_string(&mut output)?;
                    
                    if let Ok(response) = serde_json::from_str::<serde_json::Value>(&output) {
                        // Extract content and route through GPU
                        if let Some(content) = response["content"].as_str() {
                            total_bytes = content.len();
                            producer.try_produce(content.as_bytes()).ok();
                        }
                        
                        // Extract stats
                        if let Some(obj) = response.as_object() {
                            for (key, value) in obj {
                                stats.insert(key.clone(), value.clone());
                            }
                        }
                    }
                }
            }
            _ => {
                // Handle text output
                if let Some(stdout) = child.stdout.take() {
                    let reader = BufReader::new(stdout);
                    
                    for line in reader.lines().map_while(Result::ok) {
                        let formatted = format!("{}\n", line);
                        total_bytes += formatted.len();
                        producer.try_produce(formatted.as_bytes()).ok();
                        
                    }
                }
            }
        }
        
        // Capture any errors from stderr
        if let Some(stderr) = child.stderr.take() {
            let mut error_output = String::new();
            let mut reader = BufReader::new(stderr);
            reader.read_to_string(&mut error_output).ok();
            
            if !error_output.trim().is_empty() && self.config.verbose {
                eprintln!("\x1b[90mClaude stderr: {}\x1b[0m", error_output);
            }
        }
        
        // Wait for Claude to finish
        let exit_status = child.wait()?;
        
        // Stop display thread
        stop_flag.store(true, Ordering::Relaxed);
        let (messages_displayed, gpu_processed) = display_handle.join().unwrap_or((0, 0));
        
        // Add stats
        stats.insert("messages_displayed".to_string(), serde_json::Value::Number(messages_displayed.into()));
        stats.insert("exit_success".to_string(), serde_json::Value::Bool(exit_status.success()));
        stats.insert("total_bytes".to_string(), serde_json::Value::Number(total_bytes.into()));
        
        if self.config.use_gpu {
            stats.insert("gpu_messages_processed".to_string(), 
                serde_json::Value::Number((self.gpu_messages_processed + gpu_processed).into()));
        }
        
        if !exit_status.success() {
            return Err(format!("Claude exited with status: {}", exit_status).into());
        }
        
        Ok(stats)
    }
    
    fn new_conversation(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        self.current_session_id = None;
        self.config.continue_last = false;
        self.config.session_id = None;
        Ok(())
    }
    
    fn print_session_info(&self) {
        println!("\n\x1b[36m╔══════════════════════════════════════╗\x1b[0m");
        println!("\x1b[36m║          Session Information         ║\x1b[0m");
        println!("\x1b[36m╚══════════════════════════════════════╝\x1b[0m");
        
        if let Some(ref id) = self.current_session_id {
            println!("  Session ID: \x1b[32m{}\x1b[0m", id);
        } else {
            println!("  Session ID: \x1b[90m<new session>\x1b[0m");
        }
        
        println!("  Output Format: {}", self.config.output_format);
        println!("  GPU Buffer Size: {} slots", self.n_slots);
        println!("  GPU Acceleration: {}", if self.config.use_gpu { "ENABLED" } else { "DISABLED" });
        println!("  Verbose Mode: {}", if self.config.verbose { "ON" } else { "OFF" });
        
        if !self.config.allowed_tools.is_empty() {
            println!("  Allowed Tools: {}", self.config.allowed_tools.join(", "));
        }
        
        if self.gpu_messages_processed > 0 {
            println!("  GPU Messages Processed: {}", self.gpu_messages_processed);
        }
        
        println!();
    }
    
    fn print_gpu_stats(&self) -> Result<(), Box<dyn std::error::Error>> {
        println!("\n\x1b[36m╔══════════════════════════════════════╗\x1b[0m");
        println!("\x1b[36m║            GPU Statistics            ║\x1b[0m");
        println!("\x1b[36m╚══════════════════════════════════════╝\x1b[0m");
        
        #[cfg(feature = "cuda")]
        {
            if let Some(ref _gpu) = self.gpu_producer {
                println!("  Backend: CUDA");
                println!("  Device: GPU 0");
                println!("  Buffer Size: {} slots", self.n_slots);
                println!("  Messages Processed: {}", self.gpu_messages_processed);
                println!("  Status: \x1b[32mActive\x1b[0m");
            } else {
                println!("  Status: \x1b[90mNot initialized\x1b[0m");
            }
        }
        
        #[cfg(feature = "webgpu")]
        {
            println!("  Backend: WebGPU");
            println!("  Status: \x1b[32mActive\x1b[0m");
        }
        
        #[cfg(not(any(feature = "cuda", feature = "webgpu")))]
        {
            println!("  Status: \x1b[90mNo GPU support compiled\x1b[0m");
            println!("  Using CPU-only ring buffer");
        }
        
        println!();
        Ok(())
    }
    
    fn list_available_tools(&self) -> Result<(), Box<dyn std::error::Error>> {
        println!("\n\x1b[36mQuerying available MCP tools...\x1b[0m");
        
        // Run claude with list tools command
        let output = Command::new("claude")
            .arg("--list-tools")
            .output();
        
        match output {
            Ok(output) if output.status.success() => {
                let tools = String::from_utf8_lossy(&output.stdout);
                if tools.trim().is_empty() {
                    println!("\x1b[90mNo MCP tools configured\x1b[0m");
                } else {
                    println!("{}", tools);
                }
            }
            Ok(_) => {
                println!("\x1b[90mNo MCP tools configured or command not supported\x1b[0m");
            }
            Err(e) => {
                println!("\x1b[31mError querying tools: {}\x1b[0m", e);
            }
        }
        
        Ok(())
    }
}

/// Benchmark mode - Test GPU streaming with real Claude responses
fn run_benchmark_mode(n_slots: usize, config: ClaudeConfig) -> Result<(), Box<dyn std::error::Error>> {
    println!("\x1b[32m● Benchmark Mode\x1b[0m - Testing Claude + GPU Performance");
    println!("GPU: {}, Buffer: {} slots\n", 
             if config.use_gpu { "ENABLED" } else { "DISABLED" },
             n_slots);
    
    let test_queries = vec![
        ("Simple", "What is 2+2?"),
        ("Medium", "Write a haiku about GPU acceleration"),
        ("Complex", "Explain the concept of ring buffers in computer science"),
    ];
    
    let mut session = ClaudeSession::new(n_slots, config)?;
    let mut results = Vec::new();
    
    for (name, query) in test_queries {
        println!("Test: {} - \"{}\"", name, query);
        let start = Instant::now();
        
        match session.query(query) {
            Ok(stats) => {
                let elapsed = start.elapsed();
                println!("\x1b[32m✓\x1b[0m Time: {:?}", elapsed);
                
                let tokens = stats.get("total_tokens")
                    .and_then(|t| t.as_u64())
                    .unwrap_or(0);
                
                let bytes = stats.get("total_bytes")
                    .and_then(|b| b.as_u64())
                    .unwrap_or(0);
                
                if tokens > 0 {
                    let tokens_per_sec = tokens as f64 / elapsed.as_secs_f64();
                    println!("  Throughput: {:.1} tokens/sec", tokens_per_sec);
                }
                
                if bytes > 0 {
                    let mb_per_sec = (bytes as f64 / 1_000_000.0) / elapsed.as_secs_f64();
                    println!("  Bandwidth: {:.2} MB/s", mb_per_sec);
                }
                
                if let Some(gpu_msgs) = stats.get("gpu_messages_processed") {
                    println!("  GPU Messages: {}", gpu_msgs);
                }
                
                results.push((name, elapsed, tokens, bytes));
            }
            Err(e) => {
                println!("\x1b[31m✗\x1b[0m Error: {}", e);
            }
        }
        
        println!();
        
        // Reset for next test
        session.new_conversation()?;
    }
    
    // Print summary
    println!("\x1b[36m╔══════════════════════════════════════╗\x1b[0m");
    println!("\x1b[36m║          Benchmark Summary           ║\x1b[0m");
    println!("\x1b[36m╚══════════════════════════════════════╝\x1b[0m");
    
    for (name, elapsed, tokens, bytes) in results {
        println!("  {}: {:?} ({} tokens, {} bytes)", name, elapsed, tokens, bytes);
    }
    
    println!();
    Ok(())
}

fn print_banner(use_gpu: bool) -> Result<(), Box<dyn std::error::Error>> {
    execute!(
        io::stdout(),
        SetForegroundColor(Color::Cyan),
        Print("╔══════════════════════════════════════════════════════╗\n"),
        Print("║       "),
        SetForegroundColor(Color::White),
        Print("Perdix: Claude Code GPU Accelerator"),
        SetForegroundColor(Color::Cyan),
        Print("          ║\n"),
        Print("║         Real Claude Integration via CLI              ║\n"),
        Print("╚══════════════════════════════════════════════════════╝\n"),
        ResetColor
    )?;
    
    if use_gpu {
        #[cfg(feature = "cuda")]
        println!("\x1b[32m✓ CUDA support enabled\x1b[0m");
        
        #[cfg(feature = "webgpu")]
        println!("\x1b[32m✓ WebGPU support enabled\x1b[0m");
    }
    
    println!();
    Ok(())
}

fn print_help() {
    println!("Perdix - GPU-Accelerated Claude Code Terminal");
    println!();
    println!("USAGE:");
    println!("    perdix [OPTIONS] [QUERY]");
    println!();
    println!("MODES:");
    println!("    --repl           Interactive REPL mode (default)");
    println!("    --headless       Single query execution");
    println!("    --benchmark      Performance testing mode");
    println!();
    println!("OPTIONS:");
    println!("    --format FORMAT      Output format: text, json, stream-json (default)");
    println!("    --resume ID          Resume specific session");
    println!("    --continue           Continue last conversation");
    println!("    --allow-tools LIST   Comma-separated allowed MCP tools");
    println!("    --mcp-config FILE    Load MCP servers from JSON");
    println!("    --slots N            GPU buffer size (default: 8192)");
    println!("    --no-gpu             Disable GPU acceleration");
    println!("    --verbose            Enable verbose logging");
    println!("    --help               Show this help");
    println!();
    println!("EXAMPLES:");
    println!("    perdix                           # Interactive mode");
    println!("    perdix --headless \"Your query\"   # Single query");
    println!("    perdix --continue                # Resume last chat");
    println!("    perdix --benchmark               # Test performance");
    println!();
    println!("GPU SUPPORT:");
    #[cfg(feature = "cuda")]
    println!("    CUDA: Available");
    #[cfg(feature = "webgpu")]
    println!("    WebGPU: Available");
    #[cfg(not(any(feature = "cuda", feature = "webgpu")))]
    println!("    GPU: Not compiled (use --features cuda or webgpu)");
}

fn print_interactive_help() {
    println!("\n\x1b[36m╔══════════════════════════════════════╗\x1b[0m");
    println!("\x1b[36m║       Interactive Commands           ║\x1b[0m");
    println!("\x1b[36m╚══════════════════════════════════════╝\x1b[0m");
    println!();
    println!("  /exit, /quit  - Exit the session");
    println!("  /new          - Start new conversation");
    println!("  /session      - Show session info");
    println!("  /tools        - List available MCP tools");
    println!("  /gpu          - Show GPU statistics");
    println!("  /help         - Show this help");
    println!();
    println!("Just type your message to send to Claude.");
    println!();
}
use crate::buffer::{Buffer, Consumer};
use crate::runtime::runtime_compiler::{CudaFunction, CudaModule, CudaRuntimeCompiler};

/// Perdix runtime system using NVRTC compilation
pub struct JITRuntime {
    #[allow(dead_code)]
    compiler: CudaRuntimeCompiler,
    #[allow(dead_code)]
    module: CudaModule,
    #[allow(dead_code)]
    init_kernel: CudaFunction,
    test_kernel: CudaFunction,
    ansi_kernel: Option<CudaFunction>,
    buffer: Buffer,
}

impl JITRuntime {
    /// Create a new Perdix runtime with JIT-compiled kernels
    pub fn new(device_id: i32, n_slots: usize) -> Result<Self, Box<dyn std::error::Error>> {
        println!("Initializing Perdix Runtime...");
        // Ensure n_slots is power of 2
        if !n_slots.is_power_of_two() {
            return Err("n_slots must be a power of 2".into());
        }

        // Create compiler
        let compiler = CudaRuntimeCompiler::new(device_id)?;
        
        // Get kernel source
        let source = crate::runtime::kernel_source::get_kernel_source(None);
        
        // Compile to PTX
        let ptx = compiler.compile_to_ptx(&source, "perdix_kernels", "compute_89")?;
        
        // Load the compiled module
        let module = compiler.load_ptx_module(&ptx)?;

        // Get kernel functions
        let test_kernel = module.get_function("perdix_test_kernel")?;
        let init_kernel = module.get_function("perdix_init_kernel")?;
        let ansi_kernel = module.get_function("perdix_ansi_kernel").ok();

        // The Buffer now handles the CUDA initialization and allocation.
        // It returns a single object managing one contiguous block of memory.
        let buffer = Buffer::new(n_slots)?;
        
        println!("Initializing ring buffer with CUDA kernel...");
        let (header_ptr, slots_ptr) = buffer.as_raw_parts();
        
        let mut params = crate::runtime::cuda_params::CudaParams::new();
        params.add_device_ptr(slots_ptr as *mut _);
        params.add_device_ptr(header_ptr as *mut _);
        params.add_i32(n_slots as i32);
        
        init_kernel.launch((1, 1, 1), (1, 1, 1), 0, params.as_kernel_params())?;
        
        println!("Perdix Runtime initialized successfully!");
        
        Ok(Self {
            compiler,
            module,
            init_kernel,
            test_kernel,
            ansi_kernel,
            buffer,
        })
    }

    /// Returns a consumer handle for reading results from the CPU.
    /// Note: The producer handle needs to be created separately due to lifetime constraints.
    pub fn get_consumer(&mut self) -> Consumer<'_> {
        let (_prod_ref, cons_ref) = self.buffer.split_mut();
        cons_ref
    }
    
    /// Get the raw buffer pointers for launching kernels
    pub fn get_raw_pointers(&self) -> (*mut crate::buffer::Header, *mut crate::buffer::Slot) {
        self.buffer.as_raw_parts()
    }
    
    /// Get a reference to the test kernel
    pub fn test_kernel(&self) -> &CudaFunction {
        &self.test_kernel
    }
    
    /// Get a reference to the ANSI kernel if available
    pub fn ansi_kernel(&self) -> Option<&CudaFunction> {
        self.ansi_kernel.as_ref()
    }
    
    /// Run the test kernel with a specified number of messages
    pub fn run_test(&mut self, n_messages: usize) -> Result<(), Box<dyn std::error::Error>> {
        let (header_ptr, slots_ptr) = self.buffer.as_raw_parts();
        
        let mut params = crate::runtime::cuda_params::CudaParams::new();
        params.add_device_ptr(slots_ptr as *mut _);
        params.add_device_ptr(header_ptr as *mut _);
        params.add_i32(n_messages as i32);
        
        self.test_kernel.launch(
            (1, 1, 1),
            (256, 1, 1),
            0,
            params.as_kernel_params(),
        )?;
        
        Ok(())
    }
    
    /// Run the ANSI kernel with formatted messages
    pub fn run_ansi_kernel(
        &mut self,
        messages: &[String],
        agent_types: &[u32],
    ) -> Result<(), Box<dyn std::error::Error>> {
        let ansi_kernel = self.ansi_kernel.as_ref()
            .ok_or("ANSI kernel not available")?;
        
        let (header_ptr, slots_ptr) = self.buffer.as_raw_parts();
        
        // Prepare message data
        let mut params = crate::runtime::cuda_params::CudaParams::new();
        params.add_device_ptr(slots_ptr as *mut _);
        params.add_device_ptr(header_ptr as *mut _);
        params.add_i32(messages.len() as i32);
        
        // For now, we'll use the test kernel as ANSI processing would require
        // uploading the message strings to device memory
        ansi_kernel.launch(
            (((messages.len() + 255) / 256) as u32, 1, 1),
            (256, 1, 1),
            0,
            params.as_kernel_params(),
        )?;
        
        Ok(())
    }
    
    /// Verify that messages were correctly written to the buffer
    pub fn verify_messages(&mut self, expected_count: usize) -> Result<(), Box<dyn std::error::Error>> {
        let mut consumer = self.get_consumer();
        let messages = consumer.consume_available(Some(expected_count));
        
        if messages.len() != expected_count {
            return Err(format!(
                "Expected {} messages, got {}",
                expected_count,
                messages.len()
            ).into());
        }
        
        Ok(())
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_runtime_initialization() {
        let runtime = JITRuntime::new(0, 1024);
        assert!(runtime.is_ok(), "Failed to initialize runtime: {:?}", runtime.err());
    }
    
    #[test]
    fn test_basic_messaging() {
        let mut runtime = JITRuntime::new(0, 1024).expect("Failed to create runtime");
        let result = runtime.run_test(100);
        assert!(result.is_ok(), "Test failed: {:?}", result.err());
    }
    
    #[test]
    fn test_ansi_messages() {
        let mut runtime = JITRuntime::new(0, 1024).expect("Failed to create runtime");
        
        let messages = vec![
            "System starting up".to_string(),
            "User logged in".to_string(),
            "Processing request".to_string(),
            "ERROR: Connection failed".to_string(),
            "WARNING: Low memory".to_string(),
        ];
        
        // Agent types: 0=SYSTEM, 1=USER, 2=ASSISTANT, 3=ERROR, 4=WARNING, 5=INFO, 6=DEBUG, 7=TRACE
        let agent_types = vec![0, 1, 2, 3, 4, 5, 6, 7]; 

        let result = runtime.run_ansi_kernel(&messages, &agent_types);
        assert!(result.is_ok(), "ANSI test failed: {:?}", result.err());
        
        // Verify messages
        runtime.verify_messages(messages.len()).expect("Failed to verify messages");
    }
}

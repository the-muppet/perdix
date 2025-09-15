use wgpu::{Device, Queue, Buffer as WgpuBuffer, BufferUsages, Instance};
use std::sync::Arc;
use crate::buffer::{Header, Slot};
use pollster;

/// WebGPU-accelerated producer for universal GPU support
pub struct WebGpuProducer {
    device: Arc<Device>,
    queue: Arc<Queue>,
    ring_buffer: WgpuBuffer,
    header_buffer: WgpuBuffer,
    n_slots: usize,
}

impl WebGpuProducer {
    /// Initialize WebGPU and create buffers
    pub async fn new_async(n_slots: usize) -> Result<Self, String> {
        // Create instance with all available backends
        let instance = Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            dx12_shader_compiler: Default::default(),
            flags: wgpu::InstanceFlags::empty(),
            gles_minor_version: wgpu::Gles3MinorVersion::Automatic,
        });
        
        // Request adapter (GPU)
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
            })
            .await
            .ok_or("No suitable GPU adapter found")?;
        
        // Get adapter info
        let info = adapter.get_info();
        println!("WebGPU Adapter: {} ({:?})", info.name, info.backend);
        println!("  Device Type: {:?}", info.device_type);
        println!("  Driver: {}", info.driver);
        
        // Create device and queue
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("Perdix WebGPU Device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    memory_hints: Default::default(),
                },
                None,
            )
            .await
            .map_err(|e| format!("Failed to create WebGPU device: {}", e))?;
        
        let device = Arc::new(device);
        let queue = Arc::new(queue);
        
        // Create ring buffer on GPU
        let ring_buffer_size = n_slots * std::mem::size_of::<Slot>();
        let ring_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Perdix Ring Buffer"),
            size: ring_buffer_size as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        
        // Create header buffer
        let header_size = std::mem::size_of::<Header>();
        let header_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Perdix Header"),
            size: header_size as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        
        Ok(Self {
            device,
            queue,
            ring_buffer,
            header_buffer,
            n_slots,
        })
    }
    
    /// Synchronous constructor
    pub fn new(n_slots: usize) -> Result<Self, String> {
        pollster::block_on(Self::new_async(n_slots))
    }
    
    /// Write message to GPU ring buffer
    pub fn produce(&mut self, data: &[u8], seq: u64) -> Result<(), String> {
        if data.len() > 240 {
            return Err("Message too large".to_string());
        }
        
        // Create slot data
        let mut slot = Slot {
            seq,
            len: data.len() as u32,
            flags: 0,
            _pad1: 0,
            payload: [0; 240],
            _pad2: [0; 8],
        };
        slot.payload[..data.len()].copy_from_slice(data);
        
        // Calculate slot index
        let slot_idx = (seq as usize) & (self.n_slots - 1);
        let offset = slot_idx * std::mem::size_of::<Slot>();
        
        // Write to GPU buffer
        // Convert slot to bytes manually since we can't use bytemuck
        let slot_bytes = unsafe {
            std::slice::from_raw_parts(
                &slot as *const Slot as *const u8,
                std::mem::size_of::<Slot>(),
            )
        };
        self.queue.write_buffer(
            &self.ring_buffer,
            offset as u64,
            slot_bytes,
        );
        
        // Submit commands
        self.queue.submit(None);
        
        Ok(())
    }
    
    /// Launch compute shader for batch processing
    pub fn launch_compute(&self, n_messages: u32) -> Result<(), String> {
        // This would launch a WGSL compute shader
        // For now, just a placeholder
        println!("WebGPU: Would launch compute shader for {} messages", n_messages);
        Ok(())
    }
}

/// WebGPU buffer implementation that can fallback from CUDA
pub struct WebGpuBuffer {
    device: Arc<Device>,
    queue: Arc<Queue>,
    slots_buffer: WgpuBuffer,
    header_buffer: WgpuBuffer,
    n_slots: usize,
}

impl WebGpuBuffer {
    pub fn new(n_slots: usize) -> Result<Self, String> {
        pollster::block_on(Self::new_async(n_slots))
    }
    
    pub async fn new_async(n_slots: usize) -> Result<Self, String> {
        if !n_slots.is_power_of_two() {
            return Err("n_slots must be power of 2".to_string());
        }
        
        // Initialize WebGPU
        let instance = Instance::default();
        
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                ..Default::default()
            })
            .await
            .ok_or("No GPU adapter found")?;
        
        let (device, queue) = adapter
            .request_device(&Default::default(), None)
            .await
            .map_err(|e| format!("Device creation failed: {}", e))?;
        
        let device = Arc::new(device);
        let queue = Arc::new(queue);
        
        // Create buffers
        let slots_size = n_slots * std::mem::size_of::<Slot>();
        let slots_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Ring Buffer Slots"),
            size: slots_size as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        
        let header_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Ring Buffer Header"),
            size: std::mem::size_of::<Header>() as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        
        Ok(Self {
            device,
            queue,
            slots_buffer,
            header_buffer,
            n_slots,
        })
    }
    
    /// Get raw buffer handles for compute shader binding
    pub fn get_buffers(&self) -> (&WgpuBuffer, &WgpuBuffer) {
        (&self.slots_buffer, &self.header_buffer)
    }
}

/// Compute shader module for WebGPU
pub struct WebGpuCompute {
    device: Arc<Device>,
    queue: Arc<Queue>,
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
}

impl WebGpuCompute {
    /// Create compute pipeline for message processing
    pub fn new(device: Arc<Device>, queue: Arc<Queue>, buffer: &WebGpuBuffer) -> Result<Self, String> {
        // WGSL shader for processing messages
        let shader_source = r#"
            struct Slot {
                seq: u64,
                len: u32,
                flags: u32,
                payload: array<u32, 60>,  // 240 bytes / 4
            }
            
            struct Header {
                write_idx: atomic<u64>,
                read_idx: atomic<u64>,
            }
            
            @group(0) @binding(0)
            var<storage, read_write> slots: array<Slot>;
            
            @group(0) @binding(1)
            var<storage, read_write> header: Header;
            
            @compute @workgroup_size(64)
            fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
                let idx = global_id.x;
                
                // Process message at this index
                // This is where we'd implement the GPU-side logic
                // For ANSI formatting, etc.
            }
        "#;
        
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Perdix Compute Shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });
        
        // Create pipeline layout
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Perdix Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Perdix Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        
        // Create compute pipeline
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Perdix Compute Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        
        // Create bind group
        let (slots_buffer, header_buffer) = buffer.get_buffers();
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Perdix Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: slots_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: header_buffer.as_entire_binding(),
                },
            ],
        });
        
        Ok(Self {
            device,
            queue,
            pipeline,
            bind_group,
        })
    }
    
    /// Execute compute shader
    pub fn dispatch(&self, workgroups: u32) {
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Perdix Compute Encoder"),
        });
        
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Perdix Compute Pass"),
                timestamp_writes: None,
            });
            
            compute_pass.set_pipeline(&self.pipeline);
            compute_pass.set_bind_group(0, &self.bind_group, &[]);
            compute_pass.dispatch_workgroups(workgroups, 1, 1);
        }
        
        self.queue.submit(Some(encoder.finish()));
    }
}
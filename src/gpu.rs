use std::sync::Arc;
use crate::buffer::{Buffer, Producer, Consumer};
use crate::buffer::ffi::{AgentType, StreamContext};

/// GPU-accelerated producer for the ring buffer
pub struct GpuProducer {
    producer: Producer<'static>,
    stream: u64,
    device_id: i32,
}

impl GpuProducer {
    /// Create a new GPU producer from a buffer
    pub fn new(buffer: Buffer, device_id: i32) -> Result<Self, String> {
        // Split the buffer into producer and consumer
        let (producer, _consumer) = buffer.split();
        
        Ok(Self {
            producer,
            stream: 0, // Default stream
            device_id,
        })
    }
    
    /// Create from an existing producer
    pub fn from_producer(producer: Producer<'static>, device_id: i32) -> Self {
        Self {
            producer,
            stream: 0,
            device_id,
        }
    }
    
    /// Process a batch of messages using CUDA kernel
    pub fn process_batch(&mut self, contexts: &[StreamContext], enable_metrics: bool) -> Result<(), String> {
        self.producer.process_agent_responses(contexts, enable_metrics, self.stream)
    }
    
    /// Run a simple test kernel
    pub fn run_test(&mut self, n_messages: u32) -> Result<(), String> {
        self.producer.run_test(n_messages)
    }
    
    /// Get the stream handle
    pub fn stream(&self) -> u64 {
        self.stream
    }
    
    /// Set a custom CUDA stream
    pub fn set_stream(&mut self, stream: u64) {
        self.stream = stream;
    }
    
    /// Get the underlying producer
    pub fn producer(&self) -> &Producer<'static> {
        &self.producer
    }
    
    /// Get the underlying producer mutably
    pub fn producer_mut(&mut self) -> &mut Producer<'static> {
        &mut self.producer
    }
}
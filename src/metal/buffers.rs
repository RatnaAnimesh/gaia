use metal::{Device, Buffer, MTLResourceOptions, NSRange};
use std::mem;

pub struct SharedBuffer<T> {
    pub buffer: Buffer,
    pub length: usize,
    _marker: std::marker::PhantomData<T>,
}

impl<T> SharedBuffer<T> {
    /// Allocates a new metal buffer in Unified Memory accessible by both CPU and GPU
    pub fn new(device: &Device, capacity: usize) -> Self {
        let size = (capacity * mem::size_of::<T>()) as u64;
        let options = MTLResourceOptions::StorageModeShared;
        
        // Handle 0 size gracefully
        let safe_size = if size == 0 { 4 } else { size };
        
        let buffer = device.new_buffer(safe_size, options);
        
        Self {
            buffer,
            length: capacity,
            _marker: std::marker::PhantomData,
        }
    }

    /// Obtain a mutable slice on the CPU side
    pub fn as_slice_mut(&mut self) -> &mut [T] {
        if self.length == 0 {
            return &mut [];
        }
        unsafe {
            std::slice::from_raw_parts_mut(
                self.buffer.contents() as *mut T,
                self.length
            )
        }
    }

    /// Obtain an immutable slice on the CPU side
    pub fn as_slice(&self) -> &[T] {
        if self.length == 0 {
            return &[];
        }
        unsafe {
            std::slice::from_raw_parts(
                self.buffer.contents() as *const T,
                self.length
            )
        }
    }
}

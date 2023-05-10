// This is the skeleton for the CUDA implementation

use rustacuda::launch;
use rustacuda::prelude::*;
use std::error::Error;
use std::ffi::CString;

use crate::SIZE;

// #[repr(transparent)]
// pub struct ConvLayer(pub [[f64; SIZE]; SIZE]);

pub struct CudaContext {
    module: Module,
    stream: Stream,
    _context: Context,
}

impl CudaContext {
    pub fn init() -> Result<Self, Box<dyn Error>> {
        // Initialize the CUDA API
        rustacuda::init(CudaFlags::empty())?;
        
        // Get the first device
        let device = Device::get_device(0)?;

        // Create a context associated to this device
        let context = Context::create_and_push(
            ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO, device)?;

        // Load the module containing the function we want to call
        let module_data = CString::new(include_str!("../resources/weightedsum.ptx"))?;
        let module = Module::load_from_string(&module_data)?;

        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

        Ok(
            Self{
                module: module,
                stream: stream,
                _context: context,
            }
        )

    }

    #[allow(non_snake_case)]
    pub fn compute(&mut self, P:Vec<f32>, A:Vec<f32>, sums:&mut Vec<f32>) -> Result<usize, Box<dyn Error>> {
        let mut P_buffer = DeviceBuffer::from_slice(&P).unwrap();
        let mut A_buffer = DeviceBuffer::from_slice(&A).unwrap();
        let mut sums_buffer = DeviceBuffer::from_slice(&sums).unwrap();
        
        let module = &self.module;
        let stream = &self.stream;

        unsafe {
            let result = launch!(module.add<<<1, 1024, 0, stream>>>(
                P_buffer.as_device_ptr(),
                A_buffer.as_device_ptr(),
                sums_buffer.as_device_ptr()
            ));
            result?;
        }

        self.stream.synchronize()?;
        let output_vec: Vec<i32> = vec![0; SIZE + 2];

        let mut output = DeviceBuffer::from_slice(&output_vec).unwrap();

        unsafe {
            let result = launch!(module.find_max_index<<<1, 1024, 0, stream>>>(
                output.as_device_ptr(),
                sums_buffer.as_device_ptr()
            ));
            result?;
        }

        self.stream.synchronize()?;
        
        let mut indices: Vec<i32> = vec![0; SIZE + 2];
        output.copy_to(&mut indices).unwrap();

        let mut calculated_sums = vec![0.0; (SIZE+2)*(SIZE+2)];
        sums_buffer.copy_to(&mut calculated_sums).unwrap();

        let mut max_i: i32 = 0;
        for i in indices {
            if i == -1 {
                continue;
            }
            if calculated_sums[max_i as usize] < calculated_sums[i as usize] {
                max_i = i;
            }
        }
        Ok(max_i as usize)
    }
}

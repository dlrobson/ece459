// This is the skeleton for the CUDA implementation

use crate::cnn::*;
use rustacuda::launch;
use rustacuda::memory::*;
use rustacuda::stream::*;
use rustacuda::prelude::*;
use std::error::Error;
use std::ffi::CString;

// Fields need to be ordered this way so the DeviceBoxes are
// dropped before the Context. Otherwise the drop will panic.

pub struct CudaContext {
    conv_layer: DeviceBox<ConvLayer>,
    output_layer: DeviceBox<OutputLayer>,
    module: Module,
    stream: Stream,
    _context: Context,
}

impl CudaContext {
    pub fn init(cnn: &Cnn) -> Result<Self, Box<dyn Error>> {
        // Initialize the CUDA API
        rustacuda::init(CudaFlags::empty())?;
        
        // Get the first device
        let device = Device::get_device(0)?;

        // Create a context associated to this device
        let context = Context::create_and_push(
            ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO, device)?;

        // Load the module containing the function we want to call
        let module_data = CString::new(include_str!("../kernel/kernel.ptx"))?;
        let module = Module::load_from_string(&module_data)?;

        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

        Ok(
            Self{
                conv_layer: DeviceBox::new(&cnn.conv_layer).unwrap(),
                output_layer: DeviceBox::new(&cnn.output_layer).unwrap(),
                module: module,
                stream: stream,
                _context: context,
            }
        )
    }

    pub fn compute(&mut self, input: &InputMatrix) -> Result<OutputVec, Box<dyn Error>> {
        let mut device_input = DeviceBox::new(input)?;
        let mut conv_output = DeviceBox::new(
            &ConvOutput{0: [[[0f64; CONV_OUT_DIM]; CONV_OUT_DIM]; CONV_LAYER_SIZE]}
        )?;

        let module = &self.module;
        let stream = &self.stream;

        // Convolution and ReLU layers
        unsafe {
            let result = launch!(module.convolution_layer<<<10, 512, 0, stream>>>(
                    device_input.as_device_ptr(),
                    self.conv_layer.as_device_ptr(),
                    conv_output.as_device_ptr()
            ));
            result?;
        }

        self.stream.synchronize()?;

        let mut output = DeviceBox::new(
            &OutputVec{0: [0f64; OUT_LAYER_SIZE]}
        )?;

        unsafe {
            let result = launch!(
                module.output_layer<<<10, 512, 0, stream>>>(
                    conv_output.as_device_ptr(),
                    self.output_layer.as_device_ptr(),
                    output.as_device_ptr()
                )
            );
            result?;
        }
        self.stream.synchronize()?;
        
        let mut return_val = OutputVec{0: [0f64; OUT_LAYER_SIZE]};
        output.copy_to(&mut return_val).unwrap();

        Ok(return_val)
    }
}

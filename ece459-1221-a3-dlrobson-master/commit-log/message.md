# GPU Parallelization using CUDA

A CNN was implemented on a GPU using CUDA. Given that this problem is very parallelizable, CUDA is able to take advantage of a GPU's multi-threaded environment, resulting in a significantly faster completion time. On `ecetesla0`, the CUDA implementation was about 4x.

## convolution_layer()
For the convolution layer, the problem was divided into 10 blocks of 416 threads. Each block will operate on a single layer, and apply a single filter. Since there are 10 layers and filters, we will have 10 corresponding blocks. 416 threads was selected since it's the first multiple of 32 greater than 400. In the GPU architecture, 32 threads are packed together, helping to improve the performance of the program. 400 is the ideal number since the filter is applied on 20x20=400 sections of the input layer. Each of these blocks can work independently of each other, resulting in easy parallelization.

## ReLU
The ReLU layer was implemented into the convolution layer, and the value is updated before it is written to memory. It is simply a check to set the `sum = 0` if the sum is less than zero.

## output_layer()
The output layer was divided up similarly to the convolution layer, where each block corresponds to an output vector. While it would be ideal for there to be 4000 threads, most GPU's cap out at 1024. Thus, I kept it at 512 threads to let it work on most GPUs, and then each thread will be responsible for 8 multiplications. The difficult part of the dot product is the addition of the multiplications, which requires synchronization between the threads within a block (`__syncthreads();`). The original sum is held in a length-4000 double array, A decomposition approach was selected, where each iteration a thread would add it's neighbouring value. This resulted in the script using half of the operations each iteration. The size of the usable array decreased from 4000 -> 2000 -> 1000 -> 500 -> ... -> 1. `__syncthreads();` needs to be called each iteration, affecting the performance. Once the array was a single value, this value was written to the output value. Unfortunately each thread does not do an equal amount of work throughout, however it is still much more efficient than the CPU method.

## cuda.rs
`cuda.rs` is responsible for initializing the CUDA kernels. The data has to be packaged in a method where the GPU can access it, which is the purpose of `DeviceBox`. The `launch!` activates the kernel with the given parameters, and the number of blocks and threads per block. Essentially, `cuda.rs` acts as an interface to the `kernel.ptx`. The data has to be unpacked using the `copy_to()` function, to make the data accessible again.

## Correctness
The correctness was verified with the helpful `compare.py` script. The outputs match! While debugging, a lot of printing on the CUDA side to verify indices, values, and operations.

## Performance
The CUDA method was evaluated using the output value from `main.rs`. On `ecetesla0`, the average CPU and CUDA runtime was about `96 ms` and `23 ms` respectively, which is a significant improvement.

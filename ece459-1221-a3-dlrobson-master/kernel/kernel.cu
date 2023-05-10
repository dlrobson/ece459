// Very minimal skeleton for the kernel

#include <stdio.h>

static constexpr auto INPUT_DIM = 100;
static constexpr auto FILTER_DIM = 5;
static constexpr auto FILTER_SIZE = 25;
static constexpr auto CONV_OUT_DIM = 20;
static constexpr auto CONV_OUT_LAYER_COUNT = 400;
static constexpr auto OUT_NEURON_DIM = 4000;

/**
 * input: 1 x 100 x 100
 * filters: 10 x 5 x 5
 * output: 10 x 20 x 20
 */
extern "C" __global__ void convolution_layer(const double* input, const double* filter, double* output)
{
    if (threadIdx.x >= CONV_OUT_LAYER_COUNT)
    {
        return;
    }

    // Read in the current block info
    const auto& depth_index = blockIdx.x;
    const auto row_index = threadIdx.x / CONV_OUT_DIM;
    const auto col_index = threadIdx.x % CONV_OUT_DIM;

    // Choose the correct filter
    size_t filter_i = depth_index * FILTER_SIZE;

    // Choose the correct input
    size_t input_i = row_index * (INPUT_DIM * FILTER_DIM) + col_index * FILTER_DIM;

    // Choose the correct output block
    const size_t output_section_i = depth_index * CONV_OUT_LAYER_COUNT + row_index * CONV_OUT_DIM + col_index;

    // Perform the convolution.
    double sum = 0.0;
    for (size_t i = 0; i < 5; i++)
    {
        for (size_t j = 0; j < 5; j++)
        {
            sum += input[input_i] * filter[filter_i];

            input_i++;
            filter_i++;
        }
        input_i += (INPUT_DIM - FILTER_DIM);
    }

    // ReLU Layer
    // Set the output
    output[output_section_i] = sum > 0 ? sum : 0;
}

/**
 * input: 10 x 20 x 20
 * weights: 1 x 10 x 4000
 * output: 1 x 1 x 10
 */
extern "C" __global__ void output_layer(const double* input, const double* weights, double* output)
{
    
    static constexpr auto THREAD_OVERHEAD = 8;
    const auto initial_val_i = threadIdx.x * THREAD_OVERHEAD;

    if (initial_val_i >= OUT_NEURON_DIM) 
    {
        return;
    }

    const int& output_layer_i = blockIdx.x;
    const int factor_i = output_layer_i * OUT_NEURON_DIM;
    __shared__ double shared_output[OUT_NEURON_DIM];

    for (int val_i = initial_val_i; val_i < initial_val_i + THREAD_OVERHEAD; val_i++)
    {
        shared_output[val_i] = weights[factor_i + val_i] * input[val_i];
    }
    __syncthreads();

    for (int divisor = 1; divisor < OUT_NEURON_DIM; divisor *= 2)
    {   
        if ((initial_val_i + THREAD_OVERHEAD) % (2 * divisor) > THREAD_OVERHEAD)
        {
            return;
        }

        int val_i = (initial_val_i / (2 * divisor)) * (2 * divisor);

        if (val_i < initial_val_i)
        {
            val_i += (2 * divisor);
        }

        for (val_i; val_i < initial_val_i + THREAD_OVERHEAD; val_i += (2 * divisor))
        {
            const int neighbour_i = val_i + divisor;
            if (neighbour_i < OUT_NEURON_DIM)
            {
                shared_output[val_i] += shared_output[neighbour_i];
            }
        }
        __syncthreads();
    }
    
    output[output_layer_i] = shared_output[0];
}

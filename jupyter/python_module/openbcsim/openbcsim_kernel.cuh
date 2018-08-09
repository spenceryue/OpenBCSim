#pragma once
#include "definitions.h"
#include <cuda_runtime_api.h>

#define THREADS_PER_BLOCK 1024

template <class scalar_t>
void launch_projection_kernel (const Simulation<scalar_t> &args, scalar_t *output_buffer,
                               dim3 grid = 1, dim3 block = 1, size_t shared_bytes = 0, cudaStream_t stream = 0);

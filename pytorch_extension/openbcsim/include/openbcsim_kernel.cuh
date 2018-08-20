#pragma once
#include "data_types.h"
#include "export_macros.h"
#include <cuda_runtime_api.h>

template <class scalar_t>
void launch_projection_kernel (const Simulator<scalar_t> &args, scalar_t *output_buffer,
                               dim3 grid = 1, dim3 block = 1, size_t shared_bytes = 0, cudaStream_t stream = 0);

template <class T>
cudaError cuda_malloc_managed (T **pointer_to_pointer, size_t bytes);

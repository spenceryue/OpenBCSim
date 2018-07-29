#pragma once
#include "cuda_kernels_c_interface.h"
#include <cuComplex.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

__global__ void SliceLookupTable (float3 origin, float3 dir0, float3 dir1, float *output, cudaTextureObject_t lut_tex);

template <bool use_arc_projection, bool use_phase_delay, bool use_lut>
__global__ void FixedAlgKernel (const FixedAlgKernelParams params);

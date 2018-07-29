#include "common_definitions.h" // for MAX_SPLINE_DEGREE and MAX_NUM_CUDA_STREAMS
#include "cuda_helpers.h"       // for make_float3
#include "cuda_kernels_common.cuh"
#include "cuda_kernels_projection.cuh"
#include "cuda_kernels_spline2.cuh"
#include <device_launch_parameters.h>
#include <math_functions.h> // for copysignf()

__constant__ float eval_basis[(MAX_SPLINE_DEGREE + 1) * MAX_NUM_CUDA_STREAMS];

bool splineAlg2_updateConstantMemory_internal (float *src, size_t count, size_t offset, cudaMemcpyKind kind, cudaStream_t stream)
{
  const auto res = cudaMemcpyToSymbolAsync (eval_basis, src, count, offset, cudaMemcpyHostToDevice, stream);
  return (res == cudaSuccess);
}

template <bool use_arc_projection, bool use_phase_delay, bool use_lut>
__global__ void SplineAlgKernel (SplineAlgKernelParams params)
{
  const int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (global_idx >= params.NUM_SPLINES)
  {
    return;
  }

  // step 1: evaluate spline
  // to get from one control point to the next, we have
  // to make a jump of size equal to number of splines
  float rendered_x = 0.0f;
  float rendered_y = 0.0f;
  float rendered_z = 0.0f;
  for (int i = params.cs_idx_start; i <= params.cs_idx_end; i++)
  {
    size_t eval_basis_i = i + params.eval_basis_offset_elements;
    rendered_x += params.control_xs[params.NUM_SPLINES * i + global_idx] * eval_basis[eval_basis_i - params.cs_idx_start];
    rendered_y += params.control_ys[params.NUM_SPLINES * i + global_idx] * eval_basis[eval_basis_i - params.cs_idx_start];
    rendered_z += params.control_zs[params.NUM_SPLINES * i + global_idx] * eval_basis[eval_basis_i - params.cs_idx_start];
  }

  // step 2: compute projections
  const auto scatterer_point = make_float3 (rendered_x, rendered_y, rendered_z);
  const auto amplitude = params.control_as[global_idx];
  ProjectionAlg<use_arc_projection,
                use_phase_delay,
                use_lut> (params, scatterer_point, amplitude);
}

// explicit function template instantiations
template __global__ void SplineAlgKernel<false, false, false> (SplineAlgKernelParams params);
template __global__ void SplineAlgKernel<false, false, true> (SplineAlgKernelParams params);
template __global__ void SplineAlgKernel<false, true, false> (SplineAlgKernelParams params);
template __global__ void SplineAlgKernel<false, true, true> (SplineAlgKernelParams params);
template __global__ void SplineAlgKernel<true, false, false> (SplineAlgKernelParams params);
template __global__ void SplineAlgKernel<true, false, true> (SplineAlgKernelParams params);
template __global__ void SplineAlgKernel<true, true, false> (SplineAlgKernelParams params);
template __global__ void SplineAlgKernel<true, true, true> (SplineAlgKernelParams params);

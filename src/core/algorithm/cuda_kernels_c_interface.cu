#include "cuda_kernels_c_interface.h"
#include "cuda_kernels_common.cuh"  // for common kernels
#include "cuda_kernels_fixed.cuh"   // for FixedAlgKernel
#include "cuda_kernels_spline1.cuh" // for splineAlg1_updateConstantMemory_internal
#include "cuda_kernels_spline2.cuh" // for splineAlg2_updateConstantMemory_internal

template <typename T>
void launch_MemsetKernel (unsigned int grid_size, unsigned int block_size, cudaStream_t stream, T *ptr, T value, int num_samples)
{
  MemsetKernel<cuComplex><<<grid_size, block_size, 0, stream>>> (ptr, value, num_samples);
}

template <bool normalize>
void launch_MultiplyFftKernel (unsigned int grid_size, unsigned int block_size, cudaStream_t stream, cufftComplex *time_proj_fft, const cufftComplex *filter_fft, int num_samples)
{
  MultiplyFftKernel<normalize><<<grid_size, block_size, 0, stream>>> (time_proj_fft, filter_fft, num_samples);
}

void launch_DemodulateKernel (unsigned int grid_size, unsigned int block_size, cudaStream_t stream, cuComplex *signal, float w, int stop_index, int radial_decimation)
{
  DemodulateKernel<<<grid_size, block_size, 0, stream>>> (signal, w, stop_index, radial_decimation);
}

void launch_ScaleSignalKernel (unsigned int grid_size, unsigned int block_size, cudaStream_t stream, cufftComplex *signal, float factor, int num_samples)
{
  ScaleSignalKernel<<<grid_size, block_size, 0, stream>>> (signal, factor, num_samples);
}

template <bool A, bool B, bool C, bool D>
void launch_FixedAlgKernel (unsigned int grid_size, unsigned int block_size, cudaStream_t stream, FixedAlgKernelParams params)
{
  FixedAlgKernel<A, B, C, D><<<grid_size, block_size, 0, stream>>> (params);
}

// explicit function template instantiations for required datatypes
template void DLL_PUBLIC launch_MemsetKernel (unsigned int grid_size, unsigned int block_size, cudaStream_t stream, cuComplex *ptr, cuComplex value, int num_samples);

// explicit function template instantiations for different normalizations.
template void DLL_PUBLIC launch_MultiplyFftKernel<false>  (unsigned int grid_size, unsigned int block_size, cudaStream_t stream, cufftComplex *time_proj_fft, const cufftComplex *filter_fft, int num_samples);
template void DLL_PUBLIC launch_MultiplyFftKernel<true>  (unsigned int grid_size, unsigned int block_size, cudaStream_t stream, cufftComplex *time_proj_fft, const cufftComplex *filter_fft, int num_samples);

// fixed algorithm explicit function template instantiations - all combinations
template void DLL_PUBLIC launch_FixedAlgKernel<false, false, false, false> (unsigned int grid_size, unsigned int block_size, cudaStream_t stream, FixedAlgKernelParams params);
template void DLL_PUBLIC launch_FixedAlgKernel<false, false, false, true> (unsigned int grid_size, unsigned int block_size, cudaStream_t stream, FixedAlgKernelParams params);
template void DLL_PUBLIC launch_FixedAlgKernel<false, false, true, false> (unsigned int grid_size, unsigned int block_size, cudaStream_t stream, FixedAlgKernelParams params);
template void DLL_PUBLIC launch_FixedAlgKernel<false, false, true, true> (unsigned int grid_size, unsigned int block_size, cudaStream_t stream, FixedAlgKernelParams params);
template void DLL_PUBLIC launch_FixedAlgKernel<false, true, false, false> (unsigned int grid_size, unsigned int block_size, cudaStream_t stream, FixedAlgKernelParams params);
template void DLL_PUBLIC launch_FixedAlgKernel<false, true, false, true> (unsigned int grid_size, unsigned int block_size, cudaStream_t stream, FixedAlgKernelParams params);
template void DLL_PUBLIC launch_FixedAlgKernel<false, true, true, false> (unsigned int grid_size, unsigned int block_size, cudaStream_t stream, FixedAlgKernelParams params);
template void DLL_PUBLIC launch_FixedAlgKernel<false, true, true, true> (unsigned int grid_size, unsigned int block_size, cudaStream_t stream, FixedAlgKernelParams params);
template void DLL_PUBLIC launch_FixedAlgKernel<true, false, false, false> (unsigned int grid_size, unsigned int block_size, cudaStream_t stream, FixedAlgKernelParams params);
template void DLL_PUBLIC launch_FixedAlgKernel<true, false, false, true> (unsigned int grid_size, unsigned int block_size, cudaStream_t stream, FixedAlgKernelParams params);
template void DLL_PUBLIC launch_FixedAlgKernel<true, false, true, false> (unsigned int grid_size, unsigned int block_size, cudaStream_t stream, FixedAlgKernelParams params);
template void DLL_PUBLIC launch_FixedAlgKernel<true, false, true, true> (unsigned int grid_size, unsigned int block_size, cudaStream_t stream, FixedAlgKernelParams params);
template void DLL_PUBLIC launch_FixedAlgKernel<true, true, false, false> (unsigned int grid_size, unsigned int block_size, cudaStream_t stream, FixedAlgKernelParams params);
template void DLL_PUBLIC launch_FixedAlgKernel<true, true, false, true> (unsigned int grid_size, unsigned int block_size, cudaStream_t stream, FixedAlgKernelParams params);
template void DLL_PUBLIC launch_FixedAlgKernel<true, true, true, false> (unsigned int grid_size, unsigned int block_size, cudaStream_t stream, FixedAlgKernelParams params);
template void DLL_PUBLIC launch_FixedAlgKernel<true, true, true, true> (unsigned int grid_size, unsigned int block_size, cudaStream_t stream, FixedAlgKernelParams params);

bool splineAlg1_updateConstantMemory (float *src_ptr, size_t num_bytes)
{
  return splineAlg1_updateConstantMemory_internal (src_ptr, num_bytes);
}

void launch_RenderSplineKernel (unsigned int grid_size, unsigned int block_size, cudaStream_t stream,
                                const float *control_xs,
                                const float *control_ys,
                                const float *control_zs,
                                float *rendered_xs,
                                float *rendered_ys,
                                float *rendered_zs,
                                int cs_idx_start,
                                int cs_idx_end,
                                int NUM_SPLINES)
{
  RenderSplineKernel<<<grid_size, block_size, 0, stream>>> (control_xs, control_ys, control_zs,
                                                            rendered_xs, rendered_ys, rendered_zs,
                                                            cs_idx_start, cs_idx_end, NUM_SPLINES);
}

void launch_SliceLookupTable (unsigned int grid_size0, unsigned int grid_size1, unsigned int block_size, cudaStream_t stream,
                              float3 origin,
                              float3 dir0,
                              float3 dir1,
                              float *output,
                              cudaTextureObject_t lut_tex)
{
  dim3 grid_size (grid_size0, grid_size1, 1);
  SliceLookupTable<<<grid_size, block_size, 0, stream>>> (origin, dir0, dir1, output, lut_tex);
}

bool splineAlg2_updateConstantMemory (float *src, size_t count, size_t offset, cudaMemcpyKind kind, cudaStream_t stream)
{
  return splineAlg2_updateConstantMemory_internal (src, count, offset, kind, stream);
}

template <bool A, bool B, bool C>
void launch_SplineAlgKernel (unsigned int grid_size, unsigned int block_size, cudaStream_t stream, SplineAlgKernelParams params)
{
  SplineAlgKernel<A, B, C><<<grid_size, block_size, 0, stream>>> (params);
}

// spline algorithm2 explicit function template instantiations - all combinations
template void DLL_PUBLIC launch_SplineAlgKernel<false, false, false> (unsigned int grid_size, unsigned int block_size, cudaStream_t stream, SplineAlgKernelParams params);
template void DLL_PUBLIC launch_SplineAlgKernel<false, false, true> (unsigned int grid_size, unsigned int block_size, cudaStream_t stream, SplineAlgKernelParams params);
template void DLL_PUBLIC launch_SplineAlgKernel<false, true, false> (unsigned int grid_size, unsigned int block_size, cudaStream_t stream, SplineAlgKernelParams params);
template void DLL_PUBLIC launch_SplineAlgKernel<false, true, true> (unsigned int grid_size, unsigned int block_size, cudaStream_t stream, SplineAlgKernelParams params);
template void DLL_PUBLIC launch_SplineAlgKernel<true, false, false> (unsigned int grid_size, unsigned int block_size, cudaStream_t stream, SplineAlgKernelParams params);
template void DLL_PUBLIC launch_SplineAlgKernel<true, false, true> (unsigned int grid_size, unsigned int block_size, cudaStream_t stream, SplineAlgKernelParams params);
template void DLL_PUBLIC launch_SplineAlgKernel<true, true, false> (unsigned int grid_size, unsigned int block_size, cudaStream_t stream, SplineAlgKernelParams params);
template void DLL_PUBLIC launch_SplineAlgKernel<true, true, true> (unsigned int grid_size, unsigned int block_size, cudaStream_t stream, SplineAlgKernelParams params);

void launch_AddNoiseKernel (unsigned int grid_size, unsigned int block_size, cudaStream_t stream, cuComplex *noise, cuComplex *signal, int num_samples)
{
  AddNoiseKernel<<<grid_size, block_size, 0, stream>>> (signal, noise, num_samples);
}

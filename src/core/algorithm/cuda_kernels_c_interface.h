#pragma once
#include "../Aperture.hpp"
#include "../export_macros.hpp"
#include <cuComplex.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cufft.h>

// Headers for all CUDA functionality accessible from C++

struct LUTProfileGeometry
{
  float r_min, r_max;
  float l_min, l_max;
  float e_min, e_max;
};

struct BaseParams
{
  float fs_hertz;       // temporal sampling frequency in hertz
  int num_time_samples; // number of samples in time signal
  float sound_speed;    // speed of sound in meters per second
  cuComplex *result;    // the output buffer (complex projected amplitudes)
  float demod_freq;     // complex demodulation frequency.
  int num_scatterers;   // number of scatterers
};

struct ProjectionParams : BaseParams
{
  float3 rad_dir;              // radial direction unit vector
  float3 lat_dir;              // lateral direction unit vector
  float3 ele_dir;              // elevational direction unit vector
  float3 origin;               // beam's origin
  float sigma_lateral;         // lateral beam width (for analyical beam profile)
  float sigma_elevational;     // elevational beam width (for analytical beam profile)
  cudaTextureObject_t lut_tex; // 3D texture object (for lookup-table beam profile)
  LUTProfileGeometry lut;
};

struct FixedAlgKernelParams : ProjectionParams
{
  float *point_xs; // pointer to device memory x components
  float *point_ys; // pointer to device memory y components
  float *point_zs; // pointer to device memory z components
  float *point_as; // pointer to device memory amplitudes
};

struct SplineAlgKernelParams : ProjectionParams
{
  float *control_xs;              // pointer to device memory x components
  float *control_ys;              // pointer to device memory y components
  float *control_zs;              // pointer to device memory z components
  float *control_as;              // pointer to device memory amplitudes
  int cs_idx_start;               // start index for spline evaluation sum (inclusive)
  int cs_idx_end;                 // end index for spline evaluation sum (inclusive)
  int NUM_SPLINES;                // number of splines in phantom (i.e. number of scatterers)
  int eval_basis_offset_elements; // memory offset (for different CUDA streams)
};

template <typename T>
void DLL_PUBLIC launch_MemsetKernel (unsigned int grid_size, unsigned int block_size, cudaStream_t stream, T *ptr, T value, int num_elements);

template <bool normalize>
void DLL_PUBLIC launch_MultiplyFftKernel (unsigned int grid_size, unsigned int block_size, cudaStream_t stream, cufftComplex *time_proj_fft, const cufftComplex *filter_fft, int num_samples);

void DLL_PUBLIC launch_DemodulateKernel (unsigned int grid_size, unsigned int block_size, cudaStream_t stream, cuComplex *signal, float w, int max_index, int radial_decimation);

void DLL_PUBLIC launch_ScaleSignalKernel (unsigned int grid_size, unsigned int block_size, cudaStream_t stream, cufftComplex *signal, float factor, int num_samples);

template <bool A, bool B, bool C>
void DLL_PUBLIC launch_FixedAlgKernel (unsigned int grid_size, unsigned int block_size, cudaStream_t stream, FixedAlgKernelParams params);

// Upload data to constant memory [workaround the fact that constant memory cannot be allocated dynamically]
// Returns false on error.
bool DLL_PUBLIC splineAlg1_updateConstantMemory (float *src_ptr, size_t num_bytes);

void DLL_PUBLIC launch_RenderSplineKernel (unsigned int grid_size, unsigned int block_size, cudaStream_t stream,
                                           const float *control_xs,
                                           const float *control_ys,
                                           const float *control_zs,
                                           float *rendered_xs,
                                           float *rendered_ys,
                                           float *rendered_zs,
                                           int cs_idx_start,
                                           int cs_idx_end,
                                           int NUM_SPLINES);

void DLL_PUBLIC launch_SliceLookupTable (unsigned int grid_size0, unsigned int grid_size1, unsigned int block_size, cudaStream_t stream,
                                         float3 origin,
                                         float3 dir0,
                                         float3 dir1,
                                         float *output,
                                         cudaTextureObject_t lut_tex);

// Returns false on error.
bool DLL_PUBLIC splineAlg2_updateConstantMemory (float *src, size_t count, size_t offset, cudaMemcpyKind kind, cudaStream_t stream);

template <bool A, bool B, bool C>
void DLL_PUBLIC launch_SplineAlgKernel (unsigned int grid_size, unsigned int block_size, cudaStream_t stream, SplineAlgKernelParams params);

void DLL_PUBLIC launch_AddNoiseKernel (unsigned int grid_size, unsigned int block_size, cudaStream_t stream, cuComplex *noise, cuComplex *signal, int num_samples);

#pragma once
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuComplex.h>
#include <cufft.h>
#include "../export_macros.hpp"

// Headers for all CUDA functionality accessible from C++

struct DLL_PUBLIC LUTProfileGeometry {
    float r_min, r_max;
    float l_min, l_max;
    float e_min, e_max;
};

struct DLL_PUBLIC FixedAlgKernelParams {
    float* point_xs;            // pointer to device memory x components
    float* point_ys;            // pointer to device memory y components
    float* point_zs;            // pointer to device memory z components
    float* point_as;            // pointer to device memory amplitudes
    float3 rad_dir;             // radial direction unit vector
    float3 lat_dir;             // lateral direction unit vector
    float3 ele_dir;             // elevational direction unit vector
    float3 origin;              // beam's origin
    float  fs_hertz;            // temporal sampling frequency in hertz
    int    num_time_samples;    // number of samples in time signal
    float  sigma_lateral;       // lateral beam width (for analyical beam profile)
    float  sigma_elevational;   // elevational beam width (for analytical beam profile)
    float  sound_speed;         // speed of sound in meters per second
    cuComplex* res;             // the output buffer (complex projected amplitudes)
    float  demod_freq;          // complex demodulation frequency.
    int    num_scatterers;      // number of scatterers
    cudaTextureObject_t lut_tex; // 3D texture object (for lookup-table beam profile)
    LUTProfileGeometry lut;
};

struct DLL_PUBLIC SplineAlgKernelParams {
    float* control_xs;                  // pointer to device memory x components
    float* control_ys;                  // pointer to device memory y components
    float* control_zs;                  // pointer to device memory z components
    float* control_as;                  // pointer to device memory amplitudes
    float3 rad_dir;                     // radial direction unit vector
    float3 lat_dir;                     // lateral direction unit vector
    float3 ele_dir;                     // elevational direction unit vector
    float3 origin;                      // beam's origin
    float  fs_hertz;                    // temporal sampling frequency in hertz
    int    num_time_samples;            // number of samples in time signal
    float  sigma_lateral;               // lateral beam width (for analyical beam profile)
    float  sigma_elevational;           // elevational beam width (for analytical beam profile)
    float  sound_speed;                 // speed of sound in meters per second
    int    cs_idx_start;                // start index for spline evaluation sum (inclusive)
    int    cs_idx_end;                  // end index for spline evaluation sum (inclusive)
    int    NUM_SPLINES;                 // number of splines in phantom (i.e. number of scatterers)
    cuComplex* res;                     // the output buffer (complex projected amplitudes)
    size_t eval_basis_offset_elements;  // memory offset (for different CUDA streams)
    float  demod_freq;                  // complex demodulation frequency.
    cudaTextureObject_t lut_tex;        // 3D texture object (for lookup-table beam profile) 
    LUTProfileGeometry lut;
};

template <typename T>
void DLL_PUBLIC launch_MemsetKernel(int grid_size, int block_size, cudaStream_t stream, T* ptr, T value, int num_elements);

void DLL_PUBLIC launch_MultiplyFftKernel(int grid_size, int block_size, cudaStream_t stream, cufftComplex* time_proj_fft, const cufftComplex* filter_fft, int num_samples);

void DLL_PUBLIC launch_DemodulateKernel(int grid_size, int block_size, cudaStream_t stream, cuComplex* signal, float w, int num_samples);

void DLL_PUBLIC launch_ScaleSignalKernel(int grid_size, int block_size, cudaStream_t stream, cufftComplex* signal, float factor, int num_samples);

template <bool A, bool B, bool C>
void DLL_PUBLIC launch_FixedAlgKernel(int grid_size, int block_size, cudaStream_t stream, FixedAlgKernelParams params);

// Upload data to constant memory [workaround the fact that constant memory cannot be allocated dynamically]
// Returns false on error.
bool DLL_PUBLIC splineAlg1_updateConstantMemory(float* src_ptr, size_t num_bytes);

void DLL_PUBLIC launch_RenderSplineKernel(int grid_size, int block_size, cudaStream_t stream,
                               const float* control_xs,
                               const float* control_ys,
                               const float* control_zs,
                               float* rendered_xs,
                               float* rendered_ys,
                               float* rendered_zs,
                               int cs_idx_start,
                               int cs_idx_end,
                               int NUM_SPLINES);

void DLL_PUBLIC launch_SliceLookupTable(int grid_size0, int grid_size1, int block_size, cudaStream_t stream,
                             float3 origin,
                             float3 dir0,
                             float3 dir1,
                             float* output,
                             cudaTextureObject_t lut_tex);

// Returns false on error.
bool DLL_PUBLIC splineAlg2_updateConstantMemory(float* src, size_t count, size_t offset, cudaMemcpyKind kind, cudaStream_t stream);

template <bool A, bool B, bool C>
void DLL_PUBLIC launch_SplineAlgKernel(int grid_size, int block_size, cudaStream_t stream, SplineAlgKernelParams params);

void DLL_PUBLIC launch_AddNoiseKernel(int grid_size, int block_size, cudaStream_t stream, cuComplex* noise, cuComplex* signal, int num_samples);

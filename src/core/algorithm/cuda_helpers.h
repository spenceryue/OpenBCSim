#pragma once
#include <chrono>
#include <cmath>
#include <cstring> // for std::memset
#include <cuda_runtime_api.h>
#include <driver_functions.h>
#include <driver_types.h>
#include <functional>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector_functions.h> // for make_float3() etc.

// Throws a std::runtime_error in case the return value is not cudaSuccess.
#define cudaErrorCheck(ans)                 \
  {                                         \
    cudaAssert ((ans), __FILE__, __LINE__); \
  }
inline void cudaAssert (cudaError_t code, const char *file, int line)
{
  if (code != cudaSuccess)
  {
    auto msg = std::string ("CUDA error (") + std::to_string (code) + "): " + std::string (cudaGetErrorString (code)) + ", FILE: " + std::string (file) + ", LINE: " + std::to_string (line);
    throw std::runtime_error (msg);
  }
}

// RAII-style wrapper for device memory.
template <class T>
class DeviceBufferRAII
{
public:
  using u_ptr = std::unique_ptr<DeviceBufferRAII<T>>;
  using s_ptr = std::shared_ptr<DeviceBufferRAII<T>>;

  explicit DeviceBufferRAII (size_t num_bytes)
  {
    cudaErrorCheck (cudaMalloc (&memory, num_bytes));
    num_bytes_allocated = num_bytes;
  }

  ~DeviceBufferRAII ()
  {
    cudaErrorCheck (cudaFree (memory));
  }

  T *data () const
  {
    return static_cast<T *> (memory);
  }

  size_t get_num_bytes () const
  {
    return num_bytes_allocated;
  }

  void copyFromAsync (const T *src, int num_elements, cudaStream_t cuda_stream = 0)
  {
    const auto bytes = num_elements * sizeof (T);
    if (bytes > num_bytes_allocated)
    {
      throw std::out_of_range ("Source bytes (" +
                               std::to_string (bytes) +
                               ") exceeds DeviceBufferRAII memory allocated (" +
                               std::to_string (num_bytes_allocated) +
                               ").");
    }
    cudaErrorCheck (cudaMemcpyAsync (memory, src, bytes, cudaMemcpyHostToDevice, cuda_stream));
  }

  void copyToAsync (T *dest, int num_elements, int start_idx = 0, cudaStream_t cuda_stream = 0)
  {
    const auto bytes = num_elements * sizeof (T);
    if (start_idx * sizeof (T) + bytes > num_bytes_allocated)
    {
      throw std::out_of_range ("Source bytes (" +
                               std::to_string (bytes) +
                               ") exceeds DeviceBufferRAII memory allocated (" +
                               std::to_string (num_bytes_allocated) +
                               ").");
    }
    cudaErrorCheck (cudaMemcpyAsync (dest, memory + start_idx, bytes, cudaMemcpyDeviceToHost, cuda_stream));
  }

  void copyFrom (const T *src, int num_elements, cudaStream_t cuda_stream = 0)
  {
    copyFromAsync (src, num_elements, cuda_stream);
    cudaErrorCheck (cudaDeviceSynchronize ());
  }

  void copyTo (T *dest, int num_elements, int start_idx = 0, cudaStream_t cuda_stream = 0)
  {
    copyToAsync (dest, start_idx, num_elements, cuda_stream);
    cudaErrorCheck (cudaDeviceSynchronize ());
  }

private:
  void *memory;
  size_t num_bytes_allocated;
};

// RAII wrapper for pinned host memory.
template <class T>
class HostPinnedBufferRAII
{
public:
  using u_ptr = std::unique_ptr<HostPinnedBufferRAII<T>>;
  using s_ptr = std::shared_ptr<HostPinnedBufferRAII<T>>;

  explicit HostPinnedBufferRAII (size_t num_bytes)
  {
    cudaErrorCheck (cudaMallocHost (&memory, num_bytes));
  }

  ~HostPinnedBufferRAII ()
  {
    cudaErrorCheck (cudaFreeHost (memory));
  }

  T *data () const
  {
    return static_cast<T *> (memory);
  }

private:
  void *memory;
};

// RAII-style CUDA timer.
class EventTimerRAII
{
public:
  explicit EventTimerRAII (cudaStream_t cuda_stream = 0)
      : cuda_stream (cuda_stream)
  {
    cudaErrorCheck (cudaEventCreate (&begin_event));
    cudaErrorCheck (cudaEventCreate (&end_event));
  }

  ~EventTimerRAII ()
  {
    cudaErrorCheck (cudaEventDestroy (begin_event));
    cudaErrorCheck (cudaEventDestroy (end_event));
  }

  // restart the timer
  void restart ()
  {
    cudaErrorCheck (cudaEventRecord (begin_event, cuda_stream));
  }

  // return milliseconds since start
  float stop ()
  {
    float res_millisec;
    cudaErrorCheck (cudaEventRecord (end_event, cuda_stream));
    cudaErrorCheck (cudaEventSynchronize (end_event));
    cudaErrorCheck (cudaEventElapsedTime (&res_millisec, begin_event, end_event));

    return res_millisec;
  }

private:
  cudaEvent_t begin_event;
  cudaEvent_t end_event;
  cudaStream_t cuda_stream;
};

class CudaStreamRAII
{
public:
  using u_ptr = std::unique_ptr<CudaStreamRAII>;
  using s_ptr = std::shared_ptr<CudaStreamRAII>;

  explicit CudaStreamRAII ()
  {
    cudaErrorCheck (cudaStreamCreate (&stream));
  }

  ~CudaStreamRAII ()
  {
    cudaErrorCheck (cudaStreamDestroy (stream));
  }

  cudaStream_t get ()
  {
    return stream;
  }

private:
  cudaStream_t stream;
};

// selected math
inline __host__ __device__ float3 operator+ (float3 a, float3 b)
{
  return make_float3 (a.x + b.x, a.y + b.y, a.z + b.z);
}

inline __host__ __device__ float3 operator- (float3 a, float3 b)
{
  return make_float3 (a.x - b.x, a.y - b.y, a.z - b.z);
}

inline __host__ __device__ float dot (float3 a, float3 b)
{
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline __host__ __device__ float norm (float3 a)
{
#ifdef __CUDACC__
  return norm3df (a.x, a.y, a.z);
#else
  return sqrtf (dot (a, a));
#endif
}

inline __host__ __device__ float dist (float3 a, float3 b)
{
  return norm (a - b);
}

inline __host__ __device__ float3 operator* (float3 a, float b)
{
  return make_float3 (a.x * b, a.y * b, a.z * b);
}

template <class T>
void fill_host_vector_uniform_random (T low, T high, size_t length, T *data)
{
  std::random_device rd;
  std::mt19937 gen (rd ());
  std::uniform_real_distribution<float> dis (low, high);
  for (size_t i = 0; i < length; i++)
  {
    data[i] = dis (gen);
  }
}

inline int round_up_div (int num, int den)
{
  return static_cast<int> (std::ceil (static_cast<float> (num) / den));
}

// 3D texture with tri-linear interpolation.
class DeviceBeamProfileRAII
{
public:
  using LogCallback = std::function<void(const std::string &)>;
  using u_ptr = std::unique_ptr<DeviceBeamProfileRAII>;
  using s_ptr = std::shared_ptr<DeviceBeamProfileRAII>;

  typedef struct TableExtent3D
  {
    TableExtent3D () : lateral (0), elevational (0), radial (0) {}
    TableExtent3D (size_t num_samples_lat, size_t num_samples_ele, size_t num_samples_rad)
        : lateral (num_samples_lat), elevational (num_samples_ele), radial (num_samples_rad) {}
    size_t lateral;
    size_t elevational;
    size_t radial;
  } TableExtent3D;

  DeviceBeamProfileRAII (const TableExtent3D &table_extent, std::vector<float> &host_input_buffer, LogCallback log_callback_fn = [](const std::string &) {})
      : texture_object (0),
        m_log_callback_fn (log_callback_fn)
  {
    auto channel_desc = cudaCreateChannelDesc (32, 0, 0, 0, cudaChannelFormatKindFloat);
    cudaExtent extent = make_cudaExtent (table_extent.lateral, table_extent.elevational, table_extent.radial);
    cudaErrorCheck (cudaMalloc3DArray (&cu_array_3d, &channel_desc, extent, 0));
    m_log_callback_fn ("DeviceBeamProfileRAII: Allocated 3D array");

    // copy input data from host to CUDA 3D array
    cudaMemcpy3DParms par_3d = {0};
    par_3d.srcPtr = make_cudaPitchedPtr (host_input_buffer.data (), table_extent.lateral * sizeof (float), table_extent.lateral, table_extent.elevational);
    par_3d.dstArray = cu_array_3d;
    par_3d.extent = extent;
    par_3d.kind = cudaMemcpyHostToDevice;
    cudaErrorCheck (cudaMemcpy3D (&par_3d));
    m_log_callback_fn ("DeviceBeamProfileRAII: Copied memory to device");

    // specify texture
    cudaResourceDesc res_desc;
    memset (&res_desc, 0, sizeof (res_desc));
    res_desc.resType = cudaResourceTypeArray;
    res_desc.res.array.array = cu_array_3d;

    // specify texture object parameters
    cudaTextureDesc tex_desc;
    memset (&tex_desc, 0, sizeof (tex_desc));
    tex_desc.normalizedCoords = 1;
    tex_desc.filterMode = cudaFilterModeLinear;

    // use border to pad with zeros outsize
    tex_desc.addressMode[0] = cudaAddressModeBorder;
    tex_desc.addressMode[1] = cudaAddressModeBorder;
    tex_desc.addressMode[2] = cudaAddressModeBorder;
    tex_desc.readMode = cudaReadModeElementType;

    cudaErrorCheck (cudaCreateTextureObject (&texture_object, &res_desc, &tex_desc, NULL));
    m_log_callback_fn ("DeviceBeamProfileRAII: Created texture object");
  }

  cudaTextureObject_t get ()
  {
    return texture_object;
  }

  ~DeviceBeamProfileRAII ()
  {
    cudaErrorCheck (cudaDestroyTextureObject (texture_object));
    m_log_callback_fn ("DeviceBeamProfileRAII: Destroyed texture object");
    cudaErrorCheck (cudaFreeArray (cu_array_3d));
    m_log_callback_fn ("DeviceBeamProfileRAII: Freed 3D array");
  }

private:
  cudaTextureObject_t texture_object;
  cudaArray *cu_array_3d;
  LogCallback m_log_callback_fn;
};

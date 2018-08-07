#pragma once
#include <type_traits>
#include <vector_functions.hpp>

template <class T>
using scalar2 = std::conditional_t<std::is_same_v<T, float>, float2, double2>;

template <class T>
using scalar3 = std::conditional_t<std::is_same_v<T, float>, float3, double3>;

#ifdef __CUDACC__

__device__ auto exp_pi_i (float a)
{
  float2 result;
  sincospif (a, &result.x, &result.y);
  return result;
}
__device__ auto exp_pi_i (double a)
{
  double2 result;
  sincospi (a, &result.x, &result.y);
  return result;
}

__device__ auto norm3d_s (float a, float b, float c)
{
  return norm3df (a, b, c);
}
__device__ auto norm3d_s (double a, double b, double c)
{
  return norm3d (a, b, c);
}

__device__ auto pow_s (float a, float b)
{
  return powf (a, b);
}
__device__ auto pow_s (double a, double b)
{
  return pow (a, b);
}

__device__ auto make_scalar3 (float a, float b, float c)
{
  return make_float3 (a, b, c);
}
__device__ auto make_scalar3 (double a, double b, double c)
{
  return make_double3 (a, b, c);
}

__device__ float3 operator+ (float3 a, float3 b)
{
  return make_float3 (a.x + b.x, a.y + b.y, a.z + b.z);
}
__device__ double3 operator+ (double3 a, double3 b)
{
  return make_double3 (a.x + b.x, a.y + b.y, a.z + b.z);
}
__device__ float2 operator+ (float2 a, float2 b)
{
  return make_float2 (a.x + b.x, a.y + b.y);
}
__device__ double2 operator+ (double2 a, double2 b)
{
  return make_double2 (a.x + b.x, a.y + b.y);
}

__device__ float3 operator- (float3 a, float3 b)
{
  return make_float3 (a.x - b.x, a.y - b.y, a.z - b.z);
}
__device__ double3 operator- (double3 a, double3 b)
{
  return make_double3 (a.x - b.x, a.y - b.y, a.z - b.z);
}
__device__ float2 operator- (float2 a, float2 b)
{
  return make_float2 (a.x - b.x, a.y - b.y);
}
__device__ double2 operator- (double2 a, double2 b)
{
  return make_double2 (a.x - b.x, a.y - b.y);
}

template <class T, std::enable_if_t<std::is_floating_point_v<T>, int> = 0>
__device__ float3 operator* (float3 a, T b)
{
  return make_float3 (a.x * b, a.y * b, a.z * b);
}
template <class T, std::enable_if_t<std::is_floating_point_v<T>, int> = 0>
__device__ double3 operator* (double3 a, T b)
{
  return make_double3 (a.x * b, a.y * b, a.z * b);
}
template <class T, std::enable_if_t<std::is_floating_point_v<T>, int> = 0>
__device__ float2 operator* (float2 a, T b)
{
  return make_float2 (a.x * b, a.y * b);
}
template <class T, std::enable_if_t<std::is_floating_point_v<T>, int> = 0>
__device__ double2 operator* (double2 a, T b)
{
  return make_double2 (a.x * b, a.y * b);
}

__device__ float2 operator* (float2 a, float2 b)
{
  return make_float2 (a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}
__device__ double2 operator* (double2 a, double2 b)
{
  return make_double2 (a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

__device__ float dot (float3 a, float3 b)
{
  return a.x * b.x + a.y * b.y + a.z * b.z;
}
__device__ double dot (double3 a, double3 b)
{
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ float norm (float3 a)
{
  return norm3d_s (a.x, a.y, a.z);
}
__device__ double norm (double3 a)
{
  return norm3d_s (a.x, a.y, a.z);
}

__device__ float dist (float3 a, float3 b)
{
  return norm (a - b);
}
__device__ double dist (double3 a, double3 b)
{
  return norm (a - b);
}
#endif

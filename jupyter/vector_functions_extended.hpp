#pragma once
#include "invoke_one.hpp"
#include <vector_functions.hpp>

template <class scalar_t>
auto make_scalar3 (scalar_t a, scalar_t b, scalar_t c) -> decltype (invokeOne (make_float3, make_double3, a, b, c))
{
  return invokeOne (make_float3, make_double3, a, b, c);
}

__device__ float3 operator+ (float3 a, float3 b)
{
  return make_float3 (a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ float3 operator- (float3 a, float3 b)
{
  return make_float3 (a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ float3 operator* (float3 a, float b)
{
  return make_float3 (a.x * b, a.y * b, a.z * b);
}

__device__ float dot (float3 a, float3 b)
{
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ float norm (float3 a)
{
  return norm3df (a.x, a.y, a.z);
}

__device__ float dist (float3 a, float3 b)
{
  return norm (a - b);
}

__device__ double3 operator+ (double3 a, double3 b)
{
  return make_double3 (a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ double3 operator- (double3 a, double3 b)
{
  return make_double3 (a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ double3 operator* (double3 a, double b)
{
  return make_double3 (a.x * b, a.y * b, a.z * b);
}

__device__ double dot (double3 a, double3 b)
{
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ double norm (double3 a)
{
  return norm3d (a.x, a.y, a.z);
}

__device__ double dist (double3 a, double3 b)
{
  return norm (a - b);
}

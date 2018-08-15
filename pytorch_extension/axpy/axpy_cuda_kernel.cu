#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <iostream>

template <class scalar_t>
__global__ void axpy (scalar_t a, scalar_t *x, scalar_t *y)
{
  y[threadIdx.x] = a * x[threadIdx.x];
}

template <class scalar_t>
void run_it (scalar_t a, scalar_t *x, scalar_t *y, size_t N)
{
  axpy<scalar_t><<<1, N>>> (a, x, y);
}

template void run_it (float a, float *x, float *y, size_t N);
template void run_it (double a, double *x, double *y, size_t N);

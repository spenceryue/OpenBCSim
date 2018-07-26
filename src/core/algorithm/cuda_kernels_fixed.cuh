#pragma once
#include "cuda_helpers.h" // for operator-
#include "cuda_kernels_c_interface.h"
#include <cuComplex.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

__global__ void SliceLookupTable (float3 origin,
                                  float3 dir0,
                                  float3 dir1,
                                  float *output,
                                  cudaTextureObject_t lut_tex);

template <bool use_elev_hack, bool use_arc_projection, bool use_phase_delay, bool use_lut>
__global__ void FixedAlgKernel (FixedAlgKernelParams params)
{
  const auto global_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (global_idx >= params.num_scatterers)
  {
    return;
  }

  const float3 point = make_float3 (params.point_xs[global_idx],
                                    params.point_ys[global_idx],
                                    params.point_zs[global_idx]) -
                       params.origin;

  // compute dot products
  auto radial_dist = dot (point, params.rad_dir);
  const auto lateral_dist = dot (point, params.lat_dir);
  const auto elev_dist = dot (point, params.ele_dir);

  // Use "arc projection" in the radial direction: use length of vector from
  // beam's origin to the scatterer with the same sign as the projection onto
  // the line.
  if (use_arc_projection)
  {
    if (use_elev_hack)
    {
      // This scatterer point is actually the result of splitting an
      // original scatterer point into several points across the elevational
      // direction with the elevation (y-component) representing the distance
      // to a different aperture element of the transducer.

      // Divide by 2 to make up for multiplying by 2 in radial_index formula
      // The y-component is replaced with just the beam-origin y-component
      // because the true scatterer is assumed to lie in the plane.
      const float dist = (norm3df (point.x, params.origin.y, point.z) + elev_dist) / 2;
      radial_dist = copysignf (dist, radial_dist);
    }
    else
    {
      radial_dist = copysignf (norm (point), radial_dist);
    }
  }

  float weight;
  if (use_elev_hack)
  {
    // Any beam profile characteristics or transmit apodization was already
    // incorporated into the amplitude of the scatterer point.
    weight = 1;
  }
  else if (use_lut)
  {
    // Compute weight from lookup-table and radial_dist, lateral_dist, and elev_dist
    weight = ComputeWeightLUT (params.lut_tex, radial_dist, lateral_dist, elev_dist, params.lut);
  }
  else
  {
    weight = ComputeWeightAnalytical (params.sigma_lateral,
                                      params.sigma_elevational,
                                      radial_dist,
                                      lateral_dist,
                                      elev_dist);
  }

  const int radial_index = static_cast<int> (params.fs_hertz * 2.0f * radial_dist / params.sound_speed + 0.5f);

  if (radial_index >= 0 && radial_index < params.num_time_samples)
  {
    if (use_phase_delay)
    {
      // handle sub-sample displacement with a complex phase
      const auto true_index = params.fs_hertz * 2.0 * radial_dist / params.sound_speed;
      const auto ss_delay = (radial_index - true_index) / params.fs_hertz;
      const auto complex_phase = 6.283185307179586 * params.demod_freq * ss_delay;

      // exp(i*theta) = cos(theta) + i*sin(theta)
      float sin_value, cos_value;
      sincosf (complex_phase, &sin_value, &cos_value);

      const auto w = weight * params.point_as[global_idx];
      atomicAdd (&(params.res[radial_index].x), w * cos_value);
      atomicAdd (&(params.res[radial_index].y), w * sin_value);
    }
    else
    {
      atomicAdd (&(params.res[radial_index].x), weight * params.point_as[global_idx]);
    }
  }
}

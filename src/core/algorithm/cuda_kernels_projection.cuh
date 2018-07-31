#pragma once
#include "cuda_helpers.h" // for operator-
#include "cuda_kernels_c_interface.h"
#include "cuda_kernels_common.cuh"
#include <cuComplex.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <math_functions.h> // for copysignf

// Functionality shared between FixedAlgKernel and SplineAlgKernel.
// Processes one beam (scanline) per kernel launch.
template <bool use_arc_projection, bool use_phase_delay, bool use_lut>
__device__ __forceinline__ void ProjectionAlg (const ProjectionParams params,
                                               const float3 scatterer_point,
                                               const float amplitude)
{
  // Convert scatterer position to normalized coordinates centered at the beam origin.
  const float3 point = scatterer_point - params.origin;

  // Compute dot products.
  auto radial_dist = dot (point, params.rad_dir);
  const auto lateral_dist = dot (point, params.lat_dir);
  const auto elev_dist = dot (point, params.ele_dir);

  if (use_arc_projection)
  {
    // Use "arc projection" in the radial direction: use length of vector from
    // beam's origin to the scatterer with the same sign as the projection onto
    // the line.
    radial_dist = copysignf (norm (point), radial_dist);
  }

  float weight = 1.0f;
  if (use_lut)
  {
    // Compute weight from lookup-table and radial_dist, lateral_dist, and elev_dist.
    weight *= ComputeWeightLUT (params.lut_tex, radial_dist, lateral_dist, elev_dist, params.lut);
  }
  else
  {
    // Compute weight analytically.
    weight *= ComputeWeightAnalytical (params.sigma_lateral, params.sigma_elevational, radial_dist, lateral_dist, elev_dist);
  }

  const int radial_index = static_cast<int> (params.fs_hertz * 2.0f * radial_dist / params.sound_speed + 0.5f);

  if (radial_index >= 0 && radial_index < params.num_time_samples)
  {
    if (use_phase_delay)
    {
      // Handle sub-sample displacement with a complex phase.
      const auto true_index = params.fs_hertz * 2.0f * radial_dist / params.sound_speed;
      const float ss_delay = (radial_index - true_index) / params.fs_hertz;
      const float complex_phase = 6.283185307179586 * params.demod_freq * ss_delay;

      // Calculate exp(i*theta) = cos(theta) + i*sin(theta).
      float sin_value, cos_value;
      sincosf (complex_phase, &sin_value, &cos_value);

      atomicAdd (&(params.result[radial_index].x), weight * cos_value * amplitude);
      atomicAdd (&(params.result[radial_index].y), weight * sin_value * amplitude);
    }
    else
    {
      atomicAdd (&(params.result[radial_index].x), weight * amplitude);
    }
  }
}

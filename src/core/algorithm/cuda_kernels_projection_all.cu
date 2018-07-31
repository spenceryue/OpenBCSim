#include "cuda_helpers.h" // for dist
#include "cuda_kernels_projection_all.cuh"
#include <complex>
#include <device_launch_parameters.h>

__global__ void ProjectionAllKernel (const ProjectionAllParams params)
{
  /*
    Grid/Block Index Pattern:
    =========================
    X-dimension: scatterer index
    Y-dimension: transmit aperture index
    Z-dimension: receive aperture index

    Steps:
    ======
    1. Calculate scatterer index.
    2. Get scatterer position.
    3. Get transmit origin, delay, and apodization (x,y,z,d,a) via grid Y-dimension.
    4. Get receive origin, delay, and apodization (x,y,z,d,a) via grid Z-dimension.
    5. Calculate total travel distance: distance(transmit, scatterer) + distance(scatterer, receive).
    6. Calculate projection index from total travel time + delays.
    7. Calculate phase factor.
    8. Calculate depth attenuation. (Use travel distance, not time because time has delays)
    9. Calculate weight: (transmit apodization) * (receive apodization) * (depth attenuation) * (phase factor).
    10. Get index into result buffer from projection index and receive index (grid Z dimension).
    11. Add (scatterer amplitude) * weight into projection buffer.

    Data Dependencies by Step:
    ==========================
    1. Kernel launch params
    2. FixedAlgKernelParams (point_xs, point_ys, point_zs, point_as)
    3. Aperture (x,y,z), Kernel launch params
    4. Aperture (x,y,z), Kernel launch params
    5. ---
    6. BaseParams (sound_speed), Aperture (delay)
    7. BaseParams (fs_hertz)
    8. BaseParams (attenuation)
    9. ---
    10. Kernel launch params, BaseParams (num_time_samples)
    11. ---
  */

  // Step 1.
  const auto scatterer_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (scatterer_idx >= params.num_scatterers)
  {
    return;
  }

  // Step 2.
  const auto scatterer = make_float3 (params.point_xs[scatterer_idx],
                                      params.point_ys[scatterer_idx],
                                      params.point_zs[scatterer_idx]);

  // Step 3.
  const auto transmit_idx = blockIdx.y;
  const auto transmit = make_float3 (params.transmit.x[transmit_idx],
                                     params.transmit.y[transmit_idx],
                                     params.transmit.z[transmit_idx]);

  // Step 4.
  const auto receive_idx = blockIdx.z;
  const auto receive = make_float3 (params.receive.x[receive_idx],
                                    params.receive.y[receive_idx],
                                    params.receive.z[receive_idx]);

  // Step 5.
  const auto travel_distance = dist (transmit, scatterer) + dist (scatterer, receive);

  // Step 6.
  const auto transmit_delay = params.transmit.delay[transmit_idx];
  const auto receive_delay = params.receive.delay[receive_idx];
  const auto delay = transmit_delay + receive_delay;
  const auto fractional_idx = params.fs_hertz * travel_distance / params.sound_speed;
  const auto projection_idx = static_cast<int> (fractional_idx + delay + 0.5f);

  // Step 7.
  const auto subsample_delay = (projection_idx - fractional_idx) / params.fs_hertz;
  using namespace std::complex_literals;
  const auto PI = std::acos (-1);
  const auto complex_phase = 2 * PI * params.demod_freq * subsample_delay * 1i;
  const std::complex<float> phase_factor = std::exp (complex_phase);

  // Step 8.
  const auto powers_of_ten = params.attenuation / 10;
  const auto frequency_MHz = params.demod_freq / 1e6;
  const auto travel_distance_cm = travel_distance * 100;
  const auto depth_attenuation = powf (10, -powers_of_ten * frequency_MHz * travel_distance_cm);

  // Step 9.
  const auto transmit_apodization = params.transmit.apodization[transmit_idx];
  const auto receive_apodization = params.receive.apodization[receive_idx];
  const auto weight = transmit_apodization * receive_apodization * depth_attenuation * phase_factor;

  // Step 10.
  const auto result_idx = params.num_time_samples * receive_idx + projection_idx;

  // Step 11.
  const auto amplitude = params.point_as[scatterer_idx];
  const auto result = weight * amplitude;
  atomicAdd (&params.result[result_idx].x, result.real ());
  atomicAdd (&params.result[result_idx].y, result.imag ());
}

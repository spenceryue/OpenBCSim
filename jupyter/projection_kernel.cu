#include "definitions.hpp"
#include "vector_functions_extended.hpp"
#include <complex>
#include <device_launch_parameters.h>

template <class scalar_t>
__global__ void projection_kernel (const Simulation<scalar_t> args, std::complex<scalar_t> *__restrict__ output)
{
  /*
    Grid/Block Index Pattern:
    =========================
    X-dimension: scatterer index
    Y-dimension: transmitter aperture index
    Z-dimension: receiver aperture index
  */

  // 1. Calculate scatterer index.
  const auto scatterer_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (scatterer_idx >= args.num_scatterers)
  {
    return;
  }

  // 2. Get scatterer position.
  const auto scatterer = make_scalar3 (args.scatterer_x[scatterer_idx],
                                       args.scatterer_y[scatterer_idx],
                                       args.scatterer_z[scatterer_idx]);

  // 3. Get transmitter position (x,y,z) via grid Y-dimension.
  const auto transmitter_idx = blockIdx.y;
  const auto transmitter = make_scalar3 (args.transmitter.x[transmitter_idx],
                                         args.transmitter.y[transmitter_idx],
                                         args.transmitter.z[transmitter_idx]);

  // 4. Get receiver position (x,y,z) via grid Z-dimension.
  const auto receiver_idx = blockIdx.z;
  const auto receiver = make_scalar3 (args.receiver.x[receiver_idx],
                                      args.receiver.y[receiver_idx],
                                      args.receiver.z[receiver_idx]);

  // 5. Calculate total travel distance: distance(transmitter, scatterer) + distance(scatterer, receiver).
  const auto travel_distance = dist (transmitter, scatterer) + dist (scatterer, receiver);

  // 6. Calculate the transmitter and receiver element indices (as opposed to the sub-element indices above)
  const auto transmitter_element_idx = transmitter_idx / (args.transmitter.num_subdivisions);
  const auto receiver_element_idx = receiver_idx / (args.receiver.num_subdivisions);

  // 7. Calculate projection index from total travel time + delays.
  const auto transmitter_delay = args.transmitter.delay[transmitter_element_idx];
  const auto receiver_delay = args.receiver.delay[receiver_element_idx];
  const auto delay = transmitter_delay + receiver_delay;
  const auto fractional_idx = args.sampling_frequency * travel_distance / args.speed_of_sound;
  const auto projection_idx = static_cast<int> (fractional_idx + delay + 0.5);

  // 8. Calculate phase factor.
  const auto subsample_delay = (projection_idx - fractional_idx) / args.sampling_frequency;
  using namespace std::complex_literals;
  const auto PI = std::acos (-1);
  const auto complex_phase = 2.0 * PI * args.receiver.center_frequency * subsample_delay * 1i;
  const std::complex<float> phase_factor = std::exp (complex_phase);

  // 9. Calculate depth attenuation. (Use travel distance, not time because time has delays)
  const auto powers_of_ten = args.attenuation / 20;
  const auto frequency_MHz = args.transmitter.center_frequency / 1e6;
  const auto travel_distance_cm = travel_distance * 100;
  const auto depth_attenuation = powf (10, -powers_of_ten * frequency_MHz * travel_distance_cm);

  // 10. Calculate weight: (transmitter apodization) * (receiver apodization) * (depth attenuation) * (phase factor).
  const auto transmitter_apodization = args.transmitter.apodization[transmitter_element_idx];
  const auto receiver_apodization = args.receiver.apodization[receiver_element_idx];
  const auto weight = transmitter_apodization * receiver_apodization * depth_attenuation * phase_factor;

  // 11. Get index into output buffer from projection index and receiver index (grid Z dimension).
  const auto output_idx = args.num_time_samples * receiver_idx + projection_idx;

  // 12. Add (scatterer amplitude) * weight into output buffer.
  const auto amplitude = args.scatterer_amplitude[scatterer_idx];
  const auto output = weight * amplitude;
  atomicAdd (&output[output_idx].x, output.real ());
  atomicAdd (&output[output_idx].y, output.imag ());
}

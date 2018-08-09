#include "openbcsim_kernel.cuh"
#include "vector_functions_extended.hpp"

template <class scalar_t>
__global__ void projection_kernel (const Simulation<scalar_t> args, scalar_t *__restrict__ output_buffer)
{
  /*
    Grid Dimensions Computation Layout:
    =========================
    blockIdx.x: scatterer index-base
      blockDim.x: scatterer index-base-stride
      threadIdx.x: scatterer index-offset
    blockIdx.y: transmitter sub-element index
    blockIdx.z: receiver sub-element index
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
  const scalar_t travel_distance = dist (transmitter, scatterer) + dist (scatterer, receiver);

  const double two_sigma_squared = 2.0f * .001 * .001;
  const double lateral_dist = scatterer.x - receiver.x;
  const double w = exp (-lateral_dist * lateral_dist / two_sigma_squared);

  const auto projection_idx = args.sampling_frequency * travel_distance / args.speed_of_sound;
  const auto output_idx = 2 * (args.num_time_samples * blockIdx.z + (static_cast<int> (projection_idx + .5) % args.num_time_samples));
  const scalar_t value = w * args.scatterer_amplitude[scatterer_idx] * args.scatterer_x[scatterer_idx];
  // atomicAdd (&output_buffer[output_idx + 0], scatterer_idx);
  atomicAdd (&output_buffer[output_idx + 0], value);

  /*
  // 6. Calculate the transmitter and receiver element indices (as opposed to the sub-element indices above)
  const auto transmitter_element_idx = transmitter_idx / (args.transmitter.num_subdivisions);
  const auto receiver_element_idx = receiver_idx / (args.receiver.num_subdivisions);

  // 7. Calculate projection index from total travel time + delays.
  const scalar_t transmitter_delay = args.transmitter.delay[transmitter_element_idx];
  const scalar_t receiver_delay = args.receiver.delay[receiver_element_idx];
  const scalar_t delay = transmitter_delay + receiver_delay;
  const scalar_t fractional_idx = args.sampling_frequency * travel_distance / args.speed_of_sound;
  // Need to dot with radial unit vector for sign... (TODO)
  const auto projection_idx = static_cast<int> (fractional_idx + delay + 0.5);

  // 8. Return early if projection lands outside scan region.
  // Scatterer is either too deep or "behind" the plane of the transducer element.
  if (projection_idx < 0 || projection_idx >= args.num_time_samples)
    return;

  // 9 Calculate phase factor.
  const scalar_t subsample_delay = (projection_idx - fractional_idx) / args.sampling_frequency;
  const scalar_t complex_phase = 2.0 * args.receiver.center_frequency * subsample_delay;
  const scalar2<scalar_t> phase_factor = exp_pi_i (complex_phase);

  // 10 Calculate depth attenuation. (Use travel distance, not time because time has delays)
  const scalar_t powers_of_ten = args.attenuation / 20;
  const scalar_t frequency_MHz = args.transmitter.center_frequency / 1e6;
  const scalar_t travel_distance_cm = travel_distance * 100;
  const scalar_t depth_attenuation = pow_s (10, -powers_of_ten * frequency_MHz * travel_distance_cm);

  // 11. Calculate weight: (transmitter apodization) * (receiver apodization) * (depth attenuation) * (phase factor).
  const scalar_t transmitter_apodization = args.transmitter.apodization[transmitter_element_idx];
  const scalar_t receiver_apodization = args.receiver.apodization[receiver_element_idx];
  const scalar2<scalar_t> weight = phase_factor * transmitter_apodization * receiver_apodization * depth_attenuation;

  // 12. Get index into output buffer from projection index and receiver index (grid Z dimension).
  // Note: Factor of 2 is to make room for real and imag components.
  const auto output_idx = 2 * (args.num_time_samples * receiver_idx + projection_idx);

  // 13. Add (scatterer amplitude) * weight into output buffer.
  const scalar_t amplitude = args.scatterer_amplitude[scatterer_idx];
  const scalar2<scalar_t> output = weight * amplitude;
  atomicAdd (&output_buffer[output_idx + 0], output.x);
  atomicAdd (&output_buffer[output_idx + 1], output.y);*/
}

template <class scalar_t>
void launch_projection_kernel (const Simulation<scalar_t> &args, scalar_t *output_buffer,
                               dim3 grid, dim3 block, size_t shared_bytes, cudaStream_t stream)
{
  projection_kernel<scalar_t><<<grid, block, shared_bytes, stream>>> (args, output_buffer);
}

template void launch_projection_kernel (const Simulation<float> &args, float *output_buffer,
                                        dim3 grid, dim3 block, size_t shared_bytes, cudaStream_t stream);
template void launch_projection_kernel (const Simulation<double> &args, double *output_buffer,
                                        dim3 grid, dim3 block, size_t shared_bytes, cudaStream_t stream);

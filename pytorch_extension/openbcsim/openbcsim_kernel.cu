#include "openbcsim_kernel.cuh"
#include "vector_functions_extended.cuh"

template <class scalar_t>
__device__ __forceinline__ void do_time_projection (const Simulator<scalar_t> &args, scalar_t *RESTRICT output_buffer,
                                                    scalar_t amplitude, scalar_t receiver_distance,
                                                    scalar_t receiver_delay, scalar_t receiver_apodization,
                                                    unsigned receiver_idx, scalar_t transmit_distance,
                                                    scalar_t transmitter_delay, scalar_t transmitter_apodization)
{
  // 9. Calculate total travel distance.
  const scalar_t travel_distance = transmit_distance + receiver_distance;

  // 10. Calculate projection index from total travel time + delays.
  const scalar_t delay = transmitter_delay + receiver_delay;
  const scalar_t fractional_idx = args.sampling_frequency * travel_distance / args.speed_of_sound;
  const unsigned projection_idx = static_cast<unsigned> (fractional_idx + delay + 0.5);

  // 11. Skip if projection lands outside scan region (i.e. scatterer is too deep).
  /*
    Note: We do not check if a scatterer lies "behind" the transducer.
    It is assumed the transmitter and receiver lie in the nonpositive-z half-region
    and that the scatterers all lie in the positive-z half-region.
  */
  if (projection_idx >= args.num_time_samples)
    return;

  // 12. Calculate phase factor.
  const scalar_t subsample_delay = (projection_idx - fractional_idx) / args.sampling_frequency;
  const scalar_t complex_phase = 2.0 * args.receiver.center_frequency * subsample_delay;
  const scalar2<scalar_t> phase_factor = exp_pi_i (complex_phase);

  // 13. Calculate depth attenuation. (Use travel distance, not time because time has delays)
  const scalar_t powers_of_ten = args.attenuation / 20;
  const scalar_t frequency_MHz = args.transmitter.center_frequency / 1e6;
  const scalar_t travel_distance_cm = travel_distance * 100;
  const scalar_t depth_attenuation = pow_s (10, -powers_of_ten * frequency_MHz * travel_distance_cm);

  // 14. Calculate weight: (transmitter apodization) * (receiver apodization) * (depth attenuation) * (phase factor).
  const scalar2<scalar_t> weight = phase_factor * transmitter_apodization * receiver_apodization * depth_attenuation;

  // 15. Get index into output buffer from projection index and receiver index (grid Y-dimension).
  // Note: Factor of 2 is to make room for real and imag components.
  const unsigned output_idx = 2 * (args.num_time_samples * receiver_idx + projection_idx);

  // 16. Add (scatterer amplitude) * weight into output buffer.
  const scalar2<scalar_t> output = weight * amplitude;
  atomicAdd (&output_buffer[output_idx + 0], output.x);
  atomicAdd (&output_buffer[output_idx + 1], output.y);
}

template <class scalar_t>
__global__ void projection_kernel (const Simulator<scalar_t> args, scalar_t *RESTRICT output_buffer)
{
  /*
    Algorithm
    ===========================================================
    Loop over scatterers (blockIdx.x, blockDim.x, threadIdx.x):
      Loop over receiver sub-elements (blockIdx.y):
        Loop over transmitter sub-elements (blockIdx.z):
          Do time projection

    Note: The for-loops below follow the grid-stride-loop convention.
    https://devblogs.nvidia.com/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/
  */

  // 1. Calculate scatterer index.
  for (unsigned scatterer_idx = blockIdx.x * blockDim.x + threadIdx.x;
       scatterer_idx < args.num_scatterers;
       scatterer_idx += gridDim.x * blockDim.x)
  {
    // 2. Get scatterer position and amplitude.
    const auto scatterer = make_scalar3 (args.scatterer_x[scatterer_idx],
                                         args.scatterer_y[scatterer_idx],
                                         args.scatterer_z[scatterer_idx]);
    const scalar_t amplitude = args.scatterer_amplitude[scatterer_idx];

    // 3. Get receiver position (x,y,z) via grid Y-dimension.
    // Note: `receiver_idx` is the sub-element index (as opposed to element index (below)
    for (unsigned receiver_idx = blockIdx.y;
         receiver_idx < args.receiver.num_subelements;
         receiver_idx += gridDim.y)
    {
      const auto receiver = make_scalar3 (args.receiver.x[receiver_idx],
                                          args.receiver.y[receiver_idx],
                                          args.receiver.z[receiver_idx]);

      // 4. Calculate receive distance
      const scalar_t receiver_distance = dist (scatterer, receiver);

      // 5. Get receiver element index, delay, and apodization.
      // Note: `receiver_element_idx` is the element index.
      const unsigned receiver_element_idx = receiver_idx / (args.receiver.division_factor);
      const scalar_t receiver_delay = args.receiver.delay[receiver_element_idx];
      const scalar_t receiver_apodization = args.receiver.apodization[receiver_element_idx];

      // 6. Get transmitter position (x,y,z) via loop index.
      // Note: `transmitter_idx` is the sub-element index (as opposed to element index (below)
      for (unsigned transmitter_idx = blockIdx.z;
           transmitter_idx < args.transmitter.num_subelements;
           transmitter_idx += gridDim.z)
      {
        const auto transmitter = make_scalar3 (args.transmitter.x[transmitter_idx],
                                               args.transmitter.y[transmitter_idx],
                                               args.transmitter.z[transmitter_idx]);

        // 7. Calculate transmit distance.
        const scalar_t transmit_distance = dist (transmitter, scatterer);

        // 8. Get transmitter element index, delay, and apodization.
        // Note: `transmitter_element_idx` is the element index.
        const unsigned transmitter_element_idx = transmitter_idx / (args.transmitter.division_factor);
        const scalar_t transmitter_delay = args.transmitter.delay[transmitter_element_idx];
        const scalar_t transmitter_apodization = args.transmitter.apodization[transmitter_element_idx];

        do_time_projection<scalar_t> (args, output_buffer, amplitude, receiver_distance, receiver_delay,
                                      receiver_apodization, receiver_idx, transmit_distance, transmitter_delay,
                                      transmitter_apodization);
      }
    }
  }
}

template <class scalar_t>
void launch_projection_kernel (const Simulator<scalar_t> &args, scalar_t *output_buffer,
                               dim3 grid, dim3 block, size_t shared_bytes, cudaStream_t stream)
{
  cudaFuncSetCacheConfig (projection_kernel<scalar_t>, cudaFuncCachePreferL1);
  projection_kernel<scalar_t><<<grid, block, shared_bytes, stream>>> (args, output_buffer);
}

template <class T>
cudaError cuda_malloc_managed (T **pointer_to_pointer, size_t bytes)
{
  // Wrapper for cudaMallocManaged().
  // "To use these functions, your application needs to be compiled with the nvcc compiler."
  // https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__HIGHLEVEL.html
  return cudaMallocManaged (pointer_to_pointer, bytes);
}

template void launch_projection_kernel (const Simulator<float> &args, float *output_buffer,
                                        dim3 grid, dim3 block, size_t shared_bytes, cudaStream_t stream);
template void launch_projection_kernel (const Simulator<double> &args, double *output_buffer,
                                        dim3 grid, dim3 block, size_t shared_bytes, cudaStream_t stream);
template cudaError cuda_malloc_managed (float **pointer_to_pointer, size_t bytes);
template cudaError cuda_malloc_managed (double **pointer_to_pointer, size_t bytes);

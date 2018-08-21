#include "openbcsim_kernel.cuh"
#include "vector_functions_extended.cuh"

// Forward declaration
template <class scalar_t>
__device__ __forceinline__ void do_time_projection (const Simulator<scalar_t> &args,
                                                    scalar_t *RESTRICT output_buffer,
                                                    scalar_t receiver_distance,
                                                    scalar_t receiver_delay,
                                                    scalar_t receiver_apodization,
                                                    scalar_t transmitter_distance,
                                                    scalar_t transmitter_delay,
                                                    scalar_t transmitter_apodization,
                                                    scalar_t amplitude,
                                                    unsigned output_idx_0,
                                                    unsigned output_idx_1);

template <class scalar_t>
__global__ void projection_kernel (const Simulator<scalar_t> args, scalar_t *RESTRICT output_buffer)
{
  /*
    Algorithm
    ===========================================================
      Loop over scatterers (blockIdx.x, blockDim.x, threadIdx.x):
        Loop over receiver sub-elements (blockIdx.y):
          Loop over transmitter sub-elements (blockIdx.z):
            Loop over scans (args.transmitter.num_scans)
              Do time projection

    Note: The 3 outermost for-loops below follow the grid-stride-loop convention.
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
    // Note: `receiver_idx` is the sub-element index (as opposed to element index below).
    for (unsigned receiver_idx = blockIdx.y;
         receiver_idx < args.receiver.num_subelements;
         receiver_idx += gridDim.y)
    {
      const auto receiver = make_scalar3 (args.receiver.x[receiver_idx],
                                          args.receiver.y[receiver_idx],
                                          args.receiver.z[receiver_idx]);

      // 4. Calculate receive distance
      const scalar_t receiver_distance = dist (scatterer, receiver);

      // 5. Get receiver element index.
      // Note: `receiver_element_idx` is the element index (as opposed to sub-element index above).
      const unsigned receiver_element_idx = receiver_idx / (args.receiver.subdivision_factor);

      // 6. Calculate index-offset into `output_buffer[][receiver_element_idx][][]` (second dimension).
      const unsigned output_idx_1 = (2 * args.num_time_samples) * receiver_element_idx;

      // 7. Get transmitter position (x,y,z) via loop index.
      // Note: `transmitter_idx` is the sub-element index (as opposed to element index below).
      for (unsigned transmitter_idx = blockIdx.z;
           transmitter_idx < args.transmitter.num_subelements;
           transmitter_idx += gridDim.z)
      {
        const auto transmitter = make_scalar3 (args.transmitter.x[transmitter_idx],
                                               args.transmitter.y[transmitter_idx],
                                               args.transmitter.z[transmitter_idx]);

        // 8. Calculate transmit distance.
        const scalar_t transmitter_distance = dist (transmitter, scatterer);

        // 9. Get transmitter element index.
        // Note: `transmitter_element_idx` is the element index (as opposed to sub-element index above).
        const unsigned transmitter_element_idx = transmitter_idx / (args.transmitter.subdivision_factor);

        for (unsigned scan_idx = 0;
             scan_idx < args.transmitter.num_scans;
             scan_idx++)
        {
          // 10. Calculate index-offset into `output_buffer[scan_idx][][][]` (first dimension).
          const unsigned output_idx_0 = (2 * args.num_time_samples * args.receiver.num_elements) * scan_idx;

          // 11. Get receiver delay, apodization
          // e.g. `receiver_delay[scan_idx % args.receiver.num_scans][receiver_element_idx]`.
          // Note: The scan_idx dimension modulus allows the receiver to have fewer scans than the transmitter.
          const unsigned i = (args.receiver.num_elements) * (scan_idx % args.receiver.num_scans) + receiver_element_idx;
          const scalar_t receiver_delay = args.receiver.delay[i];
          const scalar_t receiver_apodization = args.receiver.apodization[i];

          // 12. Get transmitter delay, apodization
          // e.g. `transmitter_delay[scan_idx][transmitter_element_idx]`.
          const unsigned j = (args.transmitter.num_elements) * scan_idx + transmitter_element_idx;
          const scalar_t transmitter_delay = args.transmitter.delay[j];
          const scalar_t transmitter_apodization = args.transmitter.apodization[j];

          do_time_projection<scalar_t> (args,
                                        output_buffer,
                                        receiver_distance,
                                        receiver_delay,
                                        receiver_apodization,
                                        transmitter_distance,
                                        transmitter_delay,
                                        transmitter_apodization,
                                        amplitude,
                                        output_idx_0,
                                        output_idx_1);
        }
      }
    }
  }
}

template <class scalar_t>
__device__ __forceinline__ void do_time_projection (const Simulator<scalar_t> &args,
                                                    scalar_t *RESTRICT output_buffer,
                                                    scalar_t receiver_distance,
                                                    scalar_t receiver_delay,
                                                    scalar_t receiver_apodization,
                                                    scalar_t transmitter_distance,
                                                    scalar_t transmitter_delay,
                                                    scalar_t transmitter_apodization,
                                                    scalar_t amplitude,
                                                    unsigned output_idx_0,
                                                    unsigned output_idx_1)
{
  // 13. Calculate total travel distance.
  const scalar_t travel_distance = transmitter_distance + receiver_distance;

  // 14. Calculate projection index from total travel time + delays.
  const scalar_t delay = transmitter_delay + receiver_delay;
  const scalar_t fractional_idx = args.sampling_frequency * travel_distance / args.speed_of_sound;
  const unsigned projection_idx = static_cast<unsigned> (fractional_idx + delay + 0.5);

  // 15. Skip if projection lands outside scan region (i.e. scatterer is too deep).
  /*
    Note: We do not check if a scatterer lies "behind" the transducer.
    It is assumed the transmitter and receiver lie in the nonpositive-z half-region
    and that the scatterers all lie in the positive-z half-region.
  */
  if (projection_idx >= args.num_time_samples)
    return;

  // 16. Calculate phase factor.
  const scalar_t subsample_delay = (projection_idx - fractional_idx) / args.sampling_frequency;
  const scalar_t complex_phase = 2.0 * args.receiver.center_frequency * subsample_delay;
  const scalar2<scalar_t> phase_factor = exp_pi_i (complex_phase);

  // 17. Calculate depth attenuation. (Use travel distance, not time because time has delays)
  const scalar_t powers_of_ten = args.attenuation / 20;
  const scalar_t frequency_MHz = args.transmitter.center_frequency / 1e6;
  const scalar_t travel_distance_cm = travel_distance * 100;
  const scalar_t depth_attenuation = pow_s (10, -powers_of_ten * frequency_MHz * travel_distance_cm);

  // 18. Calculate weight: (transmitter apodization) * (receiver apodization) * (depth attenuation) * (phase factor).
  const scalar2<scalar_t> weight = phase_factor * transmitter_apodization * receiver_apodization * depth_attenuation;

  // 19. Calculate index-offset into `output_buffer[][][projection_idx][]` (third dimension).
  // Note: Factor of 2 is because final axis (fourth dimension) of output_buffer holds real and imaginary components.
  const unsigned output_idx_2 = (2) * projection_idx;

  // 20. Add (scatterer amplitude) * weight into output buffer.
  const scalar2<scalar_t> output = weight * amplitude;
  atomicAdd (&output_buffer[output_idx_0 + output_idx_1 + output_idx_2 + 0], output.x);
  atomicAdd (&output_buffer[output_idx_0 + output_idx_1 + output_idx_2 + 1], output.y);
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

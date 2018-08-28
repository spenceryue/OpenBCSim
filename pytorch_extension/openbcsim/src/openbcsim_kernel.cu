#include "openbcsim_kernel.cuh"
#include "vector_functions_extended.cuh"

// Forward declaration
template <class scalar_t>
__device__ __forceinline__ void do_time_projection (const Simulator<scalar_t> &args,
                                                    scalar_t *RESTRICT output_buffer,
                                                    scalar_t rx_distance,
                                                    scalar_t rx_delay,
                                                    scalar_t rx_apodization,
                                                    scalar_t tx_distance,
                                                    scalar_t tx_delay,
                                                    scalar_t tx_apodization,
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
            Loop over focal points (args.tx.num_focal_points):
              Do time projection

    Note: The 3 outermost for-loops follow the grid-stride-loop convention.
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
    // Note: `rx_idx` is the sub-element index (as opposed to element index below).
    for (unsigned rx_idx = blockIdx.y;
         rx_idx < args.rx.num_subelements;
         rx_idx += gridDim.y)
    {
      const auto rx = make_scalar3 (args.rx.x[rx_idx],
                                    args.rx.y[rx_idx],
                                    args.rx.z[rx_idx]);

      // 4. Calculate receive distance
      const scalar_t rx_distance = dist (scatterer, rx);

      // 5. Get receiver element index.
      // Note: `rx_element_idx` is the element index (as opposed to sub-element index above).
      const unsigned rx_element_idx = rx_idx / (args.rx.subdivision_factor);

      // 6. Calculate index-offset into `output_buffer[][rx_element_idx][][]` (second dimension).
      const unsigned output_idx_1 = (2 * args.num_time_samples) * rx_element_idx;

      // 7. Get transmitter position (x,y,z) via loop index.
      // Note: `tx_idx` is the sub-element index (as opposed to element index below).
      for (unsigned tx_idx = blockIdx.z;
           tx_idx < args.tx.num_subelements;
           tx_idx += gridDim.z)
      {
        const auto tx = make_scalar3 (args.tx.x[tx_idx],
                                      args.tx.y[tx_idx],
                                      args.tx.z[tx_idx]);

        // 8. Calculate transmit distance.
        const scalar_t tx_distance = dist (tx, scatterer);

        // 9. Get transmitter element index.
        // Note: `tx_element_idx` is the element index (as opposed to sub-element index above).
        const unsigned tx_element_idx = tx_idx / (args.tx.subdivision_factor);

        for (unsigned focus_idx = 0;
             focus_idx < args.tx.num_focal_points;
             focus_idx++)
        {
          // 10. Calculate index-offset into `output_buffer[focus_idx][][][]` (first dimension).
          const unsigned output_idx_0 = (2 * args.num_time_samples * args.rx.num_elements) * focus_idx;

          // 11. Get receiver delay, apodization
          // e.g. `rx_delay[focus_idx % args.rx.num_focal_points][rx_element_idx]`.
          // Note: The focus_idx dimension modulus allows the receiver to have fewer scans than the transmitter.
          const unsigned i = (args.rx.num_elements) * (focus_idx % args.rx.num_focal_points) + rx_element_idx;
          const scalar_t rx_delay = args.rx.delays[i];
          const scalar_t rx_apodization = args.rx.apodization[i];

          // 12. Get transmitter delay, apodization
          // e.g. `tx_delay[focus_idx][tx_element_idx]`.
          const unsigned j = (args.tx.num_elements) * focus_idx + tx_element_idx;
          const scalar_t tx_delay = args.tx.delays[j];
          const scalar_t tx_apodization = args.tx.apodization[j];

          do_time_projection<scalar_t> (args,
                                        output_buffer,
                                        rx_distance,
                                        rx_delay,
                                        rx_apodization,
                                        tx_distance,
                                        tx_delay,
                                        tx_apodization,
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
                                                    scalar_t rx_distance,
                                                    scalar_t rx_delay,
                                                    scalar_t rx_apodization,
                                                    scalar_t tx_distance,
                                                    scalar_t tx_delay,
                                                    scalar_t tx_apodization,
                                                    scalar_t amplitude,
                                                    unsigned output_idx_0,
                                                    unsigned output_idx_1)
{
  // 13. Calculate total travel distance.
  const scalar_t travel_distance = tx_distance + rx_distance;

  // 14. Calculate projection index from total travel time + delays.
  const scalar_t delay = tx_delay + rx_delay;
  const scalar_t fractional_idx = args.sampling_frequency * travel_distance / args.speed_of_sound;
  const unsigned projection_idx = static_cast<unsigned> (fractional_idx + delay + 0.5);

  // 15. Skip if projection lands outside scan region (i.e. scatterer is too deep).
  /*
    Note: We do not check if a scatterer lies "behind" the transducer.
    It is assumed the tx and rx lie in the nonpositive-z half-region
    and that the scatterers all lie in the positive-z half-region.
  */
  if (projection_idx >= args.num_time_samples)
    return;

  // 16. Calculate phase factor.
  const scalar_t subsample_delay = (projection_idx - fractional_idx) / args.sampling_frequency;
  const scalar_t complex_phase = 2.0 * args.rx.center_frequency * subsample_delay;
  const scalar2<scalar_t> phase_factor = exp_pi_i (complex_phase);

  // 17. Calculate depth attenuation. (Use travel distance, not time because time has delays)
  const scalar_t powers_of_ten = args.attenuation / 20;
  const scalar_t frequency_MHz = args.tx.center_frequency / 1e6;
  const scalar_t travel_distance_cm = travel_distance * 100;
  const scalar_t depth_attenuation = pow_s (10, -powers_of_ten * frequency_MHz * travel_distance_cm);

  // 18. Calculate weight: (tx apodization) * (rx apodization) * (depth attenuation) * (phase factor).
  const scalar2<scalar_t> weight = phase_factor * tx_apodization * rx_apodization * depth_attenuation;

  // 19. Calculate index-offset into `output_buffer[][][projection_idx][]` (third dimension).
  // Note: Factor of 2 is because final axis (fourth dimension) of output_buffer holds real and imaginary components.
  const unsigned output_idx_2 = (2) * projection_idx;

  // 20. Add (scatterer amplitude) * weight into output buffer.
  const scalar2<scalar_t> output = weight * amplitude;
  atomicAdd (&output_buffer[output_idx_0 + output_idx_1 + output_idx_2 + 0], output.x);
  atomicAdd (&output_buffer[output_idx_0 + output_idx_1 + output_idx_2 + 1], output.y);
}

template <class scalar_t>
DLL_PUBLIC void launch_projection_kernel (const Simulator<scalar_t> &args, scalar_t *output_buffer,
                                          dim3 grid, dim3 block, cudaStream_t stream)
{
  cudaFuncSetCacheConfig (projection_kernel<scalar_t>, cudaFuncCachePreferL1);
  projection_kernel<scalar_t><<<grid, block, /*shared bytes=*/0, stream>>> (args, output_buffer);
}

template <class T>
cudaError cuda_malloc_managed (T **pointer_to_pointer, size_t bytes)
{
  // Wrapper for cudaMallocManaged().
  // "To use these functions, your application needs to be compiled with the nvcc compiler."
  // https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__HIGHLEVEL.html
  return cudaMallocManaged (pointer_to_pointer, bytes);
}

template DLL_PUBLIC void launch_projection_kernel (const Simulator<float> &args, float *output_buffer,
                                                   dim3 grid, dim3 block, cudaStream_t stream);
template DLL_PUBLIC void launch_projection_kernel (const Simulator<double> &args, double *output_buffer,
                                                   dim3 grid, dim3 block, cudaStream_t stream);
template cudaError cuda_malloc_managed (float **pointer_to_pointer, size_t bytes);
template cudaError cuda_malloc_managed (double **pointer_to_pointer, size_t bytes);

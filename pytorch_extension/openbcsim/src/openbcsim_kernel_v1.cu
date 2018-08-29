#include "openbcsim_kernel.cuh"
#include "vector_functions_extended.cuh"

template <class scalar_t>
__global__ void projection_kernel (const Simulator<scalar_t> args, scalar_t *RESTRICT output_buffer)
{
  /*
  For scatterer in scatterers:
    d0 <- Get distance from scatterer to origin.
    For focus in focal points:
      Time-project tx subelements into scatterer shared memory buffer.
      For subelement in rx:
        d1 <- Get distance from subelement to scatterer.
        Multiply-Add scatterer shared memory buffer to rx element buffer centered around time.
        `(d0 + d1)/speed_of_sound + delays`.
      Clear shared memory buffer.
  */

  extern __shared__ scalar_t shared_buffer[];

  // 1. Calculate scatterer index.
  for (unsigned SS = blockIdx.x;
       SS < args.num_scatterers;
       SS += gridDim.x)
  {
    // 2. Get scatterer position and amplitude.
    const auto scatterer = make_scalar3 (args.scatterer_x[SS],
                                         args.scatterer_y[SS],
                                         args.scatterer_z[SS]);
    const auto amplitude = args.scatterer_amplitude[SS];

    // 3. Get distance from scatterer to origin.
    const auto d0 = norm (scatterer);

    // 1. Calculate focal point index.
    for (unsigned FF = 0;
         FF < args.tx.num_focal_points;
         FF++)
    {
      // 10. Calculate offset to `output_buffer[FF, 0, 0, 0]`.
      const unsigned offset_0 = (2 * args.num_time_samples * args.rx.num_elements) * FF;

      do_time_projection (args, shared_buffer);
    }
  }
}

template <class scalar_t>
__device__ __forceinline__ void do_time_projection (const Simulator<scalar_t> &args,
                                                    scalar_t *RESTRICT shared_buffer)
{
  // 1. Calculate tx subelement index.
  for (unsigned TT = threadIdx.x;
       TT < args.tx.num_subelements;
       TT += blockDim.x)
  {
    // 2. Get tx subelement position.
    const auto tx = make_scalar3 (args.tx.x[TT],
                                  args.tx.y[TT],
                                  args.tx.z[TT]);

    // 8. Calculate tx subelement distance.
    const auto tx_distance = dist (tx, scatterer);

    // 1. Calculate tx element index.
    const unsigned UU = TT / (args.tx.subdivision_factor);

    // 12. Get transmitter delay, apodization
    // e.g. `tx_delay[focus_idx][tx_element_idx]`.
    const unsigned j = (args.tx.num_elements) * focus_idx + tx_element_idx;
    const auto tx_delay = args.tx.delays[j];
    const auto tx_apodization = args.tx.apodization[j];
  }

  // 13. Calculate total travel distance.
  const scalar_t travel_distance = tx_distance + rx_distance;

  // 14. Calculate projection index from total travel time + delays.
  const scalar_t delay = tx_delay + rx_delay;
  const scalar_t fractional_idx = args.sampling_frequency * travel_distance / args.speed_of_sound;
  const unsigned projection_idx = static_cast<unsigned> (fractional_idx + delay + 0.5);
}
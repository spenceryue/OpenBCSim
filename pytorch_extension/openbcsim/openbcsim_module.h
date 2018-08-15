#pragma once
#include "base.h" // Must go first

#include "data_types.h"
#include "device_properties.h"
#include "openbcsim_kernel.cuh"

template <class scalar_t>
Transducer<scalar_t> create (unsigned num_elements,
                             unsigned num_subelements,
                             unsigned division_factor,
                             at::Tensor x,
                             at::Tensor y,
                             at::Tensor z,
                             at::Tensor delay,
                             at::Tensor apodization,
                             scalar_t center_frequency);

template <class scalar_t>
Simulator<scalar_t> create (scalar_t sampling_frequency,
                            unsigned decimation,
                            scalar_t scan_depth,
                            scalar_t speed_of_sound,
                            scalar_t attenuation,
                            Transducer<scalar_t> &transmitter,
                            Transducer<scalar_t> &receiver,
                            unsigned num_time_samples,
                            at::Tensor scatterer_x,
                            at::Tensor scatterer_y,
                            at::Tensor scatterer_z,
                            at::Tensor scatterer_amplitude,
                            unsigned num_scatterers);

template <class scalar_t>
at::Tensor launch (const Simulator<scalar_t> &args, int scatterer_blocks_factor = 32, unsigned receiver_threads = 1,
                   unsigned transmitter_threads = 1);

template <class scalar_t>
void launch (const Simulator<scalar_t> &args, scalar_t *output_buffer, int scatterer_blocks_factor,
             unsigned receiver_threads, unsigned transmitter_threads);

void reset_device ();
void synchronize ();

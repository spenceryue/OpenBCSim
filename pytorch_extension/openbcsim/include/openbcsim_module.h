#pragma once
#include "base.h" // Must go first

#include "data_types.h"
#include "device_properties.h"
#include "openbcsim_kernel.cuh"
#include <array>

template <class scalar_t>
DLL_PUBLIC Transducer<scalar_t> create (unsigned num_elements,
                                        unsigned num_subelements,
                                        unsigned subdivision_factor,
                                        unsigned num_scans,
                                        at::Tensor x,
                                        at::Tensor y,
                                        at::Tensor z,
                                        at::Tensor delay,
                                        at::Tensor apodization,
                                        scalar_t center_frequency);

template <class scalar_t>
DLL_PUBLIC Simulator<scalar_t> create (scalar_t sampling_frequency,
                                       unsigned decimation,
                                       scalar_t scan_depth,
                                       scalar_t speed_of_sound,
                                       scalar_t attenuation,
                                       Transducer<scalar_t> &tx,
                                       Transducer<scalar_t> &rx,
                                       unsigned num_time_samples,
                                       at::Tensor scatterer_x,
                                       at::Tensor scatterer_y,
                                       at::Tensor scatterer_z,
                                       at::Tensor scatterer_amplitude,
                                       unsigned num_scatterers);

DLL_PUBLIC dim3 make_grid (int scatterer_blocks_factor = 32,
                           unsigned rx_blocks = 1,
                           unsigned tx_blocks = 1,
                           int device = 0);

template <class scalar_t>
DLL_PUBLIC std::array<int64_t, 4> make_shape (const Simulator<scalar_t> &args);

template <class scalar_t>
DLL_PUBLIC at::Tensor launch (const Simulator<scalar_t> &args,
                              dim3 grid = make_grid (),
                              dim3 block = get_properties ().maxThreadsPerBlock);

DLL_PUBLIC void reset_device ();
DLL_PUBLIC void synchronize ();

#pragma once

template <class scalar_t>
struct Transducer
{
  // Position
  const scalar_t *__restrict__ x;
  const scalar_t *__restrict__ y;
  const scalar_t *__restrict__ z;
  // Configuration
  const scalar_t *__restrict__ delay;
  const scalar_t *__restrict__ apodization;
  // Length
  const size_t num_elements;       // Length of delay, apodization
  const size_t num_subelements;    // Length of x, y, z, radial, lateral, elevational
  const size_t num_subdivisions;   // Product of `num_sub_x * num_sub_y`
  const scalar_t center_frequency; // Center frequency of the transducer
};

template <class scalar_t>
struct LinearTransducer : Transducer<scalar_t>
{
  const size_t num_sub_x, num_sub_y; // Number of subelement divisions in x,y directions
};

template <class scalar_t>
struct Simulation
{
  const scalar_t sampling_frequency;
  const scalar_t decimation;
  const scalar_t center_frequency;
  const scalar_t attenuation;
  const scalar_t scan_depth;
  const scalar_t speed_of_sound;
  const Transducer<scalar_t> transmitter;
  const Transducer<scalar_t> receiver;
  const scalar_t *__restrict__ scatterer_x;
  const scalar_t *__restrict__ scatterer_y;
  const scalar_t *__restrict__ scatterer_z;
  const scalar_t *__restrict__ scatterer_amplitude;
  const size_t num_scatterers;
};

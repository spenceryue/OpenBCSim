#pragma once

#ifdef _MSC_VER
#define RESTRICT __restrict
#else
#define RESTRICT __restrict__
#endif

template <class scalar_t>
struct Transducer
{
  // Length
  unsigned num_elements;     // Length of delay, apodization
  unsigned num_subelements;  // Length of x, y, z
  unsigned num_subdivisions; // Equals `num_subelements / num_elements`
  // Position
  scalar_t *RESTRICT x;
  scalar_t *RESTRICT y;
  scalar_t *RESTRICT z;
  // Configuration
  scalar_t *RESTRICT delay;
  scalar_t *RESTRICT apodization;
  scalar_t center_frequency; // Center frequency of the transducer
};

template <class scalar_t>
struct LinearTransducer : Transducer<scalar_t>
{
  unsigned num_sub_x, num_sub_y; // Number of subelement divisions in x,y directions
};

template <class scalar_t>
struct Simulation
{
  scalar_t sampling_frequency;
  scalar_t decimation;
  scalar_t scan_depth;
  scalar_t speed_of_sound;
  scalar_t attenuation;
  Transducer<scalar_t> transmitter;
  Transducer<scalar_t> receiver;
  unsigned num_time_samples;
  scalar_t *RESTRICT scatterer_x;
  scalar_t *RESTRICT scatterer_y;
  scalar_t *RESTRICT scatterer_z;
  scalar_t *RESTRICT scatterer_amplitude;
  unsigned num_scatterers;
};

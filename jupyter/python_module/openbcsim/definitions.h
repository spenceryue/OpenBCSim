#pragma once

#ifdef _MSC_VER
#define RESTRICT __restrict
#else
#define RESTRICT __restrict__
#endif

template <class scalar_t>
struct Transducer
{
  /* 1 */ unsigned num_elements;     // Length of delay, apodization
  /* 2 */ unsigned num_subelements;  // Length of x, y, z
  /* 3 */ unsigned num_subdivisions; // Equals `num_subelements / num_elements`
  /* 4 */ const scalar_t *RESTRICT x;
  /* 5 */ const scalar_t *RESTRICT y;
  /* 6 */ const scalar_t *RESTRICT z;
  /* 7 */ const scalar_t *RESTRICT delay;
  /* 8 */ const scalar_t *RESTRICT apodization;
  /* 9 */ scalar_t center_frequency;
};

template <class scalar_t>
struct LinearTransducer : Transducer<scalar_t>
{
  unsigned num_sub_x, num_sub_y; // Number of subelement divisions in x,y directions
};

template <class scalar_t>
struct Simulator
{
  /*  1 */ scalar_t sampling_frequency;
  /*  2 */ unsigned decimation;
  /*  3 */ scalar_t scan_depth;
  /*  4 */ scalar_t speed_of_sound;
  /*  5 */ scalar_t attenuation;
  /*  6 */ Transducer<scalar_t> transmitter;
  /*  7 */ Transducer<scalar_t> receiver;
  /*  8 */ unsigned num_time_samples;
  /*  9 */ const scalar_t *RESTRICT scatterer_x;
  /* 10 */ const scalar_t *RESTRICT scatterer_y;
  /* 11 */ const scalar_t *RESTRICT scatterer_z;
  /* 12 */ const scalar_t *RESTRICT scatterer_amplitude;
  /* 13 */ unsigned num_scatterers;
};

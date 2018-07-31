#pragma once
#include "cuda_helpers.h"
#include <memory>

struct ApertureParams
{
  // Position
  float *__restrict__ x;
  float *__restrict__ y;
  float *__restrict__ z;
  // Normal vector coordinates
  float *__restrict__ radial;
  float *__restrict__ lateral;
  float *__restrict__ elevational;
  // Configuration
  float *__restrict__ delay;
  float *__restrict__ apodization;
  // Length
  int N_elements;    // Length of delay, apodization
  int N_subelements; // Length of x, y, z
};

namespace bcsim
{
class Aperture
{
public:
  using DeviceBuffer = DeviceBufferRAII<float>;

  // Position
  DeviceBuffer x;
  DeviceBuffer y;
  DeviceBuffer z;

  // Normal vector coordinates
  DeviceBuffer radial;
  DeviceBuffer lateral;
  DeviceBuffer elevational;

  // Configuration
  DeviceBuffer delay;
  DeviceBuffer apodization;

  // Length
  int N_elements;    // Length of delay, apodization
  int N_subelements; // Length of x, y, z

  Aperture (const struct ApertureParams *params)
      : x (params->N_subelements),
        y (params->N_subelements),
        z (params->N_subelements),
        radial (params->N_subelements),
        lateral (params->N_subelements),
        elevational (params->N_subelements),
        delay (params->N_elements),
        apodization (params->N_elements),
        N_elements (params->N_elements),
        N_subelements (params->N_subelements)
  {
    std::vector<CudaStreamRAII> stream (8);

    x.copyFromAsync (params->x, N_subelements, stream[0].get ());
    y.copyFromAsync (params->y, N_subelements, stream[1].get ());
    z.copyFromAsync (params->z, N_subelements, stream[2].get ());
    radial.copyFromAsync (params->radial, N_subelements, stream[3].get ());
    lateral.copyFromAsync (params->lateral, N_subelements, stream[4].get ());
    elevational.copyFromAsync (params->elevational, N_subelements, stream[5].get ());
    delay.copyFromAsync (params->delay, N_subelements, stream[6].get ());
    apodization.copyFrom (params->apodization, N_subelements, stream[7].get ());
  }

  void make_ApertureParams (struct ApertureParams *params) const
  {
    params->x = x.data ();
    params->y = y.data ();
    params->z = z.data ();
    params->radial = radial.data ();
    params->lateral = lateral.data ();
    params->elevational = elevational.data ();
    params->delay = delay.data ();
    params->apodization = apodization.data ();
    params->N_elements = N_elements;
    params->N_subelements = N_subelements;
  }
};
} // namespace bcsim

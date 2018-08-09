/* https://github.com/pybind/pybind11/issues/1212#issuecomment-365555709 */
#define strdup _strdup
#include <torch/torch.h>

#include "definitions.h"
#include "openbcsim_kernel.cuh"
#include <type_traits>

namespace py = pybind11;
using namespace pybind11::literals;

#define CHECK_CUDA(x) AT_ASSERT (x.type ().is_cuda (), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERT (x.is_contiguous (), #x " must be contiguous")
#define CHECK_INPUT(x)   \
  do                     \
  {                      \
    CHECK_CUDA (x);      \
    CHECK_CONTIGUOUS (x) \
  } while (0)

template <class scalar_t, class what, std::enable_if_t<std::is_same_v<what, Transducer<scalar_t>>, int> = 0>
Transducer<scalar_t> create (unsigned num_elements,
                             unsigned num_subelements,
                             unsigned num_subdivisions,
                             at::Tensor x,
                             at::Tensor y,
                             at::Tensor z,
                             at::Tensor delay,
                             at::Tensor apodization,
                             scalar_t center_frequency)
{
  CHECK_INPUT (x);
  CHECK_INPUT (y);
  CHECK_INPUT (z);
  CHECK_INPUT (delay);
  CHECK_INPUT (apodization);

  Transducer<scalar_t> result;
  result.num_elements = num_elements;
  result.num_subelements = num_subelements;
  result.num_subdivisions = num_subdivisions;
  result.x = x.data<scalar_t> ();
  result.y = y.data<scalar_t> ();
  result.z = z.data<scalar_t> ();
  result.delay = delay.data<scalar_t> ();
  result.apodization = apodization.data<scalar_t> ();
  result.center_frequency = center_frequency;

  return result;
}

template <class scalar_t, class what, std::enable_if_t<std::is_same_v<what, Simulation<scalar_t>>, int> = 0>
Simulation<scalar_t> create (scalar_t sampling_frequency,
                             scalar_t decimation,
                             scalar_t scan_depth,
                             scalar_t speed_of_sound,
                             scalar_t attenuation,
                             const Transducer<scalar_t> &transmitter,
                             const Transducer<scalar_t> &receiver,
                             unsigned num_time_samples,
                             at::Tensor scatterer_x,
                             at::Tensor scatterer_y,
                             at::Tensor scatterer_z,
                             at::Tensor scatterer_amplitude,
                             unsigned num_scatterers)
{
  CHECK_INPUT (scatterer_x);
  CHECK_INPUT (scatterer_y);
  CHECK_INPUT (scatterer_z);
  CHECK_INPUT (scatterer_amplitude);

  Simulation<scalar_t> result;
  result.sampling_frequency = sampling_frequency;
  result.decimation = decimation;
  result.scan_depth = scan_depth;
  result.speed_of_sound = speed_of_sound;
  result.attenuation = attenuation;
  result.num_time_samples = num_time_samples;
  result.transmitter = transmitter;
  result.receiver = receiver;
  result.scatterer_x = scatterer_x.data<scalar_t> ();
  result.scatterer_y = scatterer_y.data<scalar_t> ();
  result.scatterer_z = scatterer_z.data<scalar_t> ();
  result.scatterer_amplitude = scatterer_amplitude.data<scalar_t> ();
  result.num_scatterers = num_scatterers;

  return result;
}

template <class T>
T ceil_div (T num, T den)
{
  return static_cast<T> ((num + den - 1) / den);
}

template <class scalar_t>
at::Tensor launch (const Simulation<scalar_t> &args)
{
  const at::ScalarType s_type = (std::is_same_v<scalar_t, float>) ? (at::kFloat) : (at::kDouble);

  // Must use int64_t[] because at::IntList constructor is finicky.
  // If you get an "out of memory" error it was because the at::IntList constructor didn't use/convert the arguments properly.
  // (The zeros() function takes a sizes argument of type at::IntList.)
  const int64_t sizes[] = {args.receiver.num_subelements * args.num_time_samples * 2};

  // For some reason need to use `torch::` namespace instead of `at::` here...
  // https://github.com/pytorch/pytorch/issues/6103#issuecomment-377312709
  at::Tensor output = torch::CUDA (s_type).zeros (sizes);

  const dim3 grid = {ceil_div<unsigned> (args.num_scatterers, THREADS_PER_BLOCK),
                     args.transmitter.num_subelements,
                     args.receiver.num_subelements};
  const dim3 block = {THREADS_PER_BLOCK};
  launch_projection_kernel<scalar_t> (args, output.data<scalar_t> (), grid, block);

  return output;
}

#define DEFINE_FOR_TYPE(scalar_t)                                                                         \
  do                                                                                                      \
  {                                                                                                       \
    /* Module docstring */                                                                                \
    m.doc () = "Interface to pass simulation parameters to CUDA kernel.";                                 \
                                                                                                          \
    py::class_<Transducer<scalar_t>> (m, "Transducer_" #scalar_t, "Struct with transducer parameters.\n", \
                                      py::dynamic_attr ())                                                \
        .def (py::init (&create<scalar_t, Transducer<scalar_t>>),                                         \
              "num_elements"_a,                                                                           \
              "num_subelements"_a,                                                                        \
              "num_subdivisions"_a,                                                                       \
              "x"_a,                                                                                      \
              "y"_a,                                                                                      \
              "z"_a,                                                                                      \
              "delay"_a,                                                                                  \
              "apodization"_a,                                                                            \
              "center_frequency"_a)                                                                       \
        .def_readonly ("num_elements", &Transducer<scalar_t>::num_elements)                               \
        .def_readonly ("num_subelements", &Transducer<scalar_t>::num_subelements)                         \
        .def_readonly ("num_subdivisions", &Transducer<scalar_t>::num_subdivisions)                       \
        .def_readonly ("center_frequency", &Transducer<scalar_t>::center_frequency);                      \
                                                                                                          \
    py::class_<Simulation<scalar_t>> (m, "Simulation_" #scalar_t, "Struct with simulation parameters.\n", \
                                      py::dynamic_attr ())                                                \
        .def (py::init (&create<scalar_t, Simulation<scalar_t>>),                                         \
              "sampling_frequency"_a,                                                                     \
              "decimation"_a,                                                                             \
              "scan_depth"_a,                                                                             \
              "speed_of_sound"_a,                                                                         \
              "attenuation"_a,                                                                            \
              "transmitter"_a,                                                                            \
              "receiver"_a,                                                                               \
              "num_time_samples"_a,                                                                       \
              "scatterer_x"_a,                                                                            \
              "scatterer_y"_a,                                                                            \
              "scatterer_z"_a,                                                                            \
              "scatterer_amplitude"_a,                                                                    \
              "num_scatterers"_a)                                                                         \
        .def_readonly ("sampling_frequency", &Simulation<scalar_t>::sampling_frequency)                   \
        .def_readonly ("decimation", &Simulation<scalar_t>::decimation)                                   \
        .def_readonly ("scan_depth", &Simulation<scalar_t>::scan_depth)                                   \
        .def_readonly ("speed_of_sound", &Simulation<scalar_t>::speed_of_sound)                           \
        .def_readonly ("attenuation", &Simulation<scalar_t>::attenuation)                                 \
        .def_readonly ("transmitter", &Simulation<scalar_t>::transmitter)                                 \
        .def_readonly ("receiver", &Simulation<scalar_t>::receiver)                                       \
        .def_readonly ("num_time_samples", &Simulation<scalar_t>::num_time_samples)                       \
        .def_readonly ("num_scatterers", &Simulation<scalar_t>::num_scatterers);                          \
                                                                                                          \
    m.def ("launch_" #scalar_t, &launch<scalar_t>,                                                        \
           "Simulate the response from a single transmission pulse.\n"                                    \
           "Outputs a Tensor of shape:\n"                                                                 \
           "  (args.receiver.num_subelements, args.num_time_samples).",                                   \
           "args"_a);                                                                                     \
  } while (0)

PYBIND11_MODULE (TORCH_EXTENSION_NAME, m)
{
  DEFINE_FOR_TYPE (float);
  DEFINE_FOR_TYPE (double);
}

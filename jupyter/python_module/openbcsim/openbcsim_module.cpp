/* https://github.com/pybind/pybind11/issues/1212#issuecomment-365555709 */
#define strdup _strdup
#include <torch/torch.h>

#include "definitions.h"
#include "openbcsim_kernel.cuh"
#include "pretty_print.h" // for testing and debugging
#include <string>
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

  Transducer<scalar_t> result = {
      /* 1 */ num_elements,
      /* 2 */ num_subelements,
      /* 3 */ num_subdivisions,
      /* 4 */ x.data<scalar_t> (),
      /* 5 */ y.data<scalar_t> (),
      /* 6 */ z.data<scalar_t> (),
      /* 7 */ delay.data<scalar_t> (),
      /* 8 */ apodization.data<scalar_t> (),
      /* 9 */ center_frequency,
  };

  return result;
}

template <class scalar_t, class what, std::enable_if_t<std::is_same_v<what, Simulator<scalar_t>>, int> = 0>
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
                            unsigned num_scatterers)
{
  CHECK_INPUT (scatterer_x);
  CHECK_INPUT (scatterer_y);
  CHECK_INPUT (scatterer_z);
  CHECK_INPUT (scatterer_amplitude);

  Simulator<scalar_t> result = {
      /*  1 */ sampling_frequency,
      /*  2 */ decimation,
      /*  3 */ scan_depth,
      /*  4 */ speed_of_sound,
      /*  5 */ attenuation,
      /*  6 */ transmitter,
      /*  7 */ receiver,
      /*  8 */ num_time_samples,
      /*  9 */ scatterer_x.data<scalar_t> (),
      /* 10 */ scatterer_y.data<scalar_t> (),
      /* 11 */ scatterer_z.data<scalar_t> (),
      /* 12 */ scatterer_amplitude.data<scalar_t> (),
      /* 13 */ num_scatterers,
  };

  return result;
}

template <class T>
T ceil_div (T num, T den)
{
  return static_cast<T> ((num + den - 1) / den);
}

template <class scalar_t, bool verbose = false>
at::Tensor launch (const Simulator<scalar_t> &args)
{
  const at::ScalarType s_type = (std::is_same_v<scalar_t, float>) ? (at::kFloat) : (at::kDouble);

  // Must use int64_t[] because at::IntList constructor is finicky.
  // If you get an "out of memory" error it was because the at::IntList constructor didn't use/convert the arguments properly.
  // (The zeros() function takes a sizes argument of type at::IntList.)
  const int64_t sizes[] = {args.receiver.num_subelements * args.num_time_samples * 2};
  // TODO try int[]

  // For some reason need to use `torch::` namespace instead of `at::` here...
  // https://github.com/pytorch/pytorch/issues/6103#issuecomment-377312709
  at::Tensor output = torch::CUDA (s_type).zeros (sizes);

  const dim3 grid = {ceil_div<unsigned> (args.num_scatterers, THREADS_PER_BLOCK),
                     args.transmitter.num_subelements,
                     args.receiver.num_subelements};
  const dim3 block = {THREADS_PER_BLOCK};

  if constexpr (verbose)
  {
    using namespace std;
    pretty::block<true> ("Out elems.", 11) << "  {:,}"_s.format (output.sizes ()[0]) << endl;
    pretty::block<true> ("Time samps.", 11) << "  {:,}"_s.format (args.num_time_samples) << endl;
    pretty::block<true> ("Scatterers", 11) << "  {:,}"_s.format (args.num_scatterers) << endl;
    pretty::block<true> ("Threads", 11) << "  {:,}"_s.format (static_cast<long long> (grid.x) * grid.y * grid.z * 1024) << endl;
    pretty::block<true> ("Blocks", 11) << "  {:,}"_s.format (static_cast<long long> (grid.x) * grid.y * grid.z) << endl;
  }

  launch_projection_kernel<scalar_t> (args, output.data<scalar_t> (), grid, block);

  return output;
}

template <class scalar_t>
void declare_bindings (py::module m, const std::string &type_string)
{
  /* Module docstring */
  m.doc () = "Interface to pass simulation parameters to CUDA kernel.";

  py::class_<Transducer<scalar_t>> (m, ("Transducer_" + type_string).c_str (), "Struct with transducer parameters.\n",
                                    py::dynamic_attr ())
      .def (py::init (&create<scalar_t, Transducer<scalar_t>>),
            "num_elements"_a,
            "num_subelements"_a,
            "num_subdivisions"_a,
            "x"_a,
            "y"_a,
            "z"_a,
            "delay"_a,
            "apodization"_a,
            "center_frequency"_a)
      .def_readonly ("num_elements", &Transducer<scalar_t>::num_elements)
      .def_readonly ("num_subelements", &Transducer<scalar_t>::num_subelements)
      .def_readonly ("num_subdivisions", &Transducer<scalar_t>::num_subdivisions)
      .def_readonly ("center_frequency", &Transducer<scalar_t>::center_frequency);

  py::class_<Simulator<scalar_t>> (m, ("Simulator_" + type_string).c_str (), "Struct with simulation parameters.\n",
                                   py::dynamic_attr ())
      .def (py::init (&create<scalar_t, Simulator<scalar_t>>),
            "sampling_frequency"_a,
            "decimation"_a,
            "scan_depth"_a,
            "speed_of_sound"_a,
            "attenuation"_a,
            "transmitter"_a,
            "receiver"_a,
            "num_time_samples"_a,
            "scatterer_x"_a,
            "scatterer_y"_a,
            "scatterer_z"_a,
            "scatterer_amplitude"_a,
            "num_scatterers"_a)
      .def_readonly ("sampling_frequency", &Simulator<scalar_t>::sampling_frequency)
      .def_readonly ("decimation", &Simulator<scalar_t>::decimation)
      .def_readonly ("scan_depth", &Simulator<scalar_t>::scan_depth)
      .def_readonly ("speed_of_sound", &Simulator<scalar_t>::speed_of_sound)
      .def_readonly ("attenuation", &Simulator<scalar_t>::attenuation)
      .def_readonly ("transmitter", &Simulator<scalar_t>::transmitter)
      .def_readonly ("receiver", &Simulator<scalar_t>::receiver)
      .def_readonly ("num_time_samples", &Simulator<scalar_t>::num_time_samples)
      .def_readonly ("num_scatterers", &Simulator<scalar_t>::num_scatterers);

  m.def (("launch_" + type_string).c_str (), &launch<scalar_t>,
         "Simulate the response from a single transmission pulse.\n"
         "Outputs a Tensor of shape:\n"
         "  (args.receiver.num_subelements, args.num_time_samples).",
         "args"_a);
}

PYBIND11_MODULE (TORCH_EXTENSION_NAME, m)
{
  declare_bindings<float> (m, "float");
  declare_bindings<double> (m, "double");
  m.def ("reset_device", &reset_device);
}

/* C++ test */
#include <pybind11/embed.h> // everything needed for embedding

int main (int argc, const char *argv[])
{
  using namespace std;
  using namespace pretty;

  // Takes one argument: <Number of transducer elements>
  if (argc < 2)
  {
    cerr << endl;
    cerr << "  Usage: " << argv[0] << " <Number of transducer elements>" << endl;
    return 1;
  }

  // Initializing Python Interpreter, torch, and torch.cuda
  py::scoped_interpreter guard{}; // start the interpreter and keep it alive
  timestamp () << "  launched interpreter" << endl;
  py::object torch = py::module::import ("torch");
  timestamp () << "  imported torch (version: " << py::cast<string> (torch.attr ("__version__")) << ")" << endl;
  torch.attr ("cuda").attr ("init") ();
  timestamp () << "  torch.cuda.init ()" << endl;

  // Initializing Transducer
  unsigned num_elements = stoi (argv[1]);
  block<true> ("Num elems.") << "  " << num_elements << endl;
  unsigned num_subelements = num_elements;
  unsigned num_subdivisions = 1;
  at::Tensor x = torch::CUDA (at::kFloat).arange (static_cast<int> (num_elements));
  x -= (num_elements - 1) / 2.0;
  x *= .5e-3;
  at::Tensor y = torch::CUDA (at::kFloat).zeros ({num_elements});
  at::Tensor z = torch::CUDA (at::kFloat).zeros ({num_elements});
  at::Tensor delay = torch::CUDA (at::kFloat).zeros ({num_elements});
  at::Tensor apodization = torch::CUDA (at::kFloat).ones ({num_elements});
  float center_frequency = 3.5e6;
  Transducer<float> tx = create<float, Transducer<float>> (num_elements, num_subelements, num_subdivisions, x, y, z, delay, apodization,
                                                           center_frequency);
  timestamp () << "  created Transducer" << endl;

  // Initializing Simulator
  float sampling_frequency = 100e6;
  unsigned decimation = 10;
  float scan_depth = 9e-2;
  float speed_of_sound = 1540;
  float attenuation = .7;
  Transducer<float> &transmitter = tx;
  Transducer<float> &receiver = tx;
  unsigned num_time_samples = static_cast<int> (2 * scan_depth / speed_of_sound * sampling_frequency + .5);
  at::Tensor scatterer_x = torch::CUDA (at::kFloat).arange (100'000);
  scatterer_x -= (100'000 - 1) / 2.0;      // Center at 0
  scatterer_x *= 4e-2 / (100'000 - 1) * 2; // Adjust spacing
  at::Tensor scatterer_y = torch::CUDA (at::kFloat).zeros ({100'000});
  at::Tensor scatterer_z = torch::CUDA (at::kFloat).arange (100'000);
  scatterer_z *= 9e-2 / (100'000 - 1); // Adjust spacing
  at::Tensor scatterer_amplitude = torch::CUDA (at::kFloat).ones ({100'000});
  unsigned num_scatterers = 100'000;
  Simulator<float> sim = create<float, Simulator<float>> (sampling_frequency, decimation, scan_depth, speed_of_sound,
                                                          attenuation, transmitter, receiver, num_time_samples,
                                                          scatterer_x, scatterer_y, scatterer_z, scatterer_amplitude,
                                                          num_scatterers);
  timestamp () << "  created Simulator" << endl;

  // Launching kernel
  auto result = launch<float, true> (sim);
  timestamp () << "  launched" << endl;
  // Waiting for kernel
  synchronize ();
  timestamp () << "  synchronized" << endl;

  // Printing output
  block<true> ("Result") << "  " << py::str (py::cast (result)) << endl;
  timestamp () << "  printed" << endl;
  // Report total time
  tic = start;
  timestamp () << "  TOTAL" << endl;
}

#include "openbcsim_module.h"
#include <string>

template <class scalar_t>
Transducer<scalar_t> create (unsigned num_elements,
                             unsigned num_subelements,
                             unsigned division_factor,
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
  /* 1 */ result.num_elements = num_elements;
  /* 2 */ result.num_subelements = num_subelements;
  /* 3 */ result.division_factor = division_factor;
  /* 4 */ result.x = x.data<scalar_t> ();
  /* 5 */ result.y = y.data<scalar_t> ();
  /* 6 */ result.z = z.data<scalar_t> ();
  /* 7 */ result.delay = delay.data<scalar_t> ();
  /* 8 */ result.apodization = apodization.data<scalar_t> ();
  /* 9 */ result.center_frequency = center_frequency;

  return result;
}

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
                            unsigned num_scatterers)
{
  CHECK_INPUT (scatterer_x);
  CHECK_INPUT (scatterer_y);
  CHECK_INPUT (scatterer_z);
  CHECK_INPUT (scatterer_amplitude);

  Simulator<scalar_t> result;
  /*  1 */ result.sampling_frequency = sampling_frequency;
  /*  2 */ result.decimation = decimation;
  /*  3 */ result.scan_depth = scan_depth;
  /*  4 */ result.speed_of_sound = speed_of_sound;
  /*  5 */ result.attenuation = attenuation;
  /*  6 */ result.transmitter = transmitter;
  /*  7 */ result.receiver = receiver;
  /*  8 */ result.num_time_samples = num_time_samples;
  /*  9 */ result.scatterer_x = scatterer_x.data<scalar_t> ();
  /* 10 */ result.scatterer_y = scatterer_y.data<scalar_t> ();
  /* 11 */ result.scatterer_z = scatterer_z.data<scalar_t> ();
  /* 12 */ result.scatterer_amplitude = scatterer_amplitude.data<scalar_t> ();
  /* 13 */ result.num_scatterers = num_scatterers;

  return result;
}

template <class scalar_t>
at::Tensor launch (const Simulator<scalar_t> &args, int scatterer_blocks_factor, unsigned receiver_threads,
                   unsigned transmitter_threads)
{
  const at::ScalarType s_type = (std::is_same_v<scalar_t, float>) ? (at::kFloat) : (at::kDouble);

  // Must use int64_t[] to match at::IntList constructor.
  // (The zeros() function takes a sizes argument of type at::IntList.)
  // If get "out of memory" error, log arguments to make sure they converted properly...
  const int64_t sizes[] = {args.receiver.num_subelements * args.num_time_samples * 2};

  // Need to use `torch::` namespace instead of `at::` here. (Why...?)
  // https://github.com/pytorch/pytorch/issues/6103#issuecomment-377312709
  at::Tensor output = torch::CUDA (s_type).zeros (sizes);

  const dim3 block = {get_properties ().maxThreadsPerBlock};
  if (scatterer_blocks_factor > 0)
  {
    const dim3 grid = {scatterer_blocks_factor * get_properties ().multiProcessorCount,
                       receiver_threads,
                       transmitter_threads};

    launch_projection_kernel<scalar_t> (args, output.data<scalar_t> (), grid, block);
  }
  else
  {
    const dim3 grid = {ceil_div<unsigned> (args.num_scatterers, block.x),
                       receiver_threads,
                       transmitter_threads};

    launch_projection_kernel<scalar_t> (args, output.data<scalar_t> (), grid, block);
  }

  return output;
}

template <class scalar_t>
void launch (const Simulator<scalar_t> &args, scalar_t *output_buffer, int scatterer_blocks_factor,
             unsigned receiver_threads, unsigned transmitter_threads)
{
  const dim3 block = {get_properties ().maxThreadsPerBlock};
  if (scatterer_blocks_factor > 0)
  {
    const dim3 grid = {scatterer_blocks_factor * get_properties ().multiProcessorCount,
                       receiver_threads,
                       transmitter_threads};

    launch_projection_kernel<scalar_t> (args, output_buffer, grid, block);
  }
  else
  {
    const dim3 grid = {ceil_div<unsigned> (args.num_scatterers, block.x),
                       receiver_threads,
                       transmitter_threads};

    launch_projection_kernel<scalar_t> (args, output_buffer, grid, block);
  }
}

template void launch<float> (const Simulator<float> &args, float *output_buffer, int scatterer_blocks_factor,
                             unsigned receiver_threads, unsigned transmitter_threads);
template void launch<double> (const Simulator<double> &args, double *output_buffer, int scatterer_blocks_factor,
                              unsigned receiver_threads, unsigned transmitter_threads);

void reset_device ()
{
  checkCall (cudaDeviceReset ());
}

void synchronize ()
{
  checkCall (cudaDeviceSynchronize ());
}

template <class scalar_t>
static void bind_OpenBCSim (py::module &m, const std::string &type_string)
{
  /* Module docstring */
  m.doc () = "Interface to pass simulation parameters to CUDA kernel.";

  py::class_<Transducer<scalar_t>> (m, ("Transducer_" + type_string).c_str (), "Struct with transducer parameters.",
                                    py::dynamic_attr ())
      .def (py::init (with_signature<Transducer<scalar_t>> (create<scalar_t>)),
            "num_elements"_a,
            "num_subelements"_a,
            "division_factor"_a,
            "x"_a,
            "y"_a,
            "z"_a,
            "delay"_a,
            "apodization"_a,
            "center_frequency"_a)
      .def_readonly ("num_elements", &Transducer<scalar_t>::num_elements)
      .def_readonly ("num_subelements", &Transducer<scalar_t>::num_subelements)
      .def_readonly ("division_factor", &Transducer<scalar_t>::division_factor)
      .def_readonly ("center_frequency", &Transducer<scalar_t>::center_frequency);

  py::class_<Simulator<scalar_t>> (m, ("Simulator_" + type_string).c_str (), "Struct with simulation parameters.",
                                   py::dynamic_attr ())
      .def (py::init (with_signature<Simulator<scalar_t>> (create<scalar_t>)),
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

  m.def (("launch_" + type_string).c_str (), with_signature<at::Tensor> (launch<scalar_t>),
         "Simulate the response from a single transmission pulse.\n"
         "Outputs a Tensor of shape:\n"
         "  (args.receiver.num_subelements, args.num_time_samples).",
         "args"_a,
         "scatterer_blocks_factor"_a = 32,
         "receiver_threads"_a = 1,
         "transmitter_threads"_a = 1);
}

PYBIND11_MODULE (TORCH_EXTENSION_NAME, m)
{
  bind_OpenBCSim<float> (m, "float");
  bind_OpenBCSim<double> (m, "double");
  m.def ("reset_device", &reset_device);
  m.def ("synchronize", &synchronize);
  bind_DeviceProperties (m);
}

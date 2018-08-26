#include "openbcsim_module.h"
#include <string>

template <class scalar_t>
DLL_PUBLIC Transducer<scalar_t> create (unsigned num_elements,
                                        unsigned num_subelements,
                                        unsigned subdivision_factor,
                                        unsigned num_focal_points,
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
  /*  1 */ result.num_elements = num_elements;
  /*  2 */ result.num_subelements = num_subelements;
  /*  3 */ result.subdivision_factor = subdivision_factor;
  /*  4 */ result.num_focal_points = num_focal_points;
  /*  5 */ result.x = x.data<scalar_t> ();
  /*  6 */ result.y = y.data<scalar_t> ();
  /*  7 */ result.z = z.data<scalar_t> ();
  /*  8 */ result.delay = delay.data<scalar_t> ();
  /*  9 */ result.apodization = apodization.data<scalar_t> ();
  /* 10 */ result.center_frequency = center_frequency;

  return result;
}

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
  /*  6 */ result.tx = tx;
  /*  7 */ result.rx = rx;
  /*  8 */ result.num_time_samples = num_time_samples;
  /*  9 */ result.scatterer_x = scatterer_x.data<scalar_t> ();
  /* 10 */ result.scatterer_y = scatterer_y.data<scalar_t> ();
  /* 11 */ result.scatterer_z = scatterer_z.data<scalar_t> ();
  /* 12 */ result.scatterer_amplitude = scatterer_amplitude.data<scalar_t> ();
  /* 13 */ result.num_scatterers = num_scatterers;

  return result;
}

DLL_PUBLIC dim3 make_grid (int scatterer_blocks_factor, unsigned rx_blocks, unsigned tx_blocks, int device)
{
  return {scatterer_blocks_factor * get_properties (device).multiProcessorCount,
          rx_blocks,
          tx_blocks};
}

template <class scalar_t>
DLL_PUBLIC std::array<int64_t, 4> make_shape (const Simulator<scalar_t> &args)
{
  /*
    Must use `int64_t` to match `at::IntList` constructor.
    (The zeros() function takes a sizes argument of type `at::IntList`.)
    If get "out of memory" error, log arguments to make sure they converted properly...
  */
  /*
    Example: (50 focal points) x (200 elements) x (2 * (9e-2 m) / (1540 m/s) * (100e6 samples/s))
         = 50 * 200 * 2 * 9e-2 / 1540 * 100e6
         = 116,883,117 samples
         = 467,532,468 bytes (assuming float)
         = 467.5 MB
  */
  return {args.tx.num_focal_points,
          args.rx.num_elements,
          args.num_time_samples,
          2};
}

template <class scalar_t>
DLL_PUBLIC at::Tensor launch (const Simulator<scalar_t> &args, dim3 grid, dim3 block)
{
  const at::ScalarType s_type = (std::is_same_v<scalar_t, float>) ? (at::kFloat) : (at::kDouble);
  const auto sizes = make_shape (args);

  // Need to use `torch::` namespace instead of `at::` here. (Why...?)
  // https://github.com/pytorch/pytorch/issues/6103#issuecomment-377312709
  at::Tensor output = torch::CUDA (s_type).zeros (sizes);
  launch_projection_kernel<scalar_t> (args, output.data<scalar_t> (), grid, block);

  return output;
}

DLL_PUBLIC void reset_device ()
{
  checkCall (cudaDeviceReset ());
}

DLL_PUBLIC void synchronize ()
{
  checkCall (cudaDeviceSynchronize ());
}

template <class scalar_t>
static void bind_openbcsim (py::module &m, const std::string &type_string)
{
  /* Module docstring */
  m.doc () = "Interface to pass simulation parameters to CUDA kernel.";

  py::class_<Transducer<scalar_t>> (m, ("Transducer_" + type_string).c_str (), "Struct with transducer parameters.",
                                    py::dynamic_attr ())
      .def (py::init (with_signature<Transducer<scalar_t>> (create<scalar_t>)),
            "num_elements"_a,
            "num_subelements"_a,
            "subdivision_factor"_a,
            "num_focal_points"_a,
            "x"_a,
            "y"_a,
            "z"_a,
            "delay"_a,
            "apodization"_a,
            "center_frequency"_a)
      .def_readonly ("num_elements", &Transducer<scalar_t>::num_elements)
      .def_readonly ("num_subelements", &Transducer<scalar_t>::num_subelements)
      .def_readonly ("subdivision_factor", &Transducer<scalar_t>::subdivision_factor)
      .def_readonly ("num_focal_points", &Transducer<scalar_t>::num_focal_points)
      .def_readonly ("center_frequency", &Transducer<scalar_t>::center_frequency);

  py::class_<Simulator<scalar_t>> (m, ("Simulator_" + type_string).c_str (), "Struct with simulation parameters.",
                                   py::dynamic_attr ())
      .def (py::init (with_signature<Simulator<scalar_t>> (create<scalar_t>)),
            "sampling_frequency"_a,
            "decimation"_a,
            "scan_depth"_a,
            "speed_of_sound"_a,
            "attenuation"_a,
            "tx"_a,
            "rx"_a,
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
      .def_readonly ("tx", &Simulator<scalar_t>::tx)
      .def_readonly ("rx", &Simulator<scalar_t>::rx)
      .def_readonly ("num_time_samples", &Simulator<scalar_t>::num_time_samples)
      .def_readonly ("num_scatterers", &Simulator<scalar_t>::num_scatterers);

  m.def ("launch", launch<scalar_t>,
         R"""(\
Runs the `projection_kernel` CUDA function.
The kernel projects each scatterer point to its corresponding point in time
(at which its echo is heard) based on its distance to each transducer element
(projects to different time points for different transducer elements).
Outputs a Tensor of shape:
  (args.tx.num_focal_points,
   args.rx.num_elements,
   args.num_time_samples,
   2).
)""",
         "args"_a,
         "grid"_a = make_grid (),
         "block"_a = dim3{get_properties ().maxThreadsPerBlock});

  m.def ("make_shape", make_shape<scalar_t>);
}

PYBIND11_MODULE (TORCH_EXTENSION_NAME, m)
{
  // `dim3` definition must go before `bind_openbcsim()`
  py::class_<dim3> (m, "dim3", py::dynamic_attr ())
      .def (py::init<unsigned, unsigned, unsigned> (), "x"_a = 1, "y"_a = 1, "z"_a = 1)
      .def (py::init ([](py::iterable iterable) {
              auto iter = py::iter (iterable);
              dim3 result;
              result.x = py::cast<unsigned> (*iter);
              ++iter;
              result.y = py::cast<unsigned> (*iter);
              ++iter;
              result.z = py::cast<unsigned> (*iter);
              ++iter;
              return result;
            }),
            "iterable"_a)
      .def_readwrite ("x", &dim3::x)
      .def_readwrite ("y", &dim3::y)
      .def_readwrite ("z", &dim3::z)
      .def ("__repr__", [](const dim3 &self) {
        return py::str (py::cast (std::vector<unsigned>{self.x, self.y, self.z}));
      })
      .def ("__getitem__", [](const dim3 &self, int index) {
        switch (index)
        {
        case 0:
          return self.x;
        case 1:
          return self.y;
        case 2:
          return self.z;
        default:
          throw std::out_of_range ("Index " + std::to_string (index) + " out of bounds (should be in [0,2]).");
        }
      });

  bind_openbcsim<float> (m, "float");
  bind_openbcsim<double> (m, "double");

  m.def ("make_grid", &make_grid,
         "scatterer_blocks_factor"_a = 32,
         "rx_blocks"_a = 1,
         "tx_blocks"_a = 1,
         "device"_a = 0);

  m.def ("reset_device", &reset_device);
  m.def ("synchronize", &synchronize);

  bind_DeviceProperties (m);
}

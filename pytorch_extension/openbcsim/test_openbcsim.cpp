#include "base.h" // Must go first

#include "openbcsim_module.h"
#include "pretty_print.h"
#include <iterator>
#include <memory>
#include <pybind11/embed.h> // everything needed for embedding Python interpreter

template <class scalar_t>
void with_pytorch (int num_scatterers, unsigned num_elements, int scatterer_blocks_factor, unsigned receiver_threads,
                   unsigned transmitter_threads)
{
  using namespace std;
  using namespace pretty;
  const at::ScalarType s_type = (is_same_v<scalar_t, float>) ? (at::kFloat) : (at::kDouble);

  // Start timer
  toc ();
  start = tic;

  // Initializing Python Interpreter, torch, and torch.cuda
  py::scoped_interpreter guard{}; // start the interpreter and keep it alive

  timestamp () << "  launched interpreter" << endl;
  py::object torch = py::module::import ("torch");
  timestamp () << "  imported torch (version: " << py::cast<string> (torch.attr ("__version__")) << ")" << endl;
  // torch.attr ("cuda").attr ("init") ();
  // timestamp () << "  torch.cuda.init ()" << endl;

  py::object openbcsim = py::module::import ("openbcsim");
  timestamp () << "  imported openbcsim (doc: " << py::cast<string> (openbcsim.attr ("__doc__")) << ")" << endl;
  // for (auto item : )
  // {
  auto tmp = DeviceProperties (get_properties ());
  cout << py::str (py::cast (tmp).attr ("major")) << endl;
  cout << tmp.ECCEnabled << " " << tmp.major << "." << tmp.minor << " " << tmp.name << endl;
  // }
  return;

  // Initializing Transducer
  block ("Num elems.") << "  " << num_elements << endl;
  unsigned num_subelements = num_elements;
  unsigned division_factor = 1;
  unsigned num_scans = 1;
  at::Tensor x = torch::CUDA (s_type).arange (static_cast<int> (num_elements));
  x -= (num_elements - 1) / 2.0;
  x *= .5e-3;
  at::Tensor y = torch::CUDA (s_type).zeros ({num_elements});
  at::Tensor z = torch::CUDA (s_type).zeros ({num_elements});
  at::Tensor delay = torch::CUDA (s_type).zeros ({num_elements});
  at::Tensor apodization = torch::CUDA (s_type).ones ({num_elements});
  scalar_t center_frequency = 3.5e6;
  Transducer<scalar_t> tx = create<scalar_t> (num_elements, num_subelements, division_factor, num_scans, x, y, z, delay,
                                              apodization, center_frequency);
  timestamp () << "  created Transducer" << endl;

  // Initializing Simulator
  scalar_t sampling_frequency = 100e6;
  unsigned decimation = 10;
  scalar_t scan_depth = 9e-2;
  scalar_t speed_of_sound = 1540;
  scalar_t attenuation = .7;
  Transducer<scalar_t> &transmitter = tx;
  Transducer<scalar_t> &receiver = tx;
  unsigned num_time_samples = static_cast<int> (2 * scan_depth / speed_of_sound * sampling_frequency + .5);
  at::Tensor scatterer_x = torch::CUDA (s_type).arange (num_scatterers);
  scatterer_x -= (num_scatterers - 1) / 2.0;      // Center at 0
  scatterer_x *= 4e-2 / (num_scatterers - 1) * 2; // Adjust spacing
  at::Tensor scatterer_y = torch::CUDA (s_type).zeros ({num_scatterers});
  at::Tensor scatterer_z = torch::CUDA (s_type).arange (num_scatterers);
  scatterer_z *= 9e-2 / (num_scatterers - 1); // Adjust spacing
  at::Tensor scatterer_amplitude = torch::CUDA (s_type).ones ({num_scatterers});
  Simulator<scalar_t> sim = create<scalar_t> (sampling_frequency, decimation, scan_depth, speed_of_sound, attenuation,
                                              transmitter, receiver, num_time_samples, scatterer_x, scatterer_y,
                                              scatterer_z, scatterer_amplitude, num_scatterers);
  timestamp () << "  created Simulator" << endl;

  // Launch kernel
  auto result = launch<scalar_t> (sim, scatterer_blocks_factor, receiver_threads, transmitter_threads);
  timestamp () << "  launched" << endl;

  // Verbose stats
  auto threads_per_block = get_properties ().maxThreadsPerBlock;
  auto grid_x = (num_scatterers + threads_per_block - 1) / threads_per_block;
  if (scatterer_blocks_factor > 0)
  {
    grid_x = scatterer_blocks_factor * get_properties ().multiProcessorCount;
  }
  auto grid_y = receiver_threads;
  auto grid_z = transmitter_threads;

  block ("Out elems.", 11) << "  {}"_s.format (result.sizes ()) << endl;
  block ("Time samps.", 11) << "  {:,}"_s.format (num_time_samples) << endl;
  block ("Scatterers", 11) << "  {:,}"_s.format (num_scatterers) << endl;
  block ("Threads", 11) << "  {:,}"_s.format (static_cast<long long> (grid_x) * grid_y * grid_z * threads_per_block)
                        << endl;
  block ("Blocks", 11) << "  {:,}"_s.format (static_cast<long long> (grid_x) * grid_y * grid_z) << endl;
  block ("Rx threads", 11) << "  {:,}"_s.format (receiver_threads) << endl;
  block ("Tx threads", 11) << "  {:,}"_s.format (transmitter_threads) << endl;

  // Wait for kernel
  synchronize ();
  timestamp () << "  synchronized" << endl;

  // Print output
  block ("Result") << "  " << py::str (py::cast (result)) << endl;
  timestamp () << "  printed" << endl;

  // Report total time
  tic = start;
  timestamp () << "  TOTAL" << endl;
}

template <class scalar_t>
void without_pytorch (int num_scatterers, unsigned num_elements, int scatterer_blocks_factor, unsigned receiver_threads,
                      unsigned transmitter_threads)
{
  using namespace std;
  using namespace pretty;
  save (cout, DEFAULT);
  cout.imbue (LOCALE);

  static const auto cuda_deleter = [](scalar_t *pointer) {
    using namespace std;
    cout << "Cuda pointer freed!" << endl;
    cudaFree (pointer);
  };

  struct CUDA_buffer
  {
    size_t length;
    unique_ptr<scalar_t, decltype (cuda_deleter)> gpu{nullptr, cuda_deleter};
    unique_ptr<scalar_t[]> cpu;
    scalar_t *cpu_end;
    CUDA_buffer (size_t length_)
        : length (length_),
          cpu{new scalar_t[length]},
          cpu_end{cpu.get () + length}
    {
      void *gpu_;
      checkCall (cudaMalloc (&gpu_, length * sizeof (scalar_t)));
      gpu.reset ((scalar_t *)gpu_);
    }
    CUDA_buffer (size_t length, scalar_t value) : CUDA_buffer (length)
    {
      for (int i = 0; i < length; i++)
        cpu[i] = value;
    }
    static CUDA_buffer arange (size_t length)
    {
      auto result = CUDA_buffer (length);
      for (int i = 0; i < result.length; i++)
        result.cpu[i] = i;
      return result;
    }
    static CUDA_buffer zeros (size_t length)
    {
      return CUDA_buffer (length, 0);
    }
    static CUDA_buffer ones (size_t length)
    {
      return CUDA_buffer (length, 1);
    }
    CUDA_buffer &operator-= (scalar_t value)
    {
      for (int i = 0; i < length; i++)
        cpu[i] -= value;
      return *this;
    }
    CUDA_buffer &operator*= (scalar_t value)
    {
      for (int i = 0; i < length; i++)
        cpu[i] *= value;
      return *this;
    }
    void to_device (cudaStream_t stream = 0)
    {
      checkCall (cudaMemcpyAsync (gpu.get (), cpu.get (), length * sizeof (scalar_t), cudaMemcpyDefault, stream));
    }
    void to_host (cudaStream_t stream = 0)
    {
      checkCall (cudaMemcpyAsync (cpu.get (), gpu.get (), length * sizeof (scalar_t), cudaMemcpyDefault, stream));
    }
  };

  // Start timer
  toc ();
  start = tic;

  // Initializing Transducer
  block ("Num elems.") << "  " << num_elements << endl;
  Transducer<scalar_t> tx;
  tx.num_subelements = num_elements;
  tx.division_factor = 1;
  tx.num_scans = 1;
  auto X = CUDA_buffer::arange (num_elements);
  X -= (num_elements - 1) / 2.0;
  X *= .5e-3;
  tx.x = X.gpu.get ();
  auto Y = CUDA_buffer::zeros (num_elements);
  tx.y = Y.gpu.get ();
  auto Z = CUDA_buffer::zeros (num_elements);
  tx.z = Z.gpu.get ();
  auto DELAY = CUDA_buffer::zeros (num_elements);
  tx.delay = DELAY.gpu.get ();
  auto APODIZATION = CUDA_buffer::ones (num_elements);
  tx.apodization = APODIZATION.gpu.get ();
  tx.center_frequency = 3.5e6;
  X.to_device ();
  Y.to_device ();
  Z.to_device ();
  DELAY.to_device ();
  APODIZATION.to_device ();
  timestamp () << "  created Transducer" << endl;

  // Initializing Simulator
  Simulator<scalar_t> sim;
  sim.sampling_frequency = 100e6;
  sim.decimation = 10;
  sim.scan_depth = 9e-2;
  sim.speed_of_sound = 1540;
  sim.attenuation = .7;
  sim.transmitter = tx;
  sim.receiver = tx;
  sim.num_time_samples = static_cast<int> (2 * sim.scan_depth / sim.speed_of_sound * sim.sampling_frequency + .5);
  auto SCATTERER_X = CUDA_buffer::arange (num_scatterers);
  SCATTERER_X -= (num_scatterers - 1) / 2.0;      // Center at 0
  SCATTERER_X *= 4e-2 / (num_scatterers - 1) * 2; // Adjust spacing
  sim.scatterer_x = SCATTERER_X.gpu.get ();
  auto SCATTERER_Y = CUDA_buffer::zeros (num_scatterers);
  sim.scatterer_y = SCATTERER_Y.gpu.get ();
  auto SCATTERER_Z = CUDA_buffer::arange (num_scatterers);
  SCATTERER_Z *= 9e-2 / (num_scatterers - 1); // Adjust spacing
  sim.scatterer_z = SCATTERER_Z.gpu.get ();
  auto SCATTERER_AMPLITUDE = CUDA_buffer::zeros (num_scatterers);
  sim.scatterer_amplitude = SCATTERER_AMPLITUDE.gpu.get ();
  sim.num_scatterers = num_scatterers;
  SCATTERER_X.to_device ();
  SCATTERER_Y.to_device ();
  SCATTERER_Z.to_device ();
  SCATTERER_AMPLITUDE.to_device ();
  timestamp () << "  created Simulator" << endl;

  // Launch kernel
  auto RESULT = CUDA_buffer::zeros (sim.transmitter.num_scans * sim.receiver.num_elements * sim.num_time_samples * 2);
  RESULT.to_device ();
  launch<scalar_t> (sim, RESULT.gpu.get (), scatterer_blocks_factor, receiver_threads, transmitter_threads);
  timestamp () << "  launched" << endl;

  // Verbose stats
  auto threads_per_block = get_properties ().maxThreadsPerBlock;
  auto grid_x = (num_scatterers + threads_per_block - 1) / threads_per_block;
  if (scatterer_blocks_factor > 0)
  {
    grid_x = scatterer_blocks_factor * get_properties ().multiProcessorCount;
  }
  auto grid_y = receiver_threads;
  auto grid_z = transmitter_threads;

  block ("Out elems.", 11) << "  " << RESULT.length << endl;
  block ("Time samps.", 11) << "  " << sim.num_time_samples << endl;
  block ("Scatterers", 11) << "  " << sim.num_scatterers << endl;
  block ("Threads", 11) << "  " << static_cast<long long> (grid_x) * grid_y * grid_z * threads_per_block << endl;
  block ("Blocks", 11) << "  " << static_cast<long long> (grid_x) * grid_y * grid_z << endl;
  block ("Rx threads", 11) << "  " << receiver_threads << endl;
  block ("Tx threads", 11) << "  " << transmitter_threads << endl;

  // Wait for kernel
  synchronize ();
  timestamp () << "  kernel completed" << endl;
  RESULT.to_host ();
  synchronize ();
  timestamp () << "  result copied" << endl;

  // Print output
  block ("Result") << "  [";
  copy (RESULT.cpu.get (), RESULT.cpu.get () + 5, ostream_iterator<scalar_t> (cout, ", "));
  copy (RESULT.cpu_end - 5, RESULT.cpu_end, ostream_iterator<scalar_t> (cout, ", "));
  cout << "]" << endl;
  timestamp () << "  printed" << endl;

  // Report total time
  tic = start;
  timestamp () << "  TOTAL" << endl;
}

int main (int argc, const char *argv[])
{
  using namespace std;
  cout << endl;

  if (argc < 4)
  {
    cerr << endl;
    cerr << "  Usage: " << argv[0] << " <use_pytorch> <num_scatterers> <num_elements> [scatterer_blocks_factor]"
                                      " [receiver_threads] [transmitter_threads]"
         << endl;
    return 1;
  }
  auto arg = &argv[1];
  bool use_pytorch = stoi (*arg++);
  int num_scatterers = stoi (*arg++);
  int num_elements = stoi (*arg++);
  int scatterer_blocks_factor = 32;
  if (arg < argv + argc)
  {
    scatterer_blocks_factor = stoi (*arg++);
  }
  int receiver_threads = 1;
  if (arg < argv + argc)
  {
    receiver_threads = stoi (*arg++);
  }
  int transmitter_threads = 1;
  if (arg < argv + argc)
  {
    transmitter_threads = stoi (*arg++);
  }

  if (use_pytorch)
  {
    cout << "  WITH PYTORCH\n"
         << endl;
    with_pytorch<float> (num_scatterers, num_elements, scatterer_blocks_factor, receiver_threads, transmitter_threads);
    cout << "\n============================================================\n"
         << endl;
  }
  else
  {
    cout << "  WITHOUT\n"
         << endl;
    without_pytorch<float> (num_scatterers, num_elements, scatterer_blocks_factor, receiver_threads, transmitter_threads);
  }
}

/*
BUILD COMMANDS:
==============
python fast_build.py -O
python fast_build.py -Oe -cpp_in test_module.cpp -cpp_out build\test_module.o -link_in build\openbcsim_kernel.o build\openbcsim_module.o build\test_module.o -link_out build\test_module.exe

ADD THE FOLLOWING TO PATH:
==========================
set path=%path%;C:\Users\spenc\Anaconda3\pkgs\pytorch-0.4.0-py36_cuda80_cudnn7he774522_1\Lib\site-packages\torch
set path=%path%;C:\Users\spenc\Anaconda3\pkgs\pytorch-0.4.0-py36_cuda80_cudnn7he774522_1\Lib\site-packages\torch\lib;C:\Users\spenc\Anaconda3\Library\bin
set path=%path%;C:\Users\spenc\Anaconda3\libs

RUN:
====
build\test_module.exe -h
*/
/*
Copyright (c) 2015, Sigurd Storve
All rights reserved.

Licensed under the BSD license.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
* Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.
* Neither the name of the <organization> nor the
names of its contributors may be used to endorse or promote products
derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
#ifdef BCSIM_ENABLE_CUDA
#include "GpuAlgorithm.hpp"
#include "../bspline.hpp"
#include "../discrete_hilbert_mask.hpp"
#include "../fft.hpp"           // for next_power_of_two
#include "common_definitions.h" // for MAX_NUM_CUDA_STREAMS and MAX_SPLINE_DEGREE
#include "common_utils.hpp"     // for compute_num_rf_samples
#include "cuda_debug_utils.h"
#include "cuda_helpers.h"
#include "cuda_kernels_c_interface.h"
#include "cufft_helpers.h"
#include <complex>
#include <cuda.h>
#include <iomanip> // for setprecision
#include <iostream>
#include <stdexcept>
#include <tuple> // for std::tie

namespace bcsim
{
GpuAlgorithm::GpuAlgorithm ()
    : m_use_delay_compensation (true),
      // The below initializations are necessary
      // because VS2013 has incorrect semantics for
      // value-initialization.
      // i.e. Stuff isn't 0/false/nullptr when it should be.
      // See: https://stackoverflow.com/q/27668269/3624264
      m_scan_sequence_configured (false),
      m_excitation_configured (false),
      m_use_elev_hack (false),
      m_param_cuda_device_no (0),
      m_can_change_cuda_device (true),
      m_store_kernel_details (false),
      m_device_random_buffer (nullptr)
{
  // ensure that CUDA device properties is stored
  save_cuda_device_properties ();

  // Record maximum number of threads per block and number of CUDA asynchronoous copy streams
  // Must be called AFTER save_cuda_device_properties()
  init_from_hardware_constraints ();

  create_dummy_lut_profile ();
}

int GpuAlgorithm::get_num_cuda_devices () const
{
  int device_count;
  cudaErrorCheck (cudaGetDeviceCount (&device_count));
  return device_count;
}

void GpuAlgorithm::set_parameter (const std::string &key, const std::string &value)
{
  if (key == "gpu_device")
  {
    if (!m_can_change_cuda_device)
    {
      throw std::runtime_error ("cannot change CUDA device now");
    }
    const auto device_count = get_num_cuda_devices ();
    const auto device_no = std::stoi (value);
    if (device_no < 0 || device_no >= device_count)
    {
      throw std::runtime_error ("illegal device number");
    }
    m_param_cuda_device_no = device_no;
    cudaErrorCheck (cudaSetDevice (m_param_cuda_device_no));
    save_cuda_device_properties ();
  }
  else if (key == "cuda_streams")
  {
    const auto num_streams = std::stoi (value);
    if (num_streams > MAX_NUM_CUDA_STREAMS)
    {
      throw std::runtime_error ("number of CUDA streams exceeds MAX_NUM_CUDA_STREAMS");
    }
    if (num_streams <= 0)
    {
      throw std::runtime_error ("number of CUDA streams must be more than zero");
    }
    m_param_num_cuda_streams = num_streams;
  }
  else if (key == "threads_per_block")
  {
    const auto threads_per_block = std::stoi (value);
    if (threads_per_block <= 0)
    {
      throw std::runtime_error ("invalid number of threads per block");
    }
    else if (threads_per_block > m_cur_device_prop.maxThreadsPerBlock)
    {
      const std::string msg = "Requested threads per block (" +
                              std::to_string (threads_per_block) +
                              ") exceeds maximum allowed by device (" +
                              std::to_string (m_cur_device_prop.maxThreadsPerBlock) +
                              ").";
      throw std::runtime_error (msg);
    }
    m_param_threads_per_block = threads_per_block;
  }
  else if (key == "store_kernel_details")
  {
    if ((value == "on") || (value == "true"))
    {
      m_store_kernel_details = true;
    }
    else if ((value == "off") || (value == "false"))
    {
      m_store_kernel_details = false;
    }
    else
    {
      throw std::runtime_error ("invalid value");
    }
  }
  else if (key == "verbose")
  {
    const auto verbose = std::stoi (value);
    m_param_verbose = m_log_object->verbose = verbose;
  }
  else if (key == "use_elev_hack")
  {
    if ((value == "on") || (value == "true"))
    {
      m_use_elev_hack = true;
    }
    else if ((value == "off") || (value == "false"))
    {
      m_use_elev_hack = false;
    }
    else
    {
      throw std::runtime_error ("invalid value");
    }
  }
  else if (key == "use_delay_compensation")
  {
    if ((value == "on") || (value == "true"))
    {
      m_use_delay_compensation = true;
    }
    else if ((value == "off") || (value == "false"))
    {
      m_use_delay_compensation = false;
    }
    else
    {
      throw std::runtime_error ("invalid value");
    }
  }
  else
  {
    BaseAlgorithm::set_parameter (key, value);
  }
}

void GpuAlgorithm::create_cuda_stream_wrappers (int num_streams)
{
  m_stream_wrappers.clear ();
  for (int i = 0; i < num_streams; i++)
  {
    m_stream_wrappers.push_back (std::move (CudaStreamRAII::u_ptr (new CudaStreamRAII)));
  }
  m_can_change_cuda_device = false;
}

void GpuAlgorithm::save_cuda_device_properties ()
{
  const auto num_devices = get_num_cuda_devices ();
  if (m_param_cuda_device_no < 0 || m_param_cuda_device_no >= num_devices)
  {
    throw std::runtime_error ("illegal CUDA device number");
  }
  cudaErrorCheck (cudaGetDeviceProperties (&m_cur_device_prop, m_param_cuda_device_no));

  const auto &p = m_cur_device_prop;
  std::stringstream ss;
  ss << std::endl;
  ss << "=== CUDA Device " << m_param_cuda_device_no << ": " << p.name << " ===" << std::endl;
  ss << "  Compute capability:         " << p.major << "." << p.minor << std::endl;
  ss << "  ECCEnabled:                 " << p.ECCEnabled << std::endl;
  ss << "  asyncEngineCount:           " << p.asyncEngineCount << std::endl;
  ss << "  canMapHostMemory:           " << p.canMapHostMemory << std::endl;
  ss << "  clockRate:                  " << p.clockRate << std::endl;
  ss << "  computeMode:                " << p.computeMode << std::endl;
  ss << "  concurrentKernels:          " << p.concurrentKernels << std::endl;
  ss << "  integrated:                 " << p.integrated << std::endl;
  ss << "  kernelExecTimeoutEnabled:   " << p.kernelExecTimeoutEnabled << std::endl;
  ss << "  l2CacheSize:                " << p.l2CacheSize << std::endl;
  ss << "  maxGridSize:                <" << p.maxGridSize[0] << ", " << p.maxGridSize[1] << ", " << p.maxGridSize[2] << ">" << std::endl;
  ss << "  maxThreadsPerBlock:         " << p.maxThreadsPerBlock << std::endl;
  ss << "  memoryBusWidth:             " << p.memoryBusWidth << std::endl;
  ss << "  multiProcessorCount:        " << p.multiProcessorCount << std::endl;
  ss << "  totalGlobMem:               " << p.totalGlobalMem;
  m_log_object->write (ILog::INFO, ss.str ());
}

void GpuAlgorithm::init_from_hardware_constraints ()
{
  m_param_threads_per_block = m_cur_device_prop.maxThreadsPerBlock;
  m_param_num_cuda_streams = m_cur_device_prop.asyncEngineCount;
}

void GpuAlgorithm::simulate_lines (std::vector<std::vector<std::complex<float>>> & /*out*/ rf_lines)
{
  throw_if_not_configured ();
  m_can_change_cuda_device = false;

  if (m_stream_wrappers.size () == 0)
  {
    create_cuda_stream_wrappers (m_param_num_cuda_streams);
  }

  if (m_store_kernel_details)
  {
    m_debug_data.clear ();
  }

  auto num_lines = m_scan_sequence->get_num_lines ();
  if (num_lines < 1)
  {
    throw std::runtime_error ("No scanlines in scansequence");
  }

  if (m_cur_beam_profile_type == BeamProfileType::NOT_CONFIGURED)
  {
    throw std::runtime_error ("No beam profile is configured");
  }

  if (m_param_noise_amplitude > 0.0f)
  {
    const size_t num_random_numbers = num_lines * m_rf_line_num_samples * 2; // for real- and imaginary part.
    const size_t num_bytes_needed = num_random_numbers * sizeof (float);
    if (num_random_numbers % 2 != 0)
      throw std::runtime_error ("Number of random samples must be even");
    if ((m_device_random_buffer == nullptr) || (m_device_random_buffer->get_num_bytes () != num_bytes_needed))
    {
      m_log_object->write (ILog::INFO, "Reallocating device memory for random noise samples");
      m_device_random_buffer = DeviceBufferRAII<float>::u_ptr (new DeviceBufferRAII<float> (num_bytes_needed));
    }

    // recreate random numbers
    curandErrorCheck (curandGenerateNormal (m_device_rng.get (), m_device_random_buffer->data (), num_random_numbers, 0.0f, m_param_noise_amplitude));
    cudaErrorCheck (cudaDeviceSynchronize ());
  }

  // TODO: If all beams have the same timestamp, first render to fixed scatterers
  // in device memory and then simulate with the fixed algorithm
  bool use_optimized_spline_kernel = false;
  if (m_scan_sequence->all_timestamps_equal && (m_device_spline_datasets.get_num_datasets () > 0))
  {
    const auto timestamp = m_scan_sequence->get_scanline (0).get_timestamp ();
    m_device_rendered_spline_datasets.render (m_device_spline_datasets, timestamp);
    use_optimized_spline_kernel = true;
    cudaErrorCheck (cudaDeviceSynchronize ());
  }

  for (int beam_no = 0; beam_no < num_lines; beam_no++)
  {
    auto stream_no = beam_no % m_param_num_cuda_streams;
    auto cur_stream = m_stream_wrappers[stream_no]->get ();

    std::unique_ptr<EventTimerRAII> event_timer;
    if (m_store_kernel_details)
    {
      event_timer = std::unique_ptr<EventTimerRAII> (new EventTimerRAII (cur_stream));
      m_debug_data["stream_numbers"].push_back (static_cast<double> (stream_no));
      event_timer->restart ();
    }

    m_log_object->write (ILog::DEBUG, "beam_no = " + std::to_string (beam_no) + ", stream_no = " + std::to_string (stream_no));

    auto scanline = m_scan_sequence->get_scanline (beam_no);
    const auto threads_per_line = m_param_threads_per_block; // Fine tune with profiler if needed
    auto rf_ptr = m_device_time_proj->data () + beam_no * m_rf_line_num_samples;

    // clear time projections (safer than cudaMemsetAsync)
    const auto complex_zero = make_cuComplex (0.0f, 0.0f);
    if (m_store_kernel_details)
    {
      event_timer->restart ();
    }
    m_log_object->write (ILog::DEBUG, "launch_MemsetKernel...");
    launch_MemsetKernel<cuComplex> (m_rf_line_num_samples / threads_per_line, threads_per_line, cur_stream, rf_ptr, complex_zero, m_rf_line_num_samples);

    if (m_store_kernel_details)
    {
      const auto elapsed_ms = static_cast<double> (event_timer->stop ());
      m_debug_data["kernel_memset_ms"].push_back (elapsed_ms);
      event_timer->restart ();
    }

    // project fixed scatterers
    for (size_t dset_idx = 0; dset_idx < m_device_fixed_datasets.get_num_datasets (); dset_idx++)
    {
      const auto device_dataset = m_device_fixed_datasets.get_dataset (dset_idx);
      const auto num_scatterers = device_dataset->get_num_scatterers ();
      const auto num_blocks = round_up_div (num_scatterers, m_param_threads_per_block);
      if (num_blocks > m_cur_device_prop.maxGridSize[0])
      {
        throw std::runtime_error ("required number of x-blocks is larger than device supports (fixed scatterers)");
      }
      m_log_object->write (ILog::DEBUG, "fixed_projection_kernel...");
      fixed_projection_kernel (stream_no, scanline, num_blocks, rf_ptr, device_dataset);

      if (m_store_kernel_details)
      {
        const auto elapsed_ms = static_cast<double> (event_timer->stop ());
        m_debug_data["fixed_projection_kernel_ms"].push_back (elapsed_ms);
        event_timer->restart ();
      }
    }

    // project spline scatterers
    if (use_optimized_spline_kernel)
    {
      for (size_t dset_idx = 0; dset_idx < m_device_rendered_spline_datasets.get_num_datasets (); dset_idx++)
      {
        const auto device_dataset = m_device_rendered_spline_datasets.get_dataset (dset_idx);
        const auto num_scatterers = device_dataset->get_num_scatterers ();
        const auto num_blocks = round_up_div (num_scatterers, m_param_threads_per_block);
        if (num_blocks > m_cur_device_prop.maxGridSize[0])
        {
          throw std::runtime_error ("required number of x-blocks is larger than device supports (spline scatterers)");
        }
        fixed_projection_kernel (stream_no, scanline, num_blocks, rf_ptr, device_dataset);
      }
    }
    else
    {
      for (size_t dset_idx = 0; dset_idx < m_device_spline_datasets.get_num_datasets (); dset_idx++)
      {
        const auto device_dataset = m_device_spline_datasets.get_dataset (dset_idx);
        const auto num_scatterers = device_dataset->get_num_scatterers ();
        const auto num_blocks = round_up_div (num_scatterers, m_param_threads_per_block);
        if (num_blocks > m_cur_device_prop.maxGridSize[0])
        {
          throw std::runtime_error ("required number of x-blocks is larger than device supports (spline scatterers)");
        }
        spline_projection_kernel (stream_no, scanline, num_blocks, rf_ptr, device_dataset);

        if (m_store_kernel_details)
        {
          const auto elapsed_ms = static_cast<double> (event_timer->stop ());
          m_debug_data["spline_projection_kernel_ms"].push_back (elapsed_ms);
        }
      }
    }
  }

  // block to ensure that all operations are completed
  m_log_object->write (ILog::DEBUG, "cudaDeviceSynchronize 1...");
  cudaErrorCheck (cudaDeviceSynchronize ());

  if (m_param_noise_amplitude > 0.0f)
  {
    const auto threads_per_line = m_param_threads_per_block; // Fine tune with profiler if needed
    const auto num_samples = num_lines * m_rf_line_num_samples;
    const auto complex_ptr = reinterpret_cast<cuComplex *> (m_device_random_buffer->data ());
    cudaStream_t stream = 0;
    launch_AddNoiseKernel (num_samples / threads_per_line, threads_per_line, stream, complex_ptr, m_device_time_proj->data (), num_samples);
    cudaErrorCheck (cudaDeviceSynchronize ());
  }

  std::unique_ptr<EventTimerRAII> event_timer;
  if (m_store_kernel_details)
  {
    event_timer = std::unique_ptr<EventTimerRAII> (new EventTimerRAII (0));
    event_timer->restart ();
  }

  // in-place batched forward FFT, using default stream 0
  m_log_object->write (ILog::DEBUG, "cftExecC2C FORWARD...");
  cufftErrorCheck (cufftExecC2C (m_fft_plan->get (), m_device_time_proj->data (), m_device_time_proj->data (), CUFFT_FORWARD));
  if (m_store_kernel_details)
  {
    const auto elapsed_ms = static_cast<double> (event_timer->stop ());
    m_debug_data["kernel_forward_fft_ms"].push_back (elapsed_ms);
  }
  m_log_object->write (ILog::DEBUG, "cudaDeviceSynchronize 2...");
  cudaErrorCheck (cudaDeviceSynchronize ());

  // Multiply kernel
  for (int beam_no = 0; beam_no < num_lines; beam_no++)
  {
    size_t stream_no = beam_no % m_param_num_cuda_streams;
    auto cur_stream = m_stream_wrappers[stream_no]->get ();

    std::unique_ptr<EventTimerRAII> event_timer;
    if (m_store_kernel_details)
    {
      event_timer = std::unique_ptr<EventTimerRAII> (new EventTimerRAII (cur_stream));
      event_timer->restart ();
    }

    auto rf_ptr = m_device_time_proj->data () + beam_no * m_rf_line_num_samples;

    // multiply with FFT of impulse response w/Hilbert transform
    const auto threads_per_line = m_param_threads_per_block; // Fine tune with profiler if needed
    m_log_object->write (ILog::DEBUG, "launch_MultiplyFftKernel...");
    launch_MultiplyFftKernel<true> (m_rf_line_num_samples / threads_per_line, threads_per_line, cur_stream, rf_ptr, m_device_excitation_fft->data (), m_rf_line_num_samples);
    if (m_store_kernel_details)
    {
      const auto elapsed_ms = static_cast<double> (event_timer->stop ());
      m_debug_data["kernel_multiply_fft_ms"].push_back (elapsed_ms);
    }
  }

  // In-place batched backward FFT, using default stream 0
  if (m_store_kernel_details)
  {
    event_timer->restart ();
  }
  m_log_object->write (ILog::DEBUG, "cudaDeviceSynchronize 3...");
  cudaErrorCheck (cudaDeviceSynchronize ());

  m_log_object->write (ILog::DEBUG, "cftExecC2C INVERSE...");
  cufftErrorCheck (cufftExecC2C (m_fft_plan->get (), m_device_time_proj->data (), m_device_time_proj->data (), CUFFT_INVERSE));

  if (m_store_kernel_details)
  {
    const auto elapsed_ms = static_cast<double> (event_timer->stop ());
    m_debug_data["kernel_inverse_fft_ms"].push_back (elapsed_ms);
  }

  m_log_object->write (ILog::DEBUG, "cudaDeviceSynchronize 4...");
  cudaErrorCheck (cudaDeviceSynchronize ());

  rf_lines.clear ();
  int delay_compensation_num_samples = 0;
  if (m_use_delay_compensation)
  {
    delay_compensation_num_samples = m_excitation.center_index;
  }

  for (int beam_no = 0; beam_no < num_lines; beam_no++)
  {
    size_t stream_no = beam_no % m_param_num_cuda_streams;
    auto cur_stream = m_stream_wrappers[stream_no]->get ();

    // Compute current offset into device buffer.
    // Account for the delay introduced by convolving with the excitation if `m_use_delay_compensation`.
    auto rf_ptr = m_device_time_proj->data () + beam_no * m_rf_line_num_samples + delay_compensation_num_samples;

    std::unique_ptr<EventTimerRAII> event_timer;
    if (m_store_kernel_details)
    {
      event_timer = std::unique_ptr<EventTimerRAII> (new EventTimerRAII (cur_stream));
      event_timer->restart ();
    }

    // IQ demodulation and decimation (by m_radial_decimation).
    const auto threads_per_line = m_param_threads_per_block; // Fine tune with profiler if needed
    const auto f_demod = m_excitation.demod_freq;
    const float norm_f_demod = f_demod / m_excitation.sampling_frequency;
    const float PI = static_cast<float> (4.0 * std::atan (1));
    const auto normalized_angular_freq = 2 * PI * norm_f_demod;
    const auto stop_index = m_rf_line_num_samples - delay_compensation_num_samples;
    const auto num_samples = (stop_index - 1) / m_radial_decimation + 1;
    const auto num_blocks = round_up_div (num_samples, threads_per_line);
    m_log_object->write (ILog::DEBUG, "launch_DemodulateKernel...");
    launch_DemodulateKernel (num_blocks,
                             threads_per_line,
                             cur_stream,
                             rf_ptr,
                             normalized_angular_freq,
                             stop_index,
                             m_radial_decimation);

    if (m_store_kernel_details)
    {
      const auto elapsed_ms = static_cast<double> (event_timer->stop ());
      m_debug_data["kernel_demodulate_ms"].push_back (elapsed_ms);
      event_timer->restart ();
    }

    // Copy to host
    rf_lines.emplace_back (num_samples);
    auto dest = rf_lines.back ().data ();
    const auto dpitch = sizeof (std::complex<float>);
    const auto src = rf_ptr;
    const auto spitch = m_radial_decimation * sizeof (complex);
    const auto width = sizeof (complex);
    const auto height = num_samples;

    m_log_object->write (ILog::DEBUG, "cudaMemcpy2DAsync...");
    cudaErrorCheck (cudaMemcpy2DAsync (dest, dpitch, src, spitch, width, height, cudaMemcpyDeviceToHost, cur_stream));

    if (m_store_kernel_details)
    {
      const auto elapsed_ms = static_cast<double> (event_timer->stop ());
      m_debug_data["kernel_memcpy_ms"].push_back (elapsed_ms);
    }
  }

  m_log_object->write (ILog::DEBUG, "cudaDeviceSynchronize 5...");
  cudaErrorCheck (cudaDeviceSynchronize ());
}

void GpuAlgorithm::set_excitation (const ExcitationSignal &new_excitation)
{
  m_can_change_cuda_device = false;
  m_excitation_configured = true;
  m_excitation = new_excitation;

  init_rf_line_num_samples ();
  init_excitation_if_possible ();
  init_scan_sequence_if_possible ();
}

void GpuAlgorithm::set_scan_sequence (ScanSequence::s_ptr new_scan_sequence)
{
  if (!new_scan_sequence->is_valid ())
  {
    throw std::runtime_error ("Scan sequence is invalid");
  }

  m_can_change_cuda_device = false;
  m_scan_sequence_configured = true;
  m_scan_sequence = new_scan_sequence;

  // This order matters!
  init_rf_line_num_samples ();
  init_excitation_if_possible ();
  init_scan_sequence_if_possible ();
}

void GpuAlgorithm::init_excitation_if_possible ()
{
  if (!m_scan_sequence_configured || !m_excitation_configured)
  {
    return;
  }

  // Get bytes per line
  const auto device_iq_line_bytes = sizeof (complex) * m_rf_line_num_samples;

  // Setup pre-computed convolution kernel and Hilbert transformer.
  m_device_excitation_fft = DeviceBufferRAII<complex>::u_ptr (new DeviceBufferRAII<complex> (device_iq_line_bytes));
  m_log_object->write (ILog::INFO, "Number of excitation samples: " + std::to_string (m_excitation.samples.size ()));

  // Convert to complex with zero imaginary part.
  std::vector<std::complex<float>> temp (m_rf_line_num_samples);
  for (size_t i = 0; i < m_excitation.samples.size (); i++)
  {
    temp[i] = std::complex<float> (m_excitation.samples[i], 0.0f);
  }
  cudaErrorCheck (cudaMemcpy (m_device_excitation_fft->data (), temp.data (), device_iq_line_bytes, cudaMemcpyHostToDevice));

  // Create a plan for computing forward FFT of excitation
  auto excitation_fft_plan = CufftPlanRAII::u_ptr (new CufftPlanRAII (m_rf_line_num_samples, CUFFT_C2C, 1));

  // Compute FFT of excitation signal and add the Hilbert transform
  cufftErrorCheck (cufftExecC2C (excitation_fft_plan->get (), m_device_excitation_fft->data (), m_device_excitation_fft->data (), CUFFT_FORWARD));
  auto mask = discrete_hilbert_mask<std::complex<float>> (m_rf_line_num_samples);
  DeviceBufferRAII<complex> device_hilbert_mask (device_iq_line_bytes);
  cudaErrorCheck (cudaMemcpy (device_hilbert_mask.data (), mask.data (), device_iq_line_bytes, cudaMemcpyHostToDevice));

  // Result: u + i * H[u] where u is the excitation signal, i is sqrt(-1), H[.] is the Hilbert transform operator
  cudaStream_t cuda_stream = 0;
  // Note: This Hilbert transform multiplication mask doesn't require normalization. (See the `discrete_hilbert_mask` implementation to see why.)
  launch_MultiplyFftKernel<false> (m_rf_line_num_samples / m_param_threads_per_block, m_param_threads_per_block, cuda_stream, m_device_excitation_fft->data (), device_hilbert_mask.data (), m_rf_line_num_samples);
}

void GpuAlgorithm::init_scan_sequence_if_possible ()
{
  if (!m_scan_sequence_configured || !m_excitation_configured)
  {
    return;
  }

  // Get number of beams
  const auto num_beams = m_scan_sequence->get_num_lines ();
  m_log_object->write (ILog::INFO, "num_beams: " + std::to_string (num_beams));

  // Return early if samples allocated doesn't change.
  const auto new_num_samples_allocated = std::pair<unsigned int, int>{m_rf_line_num_samples, num_beams};
  if (new_num_samples_allocated == m_num_samples_allocated)
  {
    m_log_object->write (ILog::DEBUG, "Skipping init_scan_sequence_if_possible() because no new memory allocation needed.");
    return;
  }

  // Make CuFFT plan
  const auto num_samples = m_rf_line_num_samples;
  const auto batch = num_beams;
  const auto rank = 1;
  int dims[] = {static_cast<int> (m_rf_line_num_samples)};
  m_log_object->write (ILog::INFO, "Reconfiguring cuFFT batched plan");
  m_log_object->write (ILog::INFO, "batch = " + std::to_string (batch));
  m_fft_plan = CufftBatchedPlanRAII::u_ptr (new CufftBatchedPlanRAII (rank, dims, num_samples, CUFFT_C2C, batch));

  // Calculate bytes needed
  const auto device_iq_line_bytes = sizeof (complex) * m_rf_line_num_samples;
  std::stringstream ss;
  ss << std::fixed << std::setprecision (2);
  ss << "Allocating DEVICE memory ("
     << (device_iq_line_bytes * num_beams) * 1e-6
     << " MB)";
  m_log_object->write (ILog::INFO, ss.str ());

  // Allocate device memory for all RF lines
  m_device_time_proj = DeviceBufferRAII<complex>::u_ptr (new DeviceBufferRAII<complex> (device_iq_line_bytes * num_beams));

  // Save new number of samples allocated
  m_num_samples_allocated = new_num_samples_allocated;
}

void GpuAlgorithm::init_rf_line_num_samples ()
{
  if (!m_scan_sequence_configured || !m_excitation_configured)
  {
    return;
  }

  const auto line_length = m_scan_sequence->line_length;
  const auto sampling_frequency = m_excitation.sampling_frequency;

  // Calculate samples required for given depth (line_length)
  m_rf_line_num_samples = compute_num_rf_samples (m_param_sound_speed, line_length, sampling_frequency);

  // Add padding for convolution (with excitation signal)
  m_rf_line_num_samples += static_cast<unsigned int> (m_excitation.samples.size ()) - 1;

  // Round up to next power of two
  m_rf_line_num_samples = next_power_of_two (m_rf_line_num_samples);

  m_log_object->write (ILog::INFO, "m_rf_line_num_samples: " + std::to_string (m_rf_line_num_samples));
}

void GpuAlgorithm::throw_if_not_configured ()
{
  if (!m_scan_sequence_configured)
  {
    throw std::runtime_error ("Scan sequence not configured.");
  }
  if (!m_excitation_configured)
  {
    throw std::runtime_error ("Excitation not configured.");
  }
  if (m_cur_beam_profile_type == BeamProfileType::NOT_CONFIGURED)
  {
    throw std::runtime_error ("Beam profile not configured.");
  }
  if (get_total_num_scatterers () == 0)
  {
    throw std::runtime_error ("No scatterers are configured.");
  }
}

void GpuAlgorithm::set_analytical_profile (IBeamProfile::s_ptr beam_profile)
{
  m_log_object->write (ILog::INFO, "Setting analytical beam profile for GPU algorithm");
  const auto analytical_profile = std::dynamic_pointer_cast<GaussianBeamProfile> (beam_profile);
  if (!analytical_profile)
    throw std::runtime_error ("GpuAlgorithm: failed to cast beam profile");
  m_cur_beam_profile_type = BeamProfileType::ANALYTICAL;

  m_analytical_sigma_lat = analytical_profile->getSigmaLateral ();
  m_analytical_sigma_ele = analytical_profile->getSigmaElevational ();
}

void GpuAlgorithm::set_lookup_profile (IBeamProfile::s_ptr beam_profile)
{
  m_log_object->write (ILog::INFO, "Setting LUT profile for GPU algorithm");
  const auto lut_beam_profile = std::dynamic_pointer_cast<LUTBeamProfile> (beam_profile);
  if (!lut_beam_profile)
    throw std::runtime_error ("GpuAlgorithm: failed to cast beam profile");
  m_cur_beam_profile_type = BeamProfileType::LOOKUP;

  int num_samples_rad = lut_beam_profile->getNumSamplesRadial ();
  int num_samples_lat = lut_beam_profile->getNumSamplesLateral ();
  int num_samples_ele = lut_beam_profile->getNumSamplesElevational ();
  m_log_object->write (ILog::DEBUG, "=== set_lookup_profile () ===");
  m_log_object->write (ILog::DEBUG, "num_samples_rad: " + std::to_string (num_samples_rad));
  m_log_object->write (ILog::DEBUG, "num_samples_lat: " + std::to_string (num_samples_lat));
  m_log_object->write (ILog::DEBUG, "num_samples_ele: " + std::to_string (num_samples_ele));

  const auto r_range = lut_beam_profile->getRangeRange ();
  const auto l_range = lut_beam_profile->getLateralRange ();
  const auto e_range = lut_beam_profile->getElevationalRange ();

  // map to linear memory with correct 3D layout
  const auto total = num_samples_rad * num_samples_lat * num_samples_ele;
  std::vector<float> temp_samples;
  temp_samples.reserve (total);
  for (int zi = 0; zi < num_samples_rad; zi++)
  {
    for (int yi = 0; yi < num_samples_lat; yi++)
    {
      for (int xi = 0; xi < num_samples_ele; xi++)
      {
        const auto x = l_range.first + xi * (l_range.last - l_range.first) / (num_samples_lat - 1);
        const auto y = e_range.first + yi * (e_range.last - e_range.first) / (num_samples_ele - 1);
        const auto z = r_range.first + zi * (r_range.last - r_range.first) / (num_samples_rad - 1);
        temp_samples.push_back (lut_beam_profile->sampleProfile (z, x, y));
      }
    }
  }
  auto log_adapter = [&](const std::string &msg) {
    m_log_object->write (ILog::DEBUG, msg);
  };
  const auto table_extent = DeviceBeamProfileRAII::TableExtent3D (num_samples_lat, num_samples_ele, num_samples_rad);
  m_device_beam_profile = std::make_unique<DeviceBeamProfileRAII> (table_extent, temp_samples, log_adapter);
  // store spatial extent of profile.
  m_lut_r_min = r_range.first;
  m_lut_r_max = r_range.last;
  m_lut_l_min = l_range.first;
  m_lut_l_max = l_range.last;
  m_lut_e_min = e_range.first;
  m_lut_e_max = e_range.last;

  m_log_object->write (ILog::DEBUG, "Created a new DeviceBeamProfileRAII");

  if (false)
  {
    const std::string raw_lut_path ("d:/temp/raw_lookup_table/");
    dump_orthogonal_lut_slices (raw_lut_path);
    // write extents
    std::ofstream out_stream;
    out_stream.open (raw_lut_path + "/extents.txt");
    out_stream << m_lut_r_min << " " << m_lut_r_max << std::endl;
    out_stream << m_lut_l_min << " " << m_lut_l_max << std::endl;
    out_stream << m_lut_e_min << " " << m_lut_e_max << std::endl;
  }
}

void GpuAlgorithm::dump_orthogonal_lut_slices (const std::string &raw_path)
{
  const auto write_raw = [&](float3 origin, float3 dir0, float3 dir1, std::string raw_file) {
    const auto num_samples = m_param_threads_per_block;
    const auto total_num_samples = num_samples * num_samples;
    const auto num_bytes = sizeof (float) * total_num_samples;
    DeviceBufferRAII<float> device_slice (num_bytes);

    //dim3 grid_size (num_samples, num_samples, 1);
    //dim3 block_size (1, 1, 1);
    const cudaStream_t cuda_stream = 0;
    launch_SliceLookupTable (num_samples, num_samples, 1, cuda_stream,
                             origin, dir0, dir1, device_slice.data (), m_device_beam_profile->get ());
    cudaErrorCheck (cudaDeviceSynchronize ());
    dump_device_buffer_as_raw_file (device_slice, raw_file);
    m_log_object->write (ILog::DEBUG, "Wrote RAW file to " + raw_file);
  };

  // slice in the middle lateral-elevational plane (radial dist is 0.5)
  write_raw (make_float3 (0.0f, 0.0f, 0.5f),
             make_float3 (1.0f, 0.0f, 0.0f),
             make_float3 (0.0f, 1.0f, 0.0f),
             raw_path + "lut_slice_lat_ele.raw");
  // slice the middle lateral-radial plane (elevational dist is 0.5)
  write_raw (make_float3 (0.0f, 0.5f, 0.0f),
             make_float3 (1.0f, 0.0f, 0.0f),
             make_float3 (0.0f, 0.0f, 1.0f),
             raw_path + "lut_slice_lat_rad.raw");
  // slice the middle elevational-radial plane (lateral dist is 0.5)
  write_raw (make_float3 (0.5f, 0.0f, 0.0f),
             make_float3 (0.0f, 1.0f, 0.0f),
             make_float3 (0.0f, 0.0f, 1.0f),
             raw_path + "lut_slice_ele_rad.raw");

  // 6 equally spaced lateral-elevational slices of [0.0, 1.0]
  for (int i = 0; i <= 5; i++)
  {
    write_raw (make_float3 (0.0f, 0.0f, i / 5.0f),
               make_float3 (1.0f, 0.0f, 0.0f),
               make_float3 (0.0f, 1.0f, 0.0f),
               raw_path + "lut_slice_lat_ele_" + std::to_string (i) + ".raw");
  }
}

void GpuAlgorithm::create_dummy_lut_profile ()
{
  const size_t n = 16;
  std::vector<float> dummy_samples (n * n * n, 0.0f);
  m_device_beam_profile = DeviceBeamProfileRAII::u_ptr (new DeviceBeamProfileRAII (DeviceBeamProfileRAII::TableExtent3D (n, n, n), dummy_samples));
}

void GpuAlgorithm::clear_fixed_scatterers ()
{
  m_device_fixed_datasets.clear ();
}

void GpuAlgorithm::add_fixed_scatterers (FixedScatterers::s_ptr fixed_scatterers)
{
  m_device_fixed_datasets.add (fixed_scatterers);
  m_can_change_cuda_device = false;
}

void GpuAlgorithm::clear_spline_scatterers ()
{
  m_device_spline_datasets.clear ();
}

void GpuAlgorithm::add_spline_scatterers (SplineScatterers::s_ptr spline_scatterers)
{
  m_can_change_cuda_device = false;
  m_device_spline_datasets.add (spline_scatterers);
}

void GpuAlgorithm::fixed_projection_kernel (int stream_no, const Scanline &scanline, int num_blocks, cuComplex *res_buffer, DeviceFixedScatterers::s_ptr dataset)
{
  auto cur_stream = m_stream_wrappers[stream_no]->get ();

  //dim3 grid_size (num_blocks, 1, 1);
  //dim3 block_size (m_param_threads_per_block, 1, 1);

  // prepare struct with parameters
  FixedAlgKernelParams params;
  params.point_xs = dataset->get_xs_ptr ();
  params.point_ys = dataset->get_ys_ptr ();
  params.point_zs = dataset->get_zs_ptr ();
  params.point_as = dataset->get_as_ptr ();
  params.rad_dir = to_float3 (scanline.get_direction ());
  params.lat_dir = to_float3 (scanline.get_lateral_dir ());
  params.ele_dir = to_float3 (scanline.get_elevational_dir ());
  params.origin = to_float3 (scanline.get_origin ());
  params.fs_hertz = m_excitation.sampling_frequency;
  params.num_time_samples = m_rf_line_num_samples;
  params.sigma_lateral = m_analytical_sigma_lat;
  params.sigma_elevational = m_analytical_sigma_ele;
  params.sound_speed = m_param_sound_speed;
  params.result = res_buffer;
  params.demod_freq = m_excitation.demod_freq;
  params.num_scatterers = dataset->get_num_scatterers (),
  params.lut_tex = m_device_beam_profile->get ();
  params.lut.r_min = m_lut_r_min;
  params.lut.r_max = m_lut_r_max;
  params.lut.l_min = m_lut_l_min;
  params.lut.l_max = m_lut_l_max;
  params.lut.e_min = m_lut_e_min;
  params.lut.e_max = m_lut_e_max;

  // map beam profile type to boolean flag
  bool use_lut;
  switch (m_cur_beam_profile_type)
  {
  case BeamProfileType::ANALYTICAL:
    use_lut = false;
    break;
  case BeamProfileType::LOOKUP:
    use_lut = true;
    break;
  default:
    throw std::logic_error ("unknown beam profile type");
  }

  if (!m_param_use_arc_projection && !m_enable_phase_delay && !use_lut)
  {
    launch_FixedAlgKernel<false, false, false> (num_blocks, m_param_threads_per_block, cur_stream, params);
  }
  else if (!m_param_use_arc_projection && !m_enable_phase_delay && use_lut)
  {
    launch_FixedAlgKernel<false, false, true> (num_blocks, m_param_threads_per_block, cur_stream, params);
  }
  else if (!m_param_use_arc_projection && m_enable_phase_delay && !use_lut)
  {
    launch_FixedAlgKernel<false, true, false> (num_blocks, m_param_threads_per_block, cur_stream, params);
  }
  else if (!m_param_use_arc_projection && m_enable_phase_delay && use_lut)
  {
    launch_FixedAlgKernel<false, true, true> (num_blocks, m_param_threads_per_block, cur_stream, params);
  }
  else if (m_param_use_arc_projection && !m_enable_phase_delay && !use_lut)
  {
    launch_FixedAlgKernel<true, false, false> (num_blocks, m_param_threads_per_block, cur_stream, params);
  }
  else if (m_param_use_arc_projection && !m_enable_phase_delay && use_lut)
  {
    launch_FixedAlgKernel<true, false, true> (num_blocks, m_param_threads_per_block, cur_stream, params);
  }
  else if (m_param_use_arc_projection && m_enable_phase_delay && !use_lut)
  {
    launch_FixedAlgKernel<true, true, false> (num_blocks, m_param_threads_per_block, cur_stream, params);
  }
  else if (m_param_use_arc_projection && m_enable_phase_delay && use_lut)
  {
    launch_FixedAlgKernel<true, true, true> (num_blocks, m_param_threads_per_block, cur_stream, params);
  }
  else
  {
    throw std::logic_error ("this should never happen");
  }
}

void GpuAlgorithm::spline_projection_kernel (int stream_no, const Scanline &scanline, int num_blocks, cuComplex *res_buffer, DeviceSplineScatterers::s_ptr dataset)
{
  auto cur_stream = m_stream_wrappers[stream_no]->get ();
  const auto cur_knots = dataset->get_knots ();
  const auto num_cs = dataset->get_num_cs ();
  const auto spline_degree = dataset->get_spline_degree ();

  // evaluate the basis functions and upload to constant memory.
  const auto num_nonzero = spline_degree + 1;
  size_t eval_basis_offset_elements = num_nonzero * stream_no;
  std::vector<float> host_basis_functions (num_cs);
  for (int i = 0; i < num_cs; i++)
  {
    host_basis_functions[i] = bspline_storve::bsplineBasis (i, spline_degree, scanline.get_timestamp (), cur_knots);
  }

  //dim3 grid_size (num_blocks, 1, 1);
  //dim3 block_size (m_param_threads_per_block, 1, 1);

  // compute sum limits (inclusive)
  int cs_idx_start, cs_idx_end;
  std::tie (cs_idx_start, cs_idx_end) = bspline_storve::get_lower_upper_inds (cur_knots,
                                                                              scanline.get_timestamp (),
                                                                              spline_degree);
  if (!sanity_check_spline_lower_upper_bound (host_basis_functions, cs_idx_start, cs_idx_end))
  {
    throw std::runtime_error ("b-spline basis bounds failed sanity check");
  }
  if (cs_idx_end - cs_idx_start + 1 != num_nonzero)
    throw std::logic_error ("illegal number of non-zero basis functions");

  if (!splineAlg2_updateConstantMemory (host_basis_functions.data () + cs_idx_start,
                                        num_nonzero * sizeof (float),
                                        eval_basis_offset_elements * sizeof (float),
                                        cudaMemcpyHostToDevice,
                                        cur_stream))

  {
    throw std::runtime_error ("Failed to copy to symbol memory");
  }

  // prepare a struct of arguments
  SplineAlgKernelParams params;
  params.control_xs = dataset->get_xs_ptr ();
  params.control_ys = dataset->get_ys_ptr ();
  params.control_zs = dataset->get_zs_ptr ();
  params.control_as = dataset->get_as_ptr ();
  params.rad_dir = to_float3 (scanline.get_direction ());
  params.lat_dir = to_float3 (scanline.get_lateral_dir ());
  params.ele_dir = to_float3 (scanline.get_elevational_dir ());
  params.origin = to_float3 (scanline.get_origin ());
  params.fs_hertz = m_excitation.sampling_frequency;
  params.num_time_samples = m_rf_line_num_samples;
  params.sigma_lateral = m_analytical_sigma_lat;
  params.sigma_elevational = m_analytical_sigma_ele;
  params.sound_speed = m_param_sound_speed;
  params.cs_idx_start = cs_idx_start;
  params.cs_idx_end = cs_idx_end;
  params.NUM_SPLINES = dataset->get_num_scatterers (),
  params.result = res_buffer;
  params.eval_basis_offset_elements = eval_basis_offset_elements;
  params.demod_freq = m_excitation.demod_freq;
  params.lut_tex = m_device_beam_profile->get ();
  params.lut.r_min = m_lut_r_min;
  params.lut.r_max = m_lut_r_max;
  params.lut.l_min = m_lut_l_min;
  params.lut.l_max = m_lut_l_max;
  params.lut.e_min = m_lut_e_min;
  params.lut.e_max = m_lut_e_max;

  // map lut type to a boolean flag
  bool use_lut;
  switch (m_cur_beam_profile_type)
  {
  case BeamProfileType::ANALYTICAL:
    use_lut = false;
    break;
  case BeamProfileType::LOOKUP:
    use_lut = true;
    break;
  default:
    throw std::logic_error ("spline_projection_kernel (): unknown beam profile type");
  }
  if (!m_param_use_arc_projection && !m_enable_phase_delay && !use_lut)
  {
    launch_SplineAlgKernel<false, false, false> (num_blocks, m_param_threads_per_block, cur_stream, params);
  }
  else if (!m_param_use_arc_projection && !m_enable_phase_delay && use_lut)
  {
    launch_SplineAlgKernel<false, false, true> (num_blocks, m_param_threads_per_block, cur_stream, params);
  }
  else if (!m_param_use_arc_projection && m_enable_phase_delay && !use_lut)
  {
    launch_SplineAlgKernel<false, true, false> (num_blocks, m_param_threads_per_block, cur_stream, params);
  }
  else if (!m_param_use_arc_projection && m_enable_phase_delay && use_lut)
  {
    launch_SplineAlgKernel<false, true, true> (num_blocks, m_param_threads_per_block, cur_stream, params);
  }
  else if (m_param_use_arc_projection && !m_enable_phase_delay && !use_lut)
  {
    launch_SplineAlgKernel<true, false, false> (num_blocks, m_param_threads_per_block, cur_stream, params);
  }
  else if (m_param_use_arc_projection && !m_enable_phase_delay && use_lut)
  {
    launch_SplineAlgKernel<true, false, true> (num_blocks, m_param_threads_per_block, cur_stream, params);
  }
  else if (m_param_use_arc_projection && m_enable_phase_delay && !use_lut)
  {
    launch_SplineAlgKernel<true, true, false> (num_blocks, m_param_threads_per_block, cur_stream, params);
  }
  else if (m_param_use_arc_projection && m_enable_phase_delay && use_lut)
  {
    launch_SplineAlgKernel<true, true, true> (num_blocks, m_param_threads_per_block, cur_stream, params);
  }
  else
  {
    throw std::logic_error ("this should never happen");
  }
}

size_t GpuAlgorithm::get_total_num_scatterers () const
{
  const auto total_num_fixed = m_device_fixed_datasets.get_total_num_scatterers ();
  const auto total_num_spline = m_device_spline_datasets.get_total_num_scatterers ();
  return total_num_fixed + total_num_spline;
}

std::string GpuAlgorithm::get_parameter (const std::string &key) const
{
  if (key == "num_cuda_devices")
  {
    int num_devices;
    cudaErrorCheck (cudaGetDeviceCount (&num_devices));
    return std::to_string (num_devices);
  }
  else if (key == "cur_device_name")
  {
    cudaDeviceProp prop;
    cudaErrorCheck (cudaGetDeviceProperties (&prop, m_param_cuda_device_no));
    return prop.name;
  }
  else if (key == "use_elev_hack")
  {
    return std::to_string (m_use_elev_hack);
  }
  else if (key == "use_delay_compensation")
  {
    return std::to_string (m_use_delay_compensation);
  }
  else
  {
    return BaseAlgorithm::get_parameter (key);
  }
}

} // namespace bcsim

#endif // BCSIM_ENABLE_CUDA

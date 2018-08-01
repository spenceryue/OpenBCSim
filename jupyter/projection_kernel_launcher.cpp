#include "definitions.hpp"
#include <torch/torch.h>
#include <vector>

// CUDA forward declarations

std::vector<at::Tensor> launch_projection_kernel (
    at::Tensor input,
    at::Tensor weights,
    at::Tensor bias,
    at::Tensor old_h,
    at::Tensor old_cell);

// C++ interface

#define CHECK_CUDA(x) AT_ASSERT (x.type ().is_cuda (), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERT (x.is_contiguous (), #x " must be contiguous")
#define CHECK_INPUT(x) \
  CHECK_CUDA (x);      \
  CHECK_CONTIGUOUS (x)

std::vector<at::Tensor> run (
    at::Tensor input,
    at::Tensor weights,
    at::Tensor bias,
    at::Tensor old_h,
    at::Tensor old_cell)
{
  CHECK_INPUT (input);
  CHECK_INPUT (weights);
  CHECK_INPUT (bias);
  CHECK_INPUT (old_h);
  CHECK_INPUT (old_cell);

  return launch_projection_kernel (input, weights, bias, old_h, old_cell);
}

PYBIND11_MODULE (TORCH_EXTENSION_NAME, m)
{
  m.def ("run", &run, "Projection run (CUDA)");
}

std::vector<at::Tensor> launch_projection_kernel (
    at::Tensor tensor_type,
    at::Tensor input,
    at::Tensor weights,
    at::Tensor bias,
    at::Tensor old_h,
    at::Tensor old_cell)
{
  auto X = at::cat ({old_h, input}, /*dim=*/1);
  auto gates = at::addmm (bias, X, weights.transpose (0, 1));

  const auto batch_size = old_cell.size (0);
  const auto state_size = old_cell.size (1);

  auto new_h = at::zeros_like (old_cell);
  auto new_cell = at::zeros_like (old_cell);
  auto input_gate = at::zeros_like (old_cell);
  auto output_gate = at::zeros_like (old_cell);
  auto candidate_cell = at::zeros_like (old_cell);

  const int threads = 1024;
  const dim3 blocks ((state_size + threads - 1) / threads, batch_size);

  AT_DISPATCH_FLOATING_TYPES (tensor_type.type (),
                              "projection_kernel",
                              ([&] {
                                Simulation<scalar_t> args;
                                args.transmitter.x = transmitter_x.data<scalar_t> ();
                                args.transmitter.y = transmitter_y.data<scalar_t> ();
                                args.transmitter.z = transmitter_z.data<scalar_t> ();
                                args.transmitter.delay = transmitter_delay.data<scalar_t> ();
                                args.transmitter.apodization = transmitter_apodization.data<scalar_t> ();
                                args.receiver.x = receiver_x.data<scalar_t> ();
                                args.receiver.y = receiver_y.data<scalar_t> ();
                                args.receiver.z = receiver_z.data<scalar_t> ();
                                args.receiver.delay = receiver_delay.data<scalar_t> ();
                                args.receiver.apodization = receiver_apodization.data<scalar_t> ();
                                args.scatterer_x = scatterer_x.data<scalar_t> ();
                                args.scatterer_y = scatterer_y.data<scalar_t> ();
                                args.scatterer_z = scatterer_z.data<scalar_t> ();
                                args.scatterer_amplitude = scatterer_amplitude.data<scalar_t> ();
                                projection_kernel<scalar_t><<<blocks, threads>>> (args, output);
                              }));

  return {new_h, new_cell, input_gate, output_gate, candidate_cell, X, gates};
}

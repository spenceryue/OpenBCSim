#pragma once

/* https://github.com/pybind/pybind11/issues/1212#issuecomment-365555709 */
#define strdup _strdup
#include <torch/torch.h> // Must go first

#include "utils.h"
#include <cuda_runtime_api.h>

namespace py = pybind11;
using namespace pybind11::literals;

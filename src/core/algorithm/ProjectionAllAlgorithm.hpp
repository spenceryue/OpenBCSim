#ifdef BCSIM_ENABLE_CUDA
#pragma once
#include "BaseAlgorithm.hpp"
#include "cuda_helpers.h"
#include "cuda_kernels_c_interface.h"

namespace bcsim
{
class ProjectionAllAlgorithm : public IAlgorithm
{
public:
};
} // namespace bcsim

#endif // BCSIM_ENABLE_CUDA

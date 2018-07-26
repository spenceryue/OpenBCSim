#pragma once

// Returns false on failure.
bool splineAlg1_updateConstantMemory_internal(float* src_ptr, size_t num_bytes);

__global__ void RenderSplineKernel(const float* control_xs,
                                   const float* control_ys,
                                   const float* control_zs,
                                   float* rendered_xs,
                                   float* rendered_ys,
                                   float* rendered_zs,
                                   size_t cs_idx_start,
                                   size_t cs_idx_end,
                                   size_t NUM_SPLINES);

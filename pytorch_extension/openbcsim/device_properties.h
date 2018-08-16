#pragma once
#include "base.h" // Must go first

/* https://docs.nvidia.com/cuda/archive/9.0/cuda-runtime-api/structcudaDeviceProp.html */

// clang-format off
#define FORALL_DEVICE_PROPERTIES(_) \
/*  1 */ _(ECCEnabled, "Device has ECC support enabled") \
/*  2 */ _(asyncEngineCount, "Number of asynchronous engines") \
/*  3 */ _(canMapHostMemory, "Device can map host memory with cudaHostAlloc/cudaHostGetDevicePointer") \
/*  4 */ _(canUseHostPointerForRegisteredMem, "Device can access host registered memory at the same virtual address as the CPU") \
/*  5 */ _(clockRate, "Clock frequency in kilohertz") \
/*  6 */ _(computeMode, "Compute mode (See cudaComputeMode)") \
/*  7 */ _(computePreemptionSupported, "Device supports Compute Preemption") \
/*  8 */ _(concurrentKernels, "Device can possibly execute multiple kernels concurrently") \
/*  9 */ _(concurrentManagedAccess, "Device can coherently access managed memory concurrently with the CPU") \
/* 10 */ _(cooperativeLaunch, "Device supports launching cooperative kernels via cudaLaunchCooperativeKernel") \
/* 11 */ _(cooperativeMultiDeviceLaunch, "Device can participate in cooperative kernels launched via cudaLaunchCooperativeKernelMultiDevice") \
/* 12 */ _(deviceOverlap, "Device can concurrently copy memory and execute a kernel. Deprecated. Use instead asyncEngineCount.") \
/* 13 */ _(globalL1CacheSupported, "Device supports caching globals in L1") \
/* 14 */ _(hostNativeAtomicSupported, "Link between the device and the host supports native atomic operations") \
/* 15 */ _(integrated, "Device is integrated as opposed to discrete") \
/* 16 */ _(isMultiGpuBoard, "Device is on a multi-GPU board") \
/* 17 */ _(kernelExecTimeoutEnabled, "Specified whether there is a run time limit on kernels") \
/* 18 */ _(l2CacheSize, "Size of L2 cache in bytes") \
/* 19 */ _(localL1CacheSupported, "Device supports caching locals in L1") \
/* 20 */ _(major, "Major compute capability") \
/* 21 */ _(managedMemory, "Device supports allocating managed memory on this system") \
/* 22 */ _(maxGridSize, "Maximum size of each dimension of a grid") \
/* 23 */ _(maxSurface1D, "Maximum 1D surface size") \
/* 24 */ _(maxSurface1DLayered, "Maximum 1D layered surface dimensions") \
/* 25 */ _(maxSurface2D, "Maximum 2D surface dimensions") \
/* 26 */ _(maxSurface2DLayered, "Maximum 2D layered surface dimensions") \
/* 27 */ _(maxSurface3D, "Maximum 3D surface dimensions") \
/* 28 */ _(maxSurfaceCubemap, "Maximum Cubemap surface dimensions") \
/* 29 */ _(maxSurfaceCubemapLayered, "Maximum Cubemap layered surface dimensions") \
/* 30 */ _(maxTexture1D, "Maximum 1D texture size") \
/* 31 */ _(maxTexture1DLayered, "Maximum 1D layered texture dimensions") \
/* 32 */ _(maxTexture1DLinear, "Maximum size for 1D textures bound to linear memory") \
/* 33 */ _(maxTexture1DMipmap, "Maximum 1D mipmapped texture size") \
/* 34 */ _(maxTexture2D, "Maximum 2D texture dimensions") \
/* 35 */ _(maxTexture2DGather, "Maximum 2D texture dimensions if texture gather operations have to be performed") \
/* 36 */ _(maxTexture2DLayered, "Maximum 2D layered texture dimensions") \
/* 37 */ _(maxTexture2DLinear, "Maximum dimensions (width, height, pitch) for 2D textures bound to pitched memory") \
/* 38 */ _(maxTexture2DMipmap, "Maximum 2D mipmapped texture dimensions") \
/* 39 */ _(maxTexture3D, "Maximum 3D texture dimensions") \
/* 40 */ _(maxTexture3DAlt, "Maximum alternate 3D texture dimensions") \
/* 41 */ _(maxTextureCubemap, "Maximum Cubemap texture dimensions") \
/* 42 */ _(maxTextureCubemapLayered, "Maximum Cubemap layered texture dimensions") \
/* 43 */ _(maxThreadsDim, "Maximum size of each dimension of a block") \
/* 44 */ _(maxThreadsPerBlock, "Maximum number of threads per block") \
/* 45 */ _(maxThreadsPerMultiProcessor, "Maximum resident threads per multiprocessor") \
/* 46 */ _(memPitch, "Maximum pitch in bytes allowed by memory copies") \
/* 47 */ _(memoryBusWidth, "Global memory bus width in bits") \
/* 48 */ _(memoryClockRate, "Peak memory clock frequency in kilohertz") \
/* 49 */ _(minor, "Minor compute capability") \
/* 50 */ _(multiGpuBoardGroupID, "Unique identifier for a group of devices on the same multi-GPU board") \
/* 51 */ _(multiProcessorCount, "Number of multiprocessors on device") \
/* 52 */ _(name, "ASCII string identifying device") \
/* 53 */ _(pageableMemoryAccess, "Device supports coherently accessing pageable memory without calling cudaHostRegister on it") \
/* 54 */ _(pciBusID, "PCI bus ID of the device") \
/* 55 */ _(pciDeviceID, "PCI device ID of the device") \
/* 56 */ _(pciDomainID, "PCI domain ID of the device") \
/* 57 */ _(regsPerBlock, "32-bit registers available per block") \
/* 58 */ _(regsPerMultiprocessor, "32-bit registers available per multiprocessor") \
/* 59 */ _(sharedMemPerBlock, "Shared memory available per block in bytes") \
/* 60 */ _(sharedMemPerBlockOptin, "Per device maximum shared memory per block usable by special opt in") \
/* 61 */ _(sharedMemPerMultiprocessor, "Shared memory available per multiprocessor in bytes") \
/* 62 */ _(singleToDoublePrecisionPerfRatio, "Ratio of single precision performance (in floating-point operations per second) to double precision performance") \
/* 63 */ _(streamPrioritiesSupported, "Device supports stream priorities") \
/* 64 */ _(surfaceAlignment, "Alignment requirements for surfaces") \
/* 65 */ _(tccDriver, "1 if device is a Tesla device using TCC driver, 0 otherwise") \
/* 66 */ _(textureAlignment, "Alignment requirement for textures") \
/* 67 */ _(texturePitchAlignment, "Pitch alignment requirement for texture references bound to pitched memory") \
/* 68 */ _(totalConstMem, "Constant memory available on device in bytes") \
/* 69 */ _(totalGlobalMem, "Global memory available on device in bytes") \
/* 70 */ _(unifiedAddressing, "Device shares a unified address space with the host") \
/* 71 */ _(warpSize, "Warp size in threads")
// clang-format on

struct DeviceProperties : cudaDeviceProp
{
  // No new members.
  // Subclassing to avoid conflict with `cudaDeviceProp` already being registered by PyTorch.
  // (PyTorch did not include all the members of `cudaDeviceProp` in their Python binding.)
  DeviceProperties () = default;
  DeviceProperties (const cudaDeviceProp &properties) : cudaDeviceProp (properties) {}
};

DeviceProperties &get_properties (int device = 0);
void bind_DeviceProperties (py::module m);

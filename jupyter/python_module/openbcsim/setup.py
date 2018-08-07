from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup (
  name = 'openbcsim',
  ext_modules = [
    CUDAExtension ('openbcsim', [
        'openbcsim_module.cpp',
        'openbcsim_kernel.cu',
      ],
      extra_compile_args = {
        'cxx': [
          '/Ox',
          '/std:c++latest'
        ],
        'nvcc': [
          '-arch=sm_61',
          '--nvlink-options=-v'
          '-O0', '-Xptxas', '-O0'
        ]
      }
    ),
  ],
  cmdclass = {
  'build_ext': BuildExtension
  },
  test_suite='nose.collector',
  tests_require=['nose'],
)

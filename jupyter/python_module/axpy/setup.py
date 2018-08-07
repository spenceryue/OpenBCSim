from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension

# For `--compiler=unix`
# import os
# # os.environ['CC'] = 'gcc'
# os.environ['CC'] = 'clang++'
# os.environ['CPP'] = 'cpp'
# os.environ['CXX'] = 'clang++'
# os.environ['LDSHARED'] = 'clang++ -shared -stdlib=libc++ -nostdinc++ -IC:/LLVM/include/c++/v1 -LC:/LLVM/lib -fuse-ld=lld -flto'
# os.environ['LDFLAGS'] = ''
# os.environ['CFLAGS'] = ''
# os.environ['CPPFLAGS'] = ''
# os.environ['AR'] = 'ar'
# os.environ['ARFLAGS'] = ''
# os.environ['OPT'] = ''
# os.environ['SHLIB_SUFFIX'] = 'dll'
# os.environ['CCSHARED'] = ''
# from distutils import sysconfig
# sysconfig.get_config_vars ().update (os.environ)

setup (
  name = 'axpy',
  ext_modules = [
  CUDAExtension ('axpy', [
      'axpy_cuda.cpp',
      'axpy_cuda_kernel.cu',
    ],
    extra_compile_args = {
      'cxx': [],
      # For `--compiler=unix`
      # 'cxx': ['-std=c++17'],
      'nvcc': ['-arch=sm_61']
    },
    # For `--compiler=unix`
    # define_macros = [
    #   ('_hypot', 'hypot')
    # ],
    )
  # For testing default argument of `None` (doesn't work):
  # CppExtension ('none_arg', ['none_arg.cpp'], extra_compile_args=['/std:c++latest'])

  ],
  cmdclass = {
  'build_ext': BuildExtension
  },
  test_suite='nose.collector',
  tests_require=['nose']
)

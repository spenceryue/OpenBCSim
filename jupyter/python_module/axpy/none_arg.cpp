/* https://github.com/pybind/pybind11/issues/1212#issuecomment-365555709 */
#define strdup _strdup

#include <torch/torch.h>

at::Tensor sigmoid_add (at::Tensor x, at::Tensor y)
{
  return x.sigmoid () + y.sigmoid ();
}

struct MatrixMultiplier
{
  MatrixMultiplier (int A, int B)
  {
    tensor_ = at::ones (torch::CPU (at::kDouble), {A, B});
    torch::set_requires_grad (tensor_, true);
  }
  at::Tensor forward (at::Tensor weights)
  {
    return tensor_.mm (weights);
  }
  at::Tensor get () const
  {
    return tensor_;
  }

private:
  at::Tensor tensor_;
};

bool function_taking_optional (at::optional<at::Tensor> tensor)
{
  return tensor.has_value ();
}

PYBIND11_MODULE (TORCH_EXTENSION_NAME, m)
{
  m.def ("sigmoid_add", &sigmoid_add, "sigmoid(x) + sigmoid(y)");
  m.def (
      "function_taking_optional",
      &function_taking_optional,
      "function_taking_optional");
  py::class_<MatrixMultiplier> (m, "MatrixMultiplier")
      .def (py::init<int, int> ())
      .def ("forward", &MatrixMultiplier::forward)
      .def ("get", &MatrixMultiplier::get);
}

/*
Also tried:
clang++ ^
-shared ^
-I C:\Users\spenc\Anaconda3\lib\site-packages\torch\lib\include ^
-I C:\Users\spenc\Anaconda3\lib\site-packages\torch\lib\include\TH ^
-I C:\Users\spenc\Anaconda3\lib\site-packages\torch\lib\include\THC ^
-I "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\include" ^
-I C:\Users\spenc\Anaconda3\include ^
-L C:\Users\spenc\Anaconda3\lib\site-packages\torch\lib ^
-L "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\lib\x64" ^
-L C:\Users\spenc\Anaconda3\libs ^
-L "C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\LIB\amd64" ^
-L "C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\ATLMFC\LIB\amd64" ^
-L "C:\Program Files (x86)\Windows Kits\10\lib\10.0.14393.0\ucrt\x64" ^
-L "C:\Program Files (x86)\Windows Kits\NETFXSDK\4.6\lib\um\x64" ^
-L "C:\Program Files (x86)\Windows Kits\10\lib\10.0.14393.0\um\x64" ^
-nostdinc++ ^
-std=c++17 ^
-l msvcrt ^
-l cudart.lib ^
-l ATen.lib ^
-l _C.lib ^
-fuse-ld=lld ^
-DTORCH_EXTENSION_NAME=none_arg ^
-o ^
build\lib.win-amd64-3.6\none_arg.cp36-win_amd64.pyd ^
none_arg.cpp

clang++ ^
-shared ^
-L C:\Users\spenc\Anaconda3\lib\site-packages\torch\lib ^
-L "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\lib\x64" ^
-L C:\Users\spenc\Anaconda3\libs ^
-L "C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\LIB\amd64" ^
-L "C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\ATLMFC\LIB\amd64" ^
-L "C:\Program Files (x86)\Windows Kits\10\lib\10.0.14393.0\ucrt\x64" ^
-L "C:\Program Files (x86)\Windows Kits\NETFXSDK\4.6\lib\um\x64" ^
-L "C:\Program Files (x86)\Windows Kits\10\lib\10.0.14393.0\um\x64" ^
-l msvcrt ^
-l cudart.lib ^
-l ATen.lib ^
-l _C.lib ^
-fuse-ld=lld ^
-o ^
build\lib.win-amd64-3.6\axpy.cp36-win_amd64.pyd ^
build\temp.win-amd64-3.6\Release\axpy_cuda.obj ^
build\temp.win-amd64-3.6\Release\axpy_cuda_kernel.obj -v


CONCLUSION:
  somehow at::optional doesn't work on Windows.


FILE TAKEN FROM:
  pytorch\test\cpp_extensions\extension.cpp
  pytorch\test\test_cpp_extensions.py


DOESN'T WORK:
  build setup.py -c msvc


DOESN'T WORK:
        -fvisibility=hidden ^
clang++ -Wall -std=c++14 ^
        -stdlib=libc++ -nostdinc++ -IC:\LLVM\include\c++\v1 -LC:\LLVM\lib ^
        -fuse-ld=lld ^
        -flto ^
        -Xclang -flto-visibility-public-std ^
        -I C:\Users\spenc\Anaconda3\pkgs\pytorch-0.4.0-py36_cuda80_cudnn7he774522_1\Lib\site-packages\torch\lib\include ^
        -I C:\Users\spenc\Anaconda3\pkgs\pytorch-0.4.0-py36_cuda80_cudnn7he774522_1\Lib\site-packages\torch\lib\include\TH ^
        -I C:\Users\spenc\Anaconda3\pkgs\pytorch-0.4.0-py36_cuda80_cudnn7he774522_1\Lib\site-packages\torch\lib\include\THC ^
        -I C:\Users\spenc\Anaconda3\pkgs\pytorch-0.4.0-py36_cuda80_cudnn7he774522_1\Lib\site-packages\torch\lib\include\THCUNN ^
        -I C:\Users\spenc\Anaconda3\pkgs\pytorch-0.4.0-py36_cuda80_cudnn7he774522_1\Lib\site-packages\torch\lib\include\pybind11 ^
        -I C:\Users\spenc\Anaconda3\pkgs\pytorch-0.4.0-py36_cuda80_cudnn7he774522_1\Lib\site-packages\torch\lib\include\ATen ^
        -I C:\Users\spenc\Anaconda3\Include ^
        -L C:\Users\spenc\Anaconda3\libs ^
        -L C:\Users\spenc\Anaconda3\pkgs\pytorch-0.4.0-py36_cuda80_cudnn7he774522_1\Lib\site-packages\torch\lib ^
        -x c++ ^
        none_arg.cpp -o none_arg.pyd ^
        -v ^
        > output.txt 2>&1 ^
        & type output.txt
*/

/* https://github.com/pybind/pybind11/issues/1212#issuecomment-365555709 */
#define strdup _strdup
#include <torch/torch.h>
namespace py = pybind11;
using namespace pybind11::literals;

template <class scalar_t>
void run_it (scalar_t a, scalar_t *x, scalar_t *y, size_t N);

#define CHECK_CUDA(x) AT_ASSERT (x.type ().is_cuda (), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERT (x.is_contiguous (), #x " must be contiguous")
#define CHECK_INPUT(x)   \
  do                     \
  {                      \
    CHECK_CUDA (x);      \
    CHECK_CONTIGUOUS (x) \
  } while (false)

void run_it_ya (float a, at::Tensor x, at::Tensor y)
{
  CHECK_INPUT (x);
  CHECK_INPUT (y);
  size_t N = y.numel ();
  std::cout << "N: " << N << std::endl;

  AT_DISPATCH_FLOATING_TYPES (x.type (), "axpy.run",
                              [&] {
                                run_it<scalar_t> (a, x.data<scalar_t> (), y.data<scalar_t> (), N);
                              });
}

struct TestClass
{
  static TestClass create (float a, py::object x, py::object y)
  {
    TestClass result;
    result.a = a;
    if (!x.is_none ())
    {
      result.x = py::cast<at::Tensor> (x).data<float> ();
      for (auto each : x)
        std::cout << each << std::endl;
    }
    else
      std::cout << "BLEH" << std::endl;
    if (!y.is_none ())
    {
      result.y = py::cast<at::Tensor> (y).data<float> ();
      for (auto each : y)
        std::cout << each << std::endl;
    }
    else
      std::cout << "BLEH BLEH" << std::endl;
    return result;
  }
  void set_a (float a_)
  {
    this->a = a_;
  }
  void set_x (at::Tensor x_)
  {
    this->x = x_.data<float> ();
  }
  void set_y (at::Tensor y_)
  {
    this->y = y_.data<float> ();
  }
  float a = 2 * 3.14159265;
  float *x;
  float *y;
};

#include <iostream>
#include <memory>

int hello (TestClass *a)
// int hello (py::object a)
{
  std::unique_ptr<TestClass> b;
  // if (a.is_none ())
  if (py::cast (a).is_none ())
  {
    std::cout << "Ah sad, you gave me None" << std::endl;
    b = std::make_unique<TestClass> ();
  }
  else
  {
    // b = py::cast<TestClass *> (a);
    b.reset (a);
  }
  return b->a;
}

// Useful Cheatsheet:
// https://github.com/pybind/pybind11/issues/1201

PYBIND11_MODULE (TORCH_EXTENSION_NAME, m)
{
  m.def ("run", &run_it_ya, "lalala");

  py::class_<TestClass> (m, "TestClass")
      .def (py::init (&TestClass::create), "TestClass.__init__ docstring",
            "a"_a = 1.2,
            "x"_a = nullptr,
            "y"_a = nullptr)
      .def_property ("a", nullptr, &TestClass::set_a)
      .def_property ("x", nullptr, &TestClass::set_x)
      .def_property ("y", nullptr, &TestClass::set_y);

  m.def ("hello", &hello, "a"_a = nullptr);
}

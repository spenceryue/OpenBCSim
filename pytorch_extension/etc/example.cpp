/* https://github.com/pybind/pybind11/issues/1212#issuecomment-365555709 */
#define strdup _strdup

#include <pybind11/pybind11.h>

int add (int i, int j)
{
  return i + j;
}

PYBIND11_MODULE (example, m)
{
  m.doc () = "pybind11 example plugin"; // optional module docstring

  m.def ("add", &add, "A function which adds two numbers");
}

/*
clang++ -O3 -Wall -shared -std=c++17 ^
        -fuse-ld=lld ^
        -flto ^
        -fvisibility=hidden ^
        -Xclang -flto-visibility-public-std ^
        -isystem C:\Users\spenc\Anaconda3\Include ^
        -LC:\Users\spenc\Anaconda3\libs ^
        example.cpp -o example.pyd ^
        -v ^
        > output.txt 2>&1 ^
        & type output.txt
*/

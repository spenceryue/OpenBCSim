#pragma once
// Requires: base.h
#include <cuda_runtime_api.h>
#include <exception>
#include <iostream>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

#define CHECK_CUDA(x) AT_ASSERT ((x).type ().is_cuda (), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERT ((x).is_contiguous (), #x " must be contiguous")
#define CHECK_INPUT(x)   \
  do                     \
  {                      \
    CHECK_CUDA (x);      \
    CHECK_CONTIGUOUS (x) \
  } while (0)

// Select an overloaded function more easily when binding with pybind11.
// https://stackoverflow.com/a/41301717/3624264
template <class ReturnType, class... ArgTypes>
inline auto with_signature (ReturnType (*function) (ArgTypes...))
{
  return function;
}

// Convert class member to a compatible pybind11 binding type.
// Note: `std::is_base_of_v` not supported... even with clang 6.0 on Windows. (Bug?) Must use `is_base_of<>::value`.
template <class Self, class Member, class Base, std::enable_if_t<std::is_base_of<Base, Self>::value, int> = 0>
inline auto sanitize_member (const Self &self, Member Base::*member)
{
  using namespace std;
  if constexpr (is_same_v<remove_all_extents_t<Member>, char>)
  {
    return string (self.*member);
  }
  else if constexpr (is_array_v<Member>)
  {
    auto length = extent_v<Member>;
    using Element = remove_all_extents_t<Member>;
    return vector<Element> (self.*member, self.*member + length);
  }
  else
  {
    return self.*member;
  }
}

#define checkCall(CALL) \
  checkCall_ (#CALL, CALL, __FILE__, __LINE__)

#define warnCall(CALL) \
  checkCall_<true> (#CALL, CALL, __FILE__, __LINE__)

template <bool warn_only = false>
inline cudaError checkCall_ (const std::string &call, cudaError code, const std::string &file, int line)
{
  using namespace std;
  if (code != cudaSuccess)
  {
    stringstream s;
    s << "At " << file << ":" << line << "..." << endl;
    s << "  " << call + " failed with error code " << code << ":" << endl;
    s << "  " << cudaGetErrorString (code) << endl;
    if (!warn_only)
    {
      throw runtime_error (s.str ());
    }
  }
  return code;
}

template <class T>
inline T ceil_div (T num, T den)
{
  return static_cast<T> ((num + den - 1) / den);
}

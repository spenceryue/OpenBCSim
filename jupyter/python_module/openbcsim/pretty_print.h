#pragma once

#include <chrono>
#include <iomanip>
#include <iostream>
#include <sstream>

namespace pretty
{
using namespace std;
using namespace literals;
using Clock = chrono::high_resolution_clock;

static auto start = Clock::now ();
static auto tic = start;
static auto toc = [] {
  auto now = Clock::now ();
  auto delta = chrono::duration<float, milli> (now - tic).count ();
  tic = now;
  return delta;
};

template <bool angled_brackets = false>
static auto block (const string &msg, int width = 11, const string &indent = "  "s) -> ostream &
{
  if constexpr (angled_brackets)
  {
    return cout << indent << "<" << right << setw (width) << msg << ">";
  }
  else
  {
    return cout << indent << "[" << right << setw (width) << msg << "]";
  }
}

static auto timestamp (bool brackets = true) -> ostream &
{
  stringstream s;
  s << setprecision (1) << fixed << toc () << " ms";
  return block (s.str ());
}
} // namespace pretty
#pragma once

#include <chrono>
#include <iomanip>
#include <iostream>
#include <locale>
#include <sstream>

namespace pretty
{
using namespace std;
using namespace literals;
using Clock = chrono::high_resolution_clock;
#define UNUSED __attribute__ ((unused))

static UNUSED auto start = Clock::now ();
static UNUSED auto tic = start;
static UNUSED float delta = 0;
static UNUSED auto toc = [] {
  auto now = Clock::now ();
  delta = chrono::duration<float, milli> (now - tic).count ();
  tic = now;
  return delta;
};

template <char left_char = '|', char right_char = '|'>
static UNUSED auto block (const string &msg, int width = 11, const string &indent = "  "s) -> ostream &
{
  return cout << indent << left_char << right << setw (width) << msg << right_char;
}

static UNUSED auto timestamp (bool brackets = true) -> ostream &
{
  stringstream s;
  s << setprecision (1) << fixed << toc () << " ms";
  return block (s.str ());
}

static UNUSED auto LOCALE = locale ("");
static UNUSED ios DEFAULT{nullptr};
static UNUSED ios SAVE{nullptr};

static UNUSED void save (ostream &o, ios &to = SAVE)
{
  to.copyfmt (o);
}
static UNUSED void restore (ostream &o, ios &from = SAVE)
{
  o.copyfmt (from);
}

} // namespace pretty
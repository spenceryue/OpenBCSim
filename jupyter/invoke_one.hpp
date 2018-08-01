#pragma once

// https://stackoverflow.com/a/40546609/3624264

template <class F1, class F2, class... Args>
auto invokeOne (F1 f1, F2 f2, Args &&... args) -> decltype (f1 (args...))
{
  return f1 (args...);
}

template <class F1, class F2, class... Args>
auto invokeOne (F1 f1, F2 f2, Args &&... args) -> decltype (f2 (args...))
{
  return f2 (args...);
}

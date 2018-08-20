#pragma once

// clang-format off
#ifndef DLL_PUBLIC
  #if defined(_WIN32) || (_WIN64)
    #ifdef EXPORTING
      #define DLL_PUBLIC __declspec(dllexport)
    #else
      #define DLL_PUBLIC __declspec(dllimport)
    #endif
    #define DLL_PRIVATE
  #else
    #define DLL_PUBLIC __attribute__ ((visibility ("default")))
    #define DLL_PRIVATE __attribute__ ((visibility ("hidden")))
  #endif
#endif
// clang-format on

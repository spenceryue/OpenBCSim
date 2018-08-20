include_directories ("${PROJECT_SOURCE_DIR}/include")

# CUDA kernel
add_library (openbcsim_kernel
  OBJECT
  openbcsim_kernel.cu
)
target_compile_options (openbcsim_kernel
  PRIVATE
  -arch=sm_61
  -use_fast_math
  -O3
  # Print per-thread register and memory usage
  --resource-usage
)
target_include_directories (openbcsim_kernel
  PUBLIC
  ${CUDA_INCLUDE_DIRS}
)
target_link_libraries (openbcsim_kernel
  PUBLIC
  ${CUDA_LIBRARIES}
)


# Python module
add_library (openbcsim_module
  SHARED
  openbcsim_module.cpp
  device_properties.cpp
  $<TARGET_OBJECTS:openbcsim_kernel>
)
if (CMAKE_HOST_WIN32)
  # Otherwise:
  # clang++.exe: error: unsupported option '-fPIC' for target 'x86_64-pc-windows-msvc'
  set_target_properties (openbcsim_module
    PROPERTIES
    POSITION_INDEPENDENT_CODE false
    PREFIX ""
  )
endif ()
# Add -I flags
target_include_directories (openbcsim_module
  PUBLIC
  ${CUDA_INCLUDE_DIRS}
  ${Python3_INCLUDE_DIRS}
  ${PyTorch_INCLUDE_DIRS}
)
# Add -D defines or equivalent
target_compile_definitions (openbcsim_module
  PRIVATE
  TORCH_EXTENSION_NAME=openbcsim
  # Enable annotations of exports with __declspec(dllexport)
  EXPORTING
)
# Pass flags to the compiler
target_compile_options (openbcsim_module
  PUBLIC
  -std=c++17
  -O3
  -Wall
  # Suppress warning from `${PyTorch_ROOT}/lib/include/ATen/Scalar.h:12`
  -Wno-microsoft-unqualified-friend
)
# Add relevant -l switches
target_link_libraries (openbcsim_module
  PUBLIC
  ${Python3_LIBRARIES}
  ${PyTorch_LIBRARIES}
  ${CUDA_LIBRARIES}
)
# Use C++17
set_target_properties (openbcsim_module
  PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED true
)
# Set extension (e.g. `*.cp36-win_amd64.pyd` on Windows)
string (REGEX REPLACE "\\.[^.]+$" ".lib" Python3_IMPORT_SUFFIX "${Python3_EXT_SUFFIX}")
set_target_properties (openbcsim_module
  PROPERTIES
  # Basename so that we can just do `import openbcsim` in Python
  OUTPUT_NAME "openbcsim"
  # e.g. `*.cp36-win_amd64.pyd`
  SUFFIX "${Python3_EXT_SUFFIX}"
  # (Property ignored when not on Windows)
  IMPORT_SUFFIX "${Python3_IMPORT_SUFFIX}"
)


# Test program
add_executable (test_openbcsim
  test_openbcsim.cpp
)
# Link with Python module
target_link_libraries (test_openbcsim
  openbcsim_module
)


# Install
install (TARGETS openbcsim_module test_openbcsim DESTINATION "${CMAKE_INSTALL_PREFIX}")


# Define pieces used to write set_runtime_paths script.
if (CMAKE_HOST_WIN32)
  set (EXT ".bat")
  set (FIRST_LINE "@echo off")
  set (COMMENT "rem")
  set (COMMAND "set PATH=%PATH%")
  set (SEP ";")
else () # e.g. Linux
  set (EXT ".sh")
  set (FIRST_LINE "\#!/bin/sh")
  set (COMMENT "\#")
  set (COMMAND "PATH=$PATH")
  set (SEP ":")
endif ()


# Set runtime paths needed to run test_openbcsim executable.
# Also these paths to a script (set_runtime_paths) to be run before
# the executable if testing outside of CMake/CTest.
set (runtime_paths
  "${PyTorch_ROOT}"
  "${PyTorch_ROOT}/lib"
  "${Python3_ROOT}/Library/bin"
  "${Python3_ROOT}/libs"
)
set (set_runtime_paths "${PROJECT_SOURCE_DIR}/scripts/SET_RUNTIME_PATHS${EXT}")
file (WRITE "${set_runtime_paths}" "${FIRST_LINE}\n")
file (APPEND "${set_runtime_paths}" "${COMMENT} These paths need to be added to PATH before running tests.\n")
file (APPEND "${set_runtime_paths}" "${COMMENT} They are added automatically when you run the tests through CTest.\n")
foreach (path IN LISTS runtime_paths)
  file (APPEND "${set_runtime_paths}" "${COMMAND}${SEP}${path}\n")
endforeach (path)


# Report where the script (set_runtime_paths) was saved to.
execute_process (COMMAND
  python -c "\
from pathlib import Path
cwd = Path ().resolve ()
print (cwd.as_posix (), end='')
"
  OUTPUT_VARIABLE cwd
)
file (RELATIVE_PATH set_runtime_paths_relative_path "${cwd}" "${set_runtime_paths}")
message (STATUS "Paths needed to run test_openbcsim.exe saved to '${set_runtime_paths_relative_path}'")


# Add test(s). Add runtime paths to PATH for each test.
string (REPLACE ";" "${SEP}" runtime_paths "${runtime_paths}")
LOG (runtime_paths)
add_test (test_openbcsim
  # test_openbcsim 1 1 1
  test_openbcsim
)
set_tests_properties (test_openbcsim
  PROPERTIES
  ENVIRONMENT "PATH=$ENV{PATH}${SEP}${runtime_paths}"
)

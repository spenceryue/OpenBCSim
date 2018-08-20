# https://www.reddit.com/r/cpp/comments/42wme6/modern_build_systems/
cmake_minimum_required (VERSION 3.12)
project (OpenBCSim LANGUAGES CXX CUDA)

# Settings to use clang++ and lld on Windows.
# Note: Preload.cmake is a special file recognized by CMake
# that runs before the entry point CMakeLists.txt.
# https://stackoverflow.com/a/45247784/3624264
# I'm including it again here (which will run the contents a second time)
# because some of the variables are modified by the project()
# command above.
include (Preload.cmake)

# Path to Visual C++ runtime libraries
if (NOT MSVCRT_DIR AND CMAKE_HOST_WIN32)
  set (MSVCRT_DIR
    "C:/Program Files (x86)/Microsoft Visual Studio 14.0/VC/LIB/amd64"
    CACHE
    PATH
    "Path to Visual C++ Runtime Libraries."
    FORCE
    )
endif ()
# Necessary? clang++ seems to be able to find `msvcrt.lib` just fine.
link_directories ("${MSVCRT_DIR}")


# A hack to suppress the warning:
#   ADD_LIBRARY called with SHARED option but the target platform does not
#   support dynamic linking.  Building a STATIC library instead.  This may lead
#   to problems.
# Probably needed because CMAKE_SYSTEM_NAME was set to "Generic"
# in Preload.cmake.
# https://stackoverflow.com/a/12266163/3624264
set_property (GLOBAL
  PROPERTY
  TARGET_SUPPORTS_SHARED_LIBS true
)


# Default install directory
if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set (CMAKE_INSTALL_PREFIX
    "${PROJECT_SOURCE_DIR}/install"
    CACHE
    PATH
    "The install directory is useful for automatically copying all built files of the project to one place for testing."
    FORCE
  )
endif ()


# Helpful macro to set a Python distutils.sysconfig variable.
macro (SET_PYTHON_CONFIG_VAR output key)
  execute_process (COMMAND
    "${Python3_EXECUTABLE}" -c "import distutils.sysconfig as s; print (s.get_config_var ('${key}'), end='')"
    OUTPUT_VARIABLE
    ${output})
endmacro (SET_PYTHON_CONFIG_VAR)


# Get CUDA version
get_filename_component (CUDA_TOOLKIT_ROOT_DIR "${CMAKE_CUDA_COMPILER}/../.." ABSOLUTE)
link_directories ("${CUDA_TOOLKIT_ROOT_DIR}/lib/x64/")
set (CUDA_CUDART_LIBRARY "cudart.lib")
find_package (CUDA)
message (STATUS "Found CUDA version: ${CUDA_VERSION}")


# Found Python?
find_package (Python3 REQUIRED COMPONENTS Interpreter)
message (STATUS "Found Python3 version: ${Python3_VERSION}")
# https://stackoverflow.com/q/47423246/3624264
SET_PYTHON_CONFIG_VAR (Python3_ROOT "prefix")
SET_PYTHON_CONFIG_VAR (Python3_EXT_SUFFIX "EXT_SUFFIX")
SET_PYTHON_CONFIG_VAR (Python3_INCLUDE_DIRS "INCLUDEPY")
if (CMAKE_HOST_WIN32)
  set (Python3_LIBRARIES
    "${Python3_ROOT}/libs/python36.lib"
    CACHE
    PATH
    ""
    FORCE
  )
else ()
  SET_PYTHON_CONFIG_VAR (Python3_LIBRARIES "LIBDIR")
endif ()
file (TO_CMAKE_PATH "${Python3_ROOT}" Python3_ROOT)
file (TO_CMAKE_PATH "${Python3_INCLUDE_DIRS}" Python3_INCLUDE_DIRS)
file (TO_CMAKE_PATH "${Python3_LIBRARIES}" Python3_LIBRARIES)
file (TO_CMAKE_PATH "${Python3_SITELIB}" Python3_SITELIB)
# Log these to debug
#[[
LOG (Python3_ROOT)
LOG (Python3_INCLUDE_DIRS)
LOG (Python3_LIBRARIES)
LOG (Python3_SITELIB)
LOG ()
#]]


# Found PyTorch?
if (NOT PyTorch_ROOT)
  set (PyTorch_ROOT
    "${Python3_SITELIB}/torch"
    CACHE
    PATH
    "Where PyTorch is installed e.g. C:\\Users\\spenc\\Anaconda3\\lib\\site-packages\\torch"
    FORCE
  )
endif ()
file (TO_CMAKE_PATH "${PyTorch_ROOT}" PyTorch_ROOT)
if (NOT EXISTS "${PyTorch_ROOT}/lib/include/ATen/ATen.h")
  message (FATAL_ERROR
    "Where is PyTorch's root directory? This should exist:\n"
    "\$\{PyTorch_ROOT\}/lib/include/ATen/ATen.h"
  )
else ()
  # Get PyTorch version
  execute_process (COMMAND
    cmd /c cd "${PyTorch_ROOT}" && "${Python3_EXECUTABLE}" -c "import version; print (version.__version__, end='')"
    OUTPUT_VARIABLE PyTorch_VERSION
  )
  message (STATUS "Found PyTorch version: ${PyTorch_VERSION}")
  # Set PyTorch variables
  set (PyTorch_INCLUDE_DIRS
    "${PyTorch_ROOT}/lib/include"
    "${PyTorch_ROOT}/lib/include/TH"
    "${PyTorch_ROOT}/lib/include/THC"
  )
  set (PyTorch_LIBRARIES
    "${PyTorch_ROOT}/lib/_C.lib"
    "${PyTorch_ROOT}/lib/ATen.lib"
    $<${CMAKE_HOST_WIN32}:"msvcrt.lib">
  )
endif ()
# Log these to debug
#[[
LOG (PyTorch_ROOT)
LOG (PyTorch_INCLUDE_DIRS)
LOG (PyTorch_LIBRARIES)
LOG ()
#]]


# Go to CMakeLists.txt of actual targets.
enable_testing ()
add_subdirectory ("${PROJECT_SOURCE_DIR}/src")

# Settings to use clang++ and lld on Windows.
# Note: Preload.cmake is a special file recognized by CMake
# that runs before the entry point CMakeLists.txt.
# https://stackoverflow.com/a/45247784/3624264
set (CMAKE_GENERATOR
  "Ninja"
  CACHE
  STRING
  ""
  FORCE
)
set (CMAKE_BUILD_TYPE
  Release
  CACHE
  STRING
  ""
  FORCE
)
set (CMAKE_LINKER
  "clang++.exe"
  CACHE
  FILEPATH
  ""
  FORCE
)
set (CMAKE_CXX_COMPILER
  "clang++.exe"
  CACHE
  FILEPATH
  ""
  FORCE
)
set (CMAKE_CXX_COMPILER_ID
  "Clang"
  CACHE
  STRING
  ""
  FORCE
)
set (CMAKE_SYSTEM_NAME
  "Generic"
  CACHE
  STRING
  ""
  FORCE
)
set (ENV{LDFLAGS}
  "-fuse-ld=lld -flto"
)
set (CMAKE_STATIC_LIBRARY_PREFIX
  ""
  CACHE
  STRING
  ""
  FORCE
)
set (CMAKE_STATIC_LIBRARY_SUFFIX
  ".lib"
  CACHE
  STRING
  ""
  FORCE
)
set (CMAKE_EXECUTABLE_SUFFIX
  ".exe"
  CACHE
  STRING
  ""
  FORCE
)
set (CMAKE_SHARED_LIBRARY_PREFIX
  ""
  CACHE
  STRING
  ""
  FORCE
)
set (CMAKE_SHARED_LIBRARY_SUFFIX
  ".dll"
  CACHE
  STRING
  ""
  FORCE
)
# https://stackoverflow.com/a/33297026/3624264
set (CMAKE_IMPORT_LIBRARY_SUFFIX
  ".lib"
  CACHE
  STRING
  ""
  FORCE
)

# Useful functions to log variables.
function (LOG)
  if (NOT ARGC)
    message (FATAL_ERROR)
  else ()
    message (STATUS "${ARGN}:  ${${ARGN}}")
  endif ()
endfunction (LOG)

function (LOG_E)
  if (NOT ARGC)
    message (FATAL_ERROR)
  else ()
    message (STATUS "ENV{${ARGN}}:  $ENV{${ARGN}}")
  endif ()
endfunction (LOG_E)

# Log these to debug
#[[
LOG (CMAKE_GENERATOR)
LOG (CMAKE_BUILD_TYPE)
LOG (CMAKE_LINKER)
LOG (CMAKE_CXX_COMPILER)
LOG (CMAKE_CXX_COMPILER_ID)
LOG (CMAKE_SYSTEM_NAME)
LOG_E (LDFLAGS)
LOG (CMAKE_STATIC_LIBRARY_PREFIX)
LOG (CMAKE_STATIC_LIBRARY_SUFFIX)
LOG (CMAKE_EXECUTABLE_SUFFIX)
LOG (CMAKE_SHARED_LIBRARY_PREFIX)
LOG (CMAKE_SHARED_LIBRARY_SUFFIX)
LOG (CMAKE_IMPORT_LIBRARY_SUFFIX)
LOG ()
#]]

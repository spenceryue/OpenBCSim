#!/usr/bin/env python

def build (args):
  import subprocess as sub
  import time
  from pathlib import Path
  from tee import tee
  import os

  BUILD_DIR = args.BUILD_DIR or Path ('build').resolve ()
  LOGS_DIR = args.LOGS_DIR or Path ('logs').resolve ()

  def try_mkdir (path, except_callback=None):
    try:
      Path (path).mkdir ()
    except:
      if except_callback:
        except_callback ()

  try_mkdir (BUILD_DIR)
  try_mkdir (LOGS_DIR)

  # cpp_in = args.cpp_in or [r'openbcsim_module.cpp', 'device_properties.cpp', 'test_module.cpp'])
  cpp_in = args.cpp_in or [x for x in [*Path ().glob ('*.cpp'),
                           *Path ().glob ('*.h'), *Path ().glob ('*.hpp')]]
  cpp_in = [f'{Path (x).resolve ()}' for x in cpp_in]
  # cpp_out = args.cpp_out or str (BUILD_DIR / 'openbcsim_module.o')
  cpp_out = args.cpp_out or ''
  cpp_out = f'{Path (cpp_out).resolve ()}' if cpp_out else ''
  cpp_cmd = args.cpp_cmd or [
      r'clang++',
      r'-I C:\Users\spenc\Anaconda3\lib\site-packages\torch\lib\include',
      r'-I C:\Users\spenc\Anaconda3\lib\site-packages\torch\lib\include\TH',
      r'-I C:\Users\spenc\Anaconda3\lib\site-packages\torch\lib\include\THC',
      r'-I "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\include"',
      r'-I C:\Users\spenc\Anaconda3\include',
      r'-DTORCH_EXTENSION_NAME=openbcsim',
      r'-std=c++17',
      r'-Wall',
      r'-O3' if (args.optimize) else (''), # Costs 3 extra seconds
    ]
  cpp_flags = args.cpp_flags or []
  cpp_cmd = [
      *(x for x in cpp_cmd if x),
      *((rf'-o', cpp_out) if cpp_out else ()),
      r'-c',
      *cpp_flags,
      *[x for x in cpp_in if x.endswith ('.cpp')],
  ]

  # cuda_in = args.cuda_in or ['openbcsim_kernel.cu']
  cuda_in = args.cuda_in or [x for x in [*Path ().glob ('*.cu'),
                             *Path ().glob ('*.cuh')]]
  cuda_in = [f'{Path (x).resolve ()}' for x in cuda_in]
  # cuda_out = args.cuda_out or str (BUILD_DIR / 'openbcsim_kernel.o')
  cuda_out = args.cuda_out or ''
  cuda_out = f'{Path (cuda_out).resolve ()}' if cuda_out else ''
  cuda_cmd = args.cuda_cmd or [
      r'nvcc',
      r'-arch=sm_61',
      r'-use_fast_math',
      r'--resource-usage',
    ]
  cuda_flags = args.cuda_flags or []
  cuda_cmd = [
      *(x for x in cuda_cmd if x),
      *((rf'-o', cuda_out) if cuda_out else ()),
      r'-c',
      *cuda_flags,
      *[x for x in cuda_in if x.endswith ('.cu')],
  ]

  link_in = args.link_in or [x for x in [*BUILD_DIR.glob ('*.o')]]
  link_in = [f'{Path (x).resolve ()}' for x in link_in]
  if not args.executable:
    link_out = args.link_out or 'openbcsim.pyd'
  else:
    link_out = args.link_out or 'openbcsim.exe'
  link_cmd = args.link_cmd or [
      r'clang++',
      r'-shared' if (not args.executable) else (''),
      r'-fuse-ld=lld',
      r'-flto',
      r'-L C:\Users\spenc\Anaconda3\libs',
      r'-L C:\Users\spenc\Anaconda3\lib\site-packages\torch\lib',
      r'-L "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\lib\x64"',
      r'-L "C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\LIB\amd64"',
      r'-L "C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\ATLMFC\LIB\amd64"',
      r'-L "C:\Program Files (x86)\Windows Kits\10\lib\10.0.14393.0\ucrt\x64"',
      r'-L "C:\Program Files (x86)\Windows Kits\NETFXSDK\4.6\lib\um\x64"',
      r'-L "C:\Program Files (x86)\Windows Kits\10\lib\10.0.14393.0\um\x64"',
      r'-l msvcrt',
      r'-l _C',
      r'-l cudart',
      r'-l ATen',
    ]
  link_flags = args.link_flags or []
  link_cmd = [
      *(x for x in link_cmd if x),
      r'-o',
      link_out,
      *link_flags,
      *link_in,
  ]

  def format_ts (seconds, show_date=True, show_time=True):
    if seconds is None:
      return 'N/A'
    t = time.localtime (seconds)
    month = [None, 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep',
             'Oct', 'Nov', 'Dec']
    date = f'{month[t[1]]} {t[2]}'
    hms = f'{(t[3]-1)%12+1:2d}:{t[4]:02d}:{t[5]:02d} {"pm" if t[3]>=12 else "am"}'
    result = []
    if show_date: result.append (date)
    if show_time: result.append (hms)
    return ' '.join (result)

  def format_msg (source, modified, timestamp, force=False, show_ts=False,
                  silent=False, msg_head='  Needs building:  ', path_length=30):
    if silent:
      return
    src = str (source)
    if (show_ts or not force) and len (src) > path_length:
      src = '...' + src[-path_length+3:]
    msg = f'{msg_head}{src:{path_length}}'
    if (show_ts or not force):
      try: show_date = abs (modified - timestamp) > 24 * 3600
      except: show_date = False
      msg += f'  [ edit: {format_ts (modified, show_date)}'
      msg += f' ][ built: {format_ts (timestamp, show_date)} ]'
    print (msg)

  def needs_building (*files, enable=True, build_dict=None, silent=False):
    if needs_building.first:
        needs_building.first = False
        print ()
    if not build_dict:
      build_dict = {}
    if not enable:
      return build_dict
    for file in files:
      source = Path (file)
      if source in build_dict:
        continue
      suffix = source.resolve ().suffix + '.timestamp'
      timestamp_file = LOGS_DIR / source.resolve ().with_suffix (suffix).name
      timestamp = modified = None
      if source.exists ():
        modified = source.stat ().st_mtime
      if timestamp_file.exists ():
        timestamp = float (timestamp_file.read_text ())
        if timestamp != modified or args.force:
          format_msg (source, modified, timestamp, args.force, args.show_ts, silent)
          build_dict[source] = timestamp_file
        elif args.show_ts:
          format_msg (source, modified, timestamp, args.force, args.show_ts, silent,
              msg_head='  Timestamp only:  '
            )
      else:
        format_msg (source, modified, timestamp, True, args.show_ts)
        build_dict[source] = timestamp_file
    return build_dict
  needs_building.first = True

  def record_built (build_dict):
    for source, timestamp_file in build_dict.items ():
      timestamp_file.write_text (str (source.stat ().st_mtime))

  def show_status (error, step, log, tic, after='\n'):
    toc = time.time ()
    sep = ('*') if (error) else ('=')
    msg = ('failed') if (error) else ('completed')
    print (sep*60 + '\n')
    print (f'< {step} {msg} ({toc - tic:.1f} seconds)')
    if error:
      print (f'  See logs at "{log}"')
    tic = time.time ()
    print (after, end='')
    return tic

  def do_step (cmd, step, log, tic):
    if do_step.first:
        do_step.first = False
        print ()
    print ('='*60, end='\n\n')
    print ('> ' + '\n  '.join (cmd), end='\n\n')
    has_line_callback = lambda b: print () if b else None
    tee (' '.join (cmd), file=log, check=True, has_line_callback=has_line_callback)
    return show_status (0, step, log, tic)
  do_step.first = True

  cpp_build_dict = needs_building (*cpp_in, enable=args.cpp)
  cuda_build_dict = needs_building (*cuda_in, enable=args.cuda)
  link_build_dict = needs_building (*link_in, enable=args.link)

  start = time.time ()
  tic = start
  count = 0
  try:
    os.chdir (BUILD_DIR)

    cpp_build_dict = needs_building (*cpp_in, enable=args.cpp, build_dict=cpp_build_dict, silent=True)
    if cpp_build_dict:
      cmd, step, log = cpp_cmd, 'Compile C++', (LOGS_DIR / 'cpp.log')
      tic = do_step (cmd, step, log, tic)
      record_built (cpp_build_dict)
      count += 1

    cuda_build_dict = needs_building (*cuda_in, enable=args.cuda, build_dict=cuda_build_dict, silent=True)
    if cuda_build_dict:
      cmd, step, log = cuda_cmd, 'Compile CUDA', (LOGS_DIR / 'cuda.log')
      tic = do_step (cmd, step, log, tic)
      record_built (cuda_build_dict)
      count += 1

    link_build_dict = needs_building (*link_in, enable=args.link, build_dict=link_build_dict, silent=True)
    if link_build_dict:
      cmd, step, log = link_cmd, 'Link C++ and CUDA', (LOGS_DIR / 'link.log')
      tic = do_step (cmd, step, log, tic)
      record_built (link_build_dict)
      count += 1

  except sub.CalledProcessError as e:
    import sys
    show_status (1, step, log, tic, after='')
    sys.exit (1)

  else:
    if count == 0:
      print ('  Already up to date.')

    else:
      step, log, tic = 'Build', None, start
      show_status (0, step, log, tic, after='')


def from_file (filename):
  from pathlib import Path
  import argparse

  file = Path (filename)
  if not file.exists ():
    msg = f'\n  File doesn\'t exist:  {file}'
    raise argparse.ArgumentTypeError (msg)
  try:
    return file.read_text ()
  except:
    msg = f'\n  Could not read file (check permissions?):  {file}'
    raise argparse.ArgumentTypeError (msg)


if __name__ == '__main__':
  from argparse import ArgumentParser

  description = 'Basically a Makefile to compile the Python module.'
  epilog = 'Defaults to building all if no step flags are given (-c,-u,-l).'
  epilog += ' Also, can merge flags: `-clf` is equivalent to `-c -l -f`.'

  parser = ArgumentParser (description=description, epilog=epilog)
  parser.add_argument ('-c', '-cpp', '-c++',
                       action='store_true',
                       help='Compile C++.',
                       dest='cpp',
                      )
  parser.add_argument ('-u', '-cuda',
                       action='store_true',
                       help='Compile CUDA.',
                       dest='cuda',
                      )
  parser.add_argument ('-l', '-link',
                       action='store_true',
                       help='Link C++ and CUDA.',
                       dest='link',
                      )
  parser.add_argument ('-f', '-force',
                       action='store_true',
                       help='Force build even if source is up-to-date.',
                       dest='force',
                      )
  parser.add_argument ('-t', '-timestamp',
                       action='store_true',
                       help='Show last edited/last built timestamp.',
                       dest='show_ts',
                      )
  parser.add_argument ('-O', '-optimize',
                       action='store_true',
                       help='Use `-O3` flag in Compile C++ step.',
                       dest='optimize',
                      )
  parser.add_argument ('-e', '-executable',
                       action='store_true',
                       help='Build executable (otherwise *.pyd module).',
                       dest='executable',
                      )
  parser.add_argument ('-BUILD_DIR',
                       default=None,
                       type=str,
                       metavar='path',
                       dest='BUILD_DIR',
                      )
  parser.add_argument ('-LOGS_DIR',
                       default=None,
                       type=str,
                       metavar='path',
                       dest='LOGS_DIR',
                      )
  parser.add_argument ('-cpp_in',
                       default=None,
                       type=str,
                       metavar='files',
                       nargs='+',
                       dest='cpp_in',
                      )
  parser.add_argument ('-cpp_out',
                       default=None,
                       type=str,
                       metavar='name',
                       dest='cpp_out',
                      )
  parser.add_argument ('-cpp_flags',
                       default=None,
                       type=lambda x: x.split (','),
                       help='Comma separated list of flags',
                       metavar='flag,flag,...',
                       dest='cpp_flags',
                      )
  parser.add_argument ('-cpp_cmd',
                       default=None,
                       type=from_file,
                       help='Path to command (file).',
                       metavar='<path to file>',
                       dest='cpp_cmd',
                      )
  parser.add_argument ('-cuda_in',
                       default=None,
                       type=str,
                       metavar='files',
                       nargs='+',
                       dest='cuda_in',
                      )
  parser.add_argument ('-cuda_out',
                       default=None,
                       type=str,
                       metavar='name',
                       dest='cuda_out',
                      )
  parser.add_argument ('-cuda_flags',
                       default=None,
                       type=lambda x: x.split (','),
                       help='Comma separated list of flags',
                       metavar='flag,flag,...',
                       dest='cuda_flags',
                      )
  parser.add_argument ('-cuda_cmd',
                       default=None,
                       type=from_file,
                       help='Path to command (file).',
                       metavar='<path to file>',
                       dest='cuda_cmd',
                      )
  parser.add_argument ('-link_in',
                       default=None,
                       type=str,
                       metavar='files',
                       nargs='+',
                       dest='link_in',
                      )
  parser.add_argument ('-link_out',
                       default=None,
                       type=str,
                       metavar='name',
                       dest='link_out',
                      )
  parser.add_argument ('-link_flags',
                       default=None,
                       type=lambda x: x.split (','),
                       help='Comma separated list of flags',
                       metavar='flag,flag,...',
                       dest='link_flags',
                      )
  parser.add_argument ('-link_cmd',
                       default=None,
                       type=from_file,
                       help='Path to command (file).',
                       metavar='<path to file>',
                       dest='link_cmd',
                      )

  args = parser.parse_args ()

  # Default to build all if no step flags set.
  check_flags = ['cpp', 'cuda', 'link']
  if not any (v for k,v in args.__dict__.items () if k in check_flags):
    for k in args.__dict__:
      if k in check_flags:
        args.__dict__[k] = True

  # print (args)
  # build (args)
  print ("Don't use. Run CMake instead.")


'''  project (OpenBCSim LANGUAGES CXX CUDA)


  # Build modes
  set (CMAKE_CONFIGURATION_TYPES
    Debug Release
    CACHE
    TYPE
    INTERNAL
    FORCE
  )
  # Default install directory
  if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set (CMAKE_INSTALL_PREFIX
      "../install"
      CACHE
      PATH
      "The install directory is useful for automatically copying all built files of the project to one place for testing."
      FORCE
    )
  endif ()
  # Source headers
  include_directories (${PROJECT_SOURCE_DIR})


  # Found CUDA?
  if (NOT CUDA_FOUND)
    message (FATAL_ERROR "CUDA not found.")
  else ()
    message (STATUS "Found CUDA version: ${CUDA_VERSION}")
  endif ()
  # Found Python?
  find_package (Python3 COMPONENTS Interpreter Development)
  if (NOT Python3_FOUND)
    message (FATAL_ERROR "Python3 not found.")
  else ()
    message (STATUS "Found Python3 version: ${Python3_VERSION}")
  endif ()
  # Found PyTorch?
  set (PyTorch_ROOT
    "NOT SET"
    CACHE
    PATH
    "Where PyTorch is installed e.g. C:\\Users\\spenc\\Anaconda3\\lib\\site-packages\\torch"
  )
  if (NOT EXISTS "${PyTorch_ROOT}/lib/include/ATen/ATen.h")
    message (FATAL_ERROR "Invalid PyTorch root directory. Should be able to find: ${PyTorch_ROOT}/lib/include/ATen/ATen.h")
  else ()
    # Find PyTorch version
    # https://stackoverflow.com/a/13037728/3624264
    # https://stackoverflow.com/a/5023579/3624264
    execute_process (COMMAND
      cmd /c set /p version= <${PyTorch_ROOT}/version.py && echo %version%
      OUTPUT_VARIABLE PyTorch_VERSION
    )
    message (STATUS "Found PyTorch version: ${PyTorch_VERSION}")
    # Set PyTorch variables
    set (PyTorch_LIBRARIES
      "${PyTorch_ROOT}/lib/_C"
      "${PyTorch_ROOT}/lib/ATen"
    )
    set (PyTorch_INCLUDE_DIRS
      "${PyTorch_ROOT}/lib/include"
      "${PyTorch_ROOT}/lib/include/TH"
      "${PyTorch_ROOT}/lib/include/THC"
    )
  endif ()


  # CUDA kernel
  add_library (cuda_kernel
    openbcsim_kernel.cu
    openbcsim_kernel.cuh
    vector_functions_extended.cuh
  )
  set_target_properties (cuda_kernel
    PROPERTIES
    CUDA_NVCC_FLAGS_DEFAULT
      "-arch=sm_61     \
       -use_fast_math  \
       --resource-usage"
  )


  # Python module
  add_library (python_module
    SHARED
    openbcsim_module.cpp
    openbcsim_module.h
    device_properties.cpp
    device_properties.h
    base.h
    utils.h
    data_types.h
  )
  # Add -I flags
  target_include_directories (python_module
    PUBLIC
    ${Python3_INCLUDE_DIRS}
    ${PyTorch_INCLUDE_DIRS}
  )
  # Add -D defines or equivalent
  target_compile_definitions (python_module
    PRIVATE
    DTORCH_EXTENSION_NAME=python_module
  )
  # Pass flags to the compiler
  target_compile_options (python_module
    PRIVATE
    -std=c++17
    -Wall
    -O3
  )
  # Add relevant -l switches
  target_link_libraries (python_module
    PUBLIC
    ${Python3_LIBRARIES}
    ${PyTorch_LIBRARIES}
    PRIVATE
    cuda_kernel
  )
  # Use C++17
  target_compile_features (python_module
    PUBLIC
    CXX_STANDARD 17
  )
  if (WIN32)
    # Set target extension to *.pyd
    set_target_properties (python_module
      PROPERTIES
      SUFFIX ".pyd"
    )
  endif()


  # Test program
  add_executable (test_python_module
    test_python_module.cpp
    pretty_print.h
  )
  # Link with Python module
  target_link_libraries (test_python_module
    python_module
  )
  # Use C++17
  target_compile_features (test_python_module
    CXX_STANDARD 17
  )
  if (WIN32)
    # Set target extension to *.exe
    set_target_properties (test_python_module
      PROPERTIES
      SUFFIX ".exe"
    )
  endif()


  # Add test(s)
  include (CTest)
  add_test (test_python_module test_python_module --command-line-switch)

'''
#!/usr/bin/env python

def build (args):
  import subprocess as sub
  import time
  from pathlib import Path
  from tee import tee

  BUILD_DIR = args.BUILD_DIR or Path ('build')
  LOGS_DIR = args.LOGS_DIR or Path ('logs')

  def try_mkdir (path, except_callback=None):
    try:
      Path (path).mkdir ()
    except:
      if except_callback:
        except_callback ()

  try_mkdir (BUILD_DIR)
  try_mkdir (LOGS_DIR)

  cpp_in = args.cpp_in or [r'openbcsim_module.cpp']
  cpp_out = args.cpp_out or str (BUILD_DIR / 'openbcsim_module.obj')
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
  cpp_cmd = [
      *(x for x in cpp_cmd if x),
      r'-o',
      cpp_out,
      r'-c',
      *cpp_in,
  ]

  cuda_in = args.cuda_in or ['openbcsim_kernel.cu']
  cuda_out = args.cuda_out or str (BUILD_DIR / 'openbcsim_kernel.obj')
  cuda_cmd = args.cuda_cmd or [
      r'nvcc',
      r'-arch=sm_61',
      r'-use_fast_math',
      r'--resource-usage',
    ]
  cuda_cmd = [
      *(x for x in cuda_cmd if x),
      r'-o',
      cuda_out,
      r'-c',
      *cuda_in,
  ]

  link_in = args.link_in or [cpp_out, cuda_out]
  if not args.executable:
    link_out = args.link_out or str (BUILD_DIR / 'openbcsim.pyd')
  else:
    link_out = args.link_out or str (BUILD_DIR / 'openbcsim.exe')
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
  link_cmd = [
      *(x for x in link_cmd if x),
      r'-o',
      link_out,
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
    show_status (1, step, log, tic, after='')

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
  parser.add_argument ('-B', '-BUILD_DIR',
                       default=None,
                       type=str,
                       metavar='path',
                       dest='BUILD_DIR',
                      )
  parser.add_argument ('-L', '-LOGS_DIR',
                       default=None,
                       type=str,
                       metavar='path',
                       dest='LOGS_DIR',
                      )
  parser.add_argument ('-1', '-cpp_in',
                       default=None,
                       type=str,
                       metavar='files',
                       nargs='+',
                       dest='cpp_in',
                      )
  parser.add_argument ('-2', '-cpp_out',
                       default=None,
                       type=str,
                       metavar='name',
                       dest='cpp_out',
                      )
  parser.add_argument ('-3', '-cpp_cmd',
                       default=None,
                       type=from_file,
                       help='Path to command (file).',
                       metavar='<path to file>',
                       dest='cpp_cmd',
                      )
  parser.add_argument ('-4', '-cuda_in',
                       default=None,
                       type=str,
                       metavar='files',
                       nargs='+',
                       dest='cuda_in',
                      )
  parser.add_argument ('-5', '-cuda_out',
                       default=None,
                       type=str,
                       metavar='name',
                       dest='cuda_out',
                      )
  parser.add_argument ('-6', '-cuda_cmd',
                       default=None,
                       type=from_file,
                       help='Path to command (file).',
                       metavar='<path to file>',
                       dest='cuda_cmd',
                      )
  parser.add_argument ('-7', '-link_in',
                       default=None,
                       type=str,
                       metavar='files',
                       nargs='+',
                       dest='link_in',
                      )
  parser.add_argument ('-8', '-link_out',
                       default=None,
                       type=str,
                       metavar='name',
                       dest='link_out',
                      )
  parser.add_argument ('-9', '-link_cmd',
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
  build (args)

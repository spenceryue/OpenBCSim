#!/usr/bin/env python

def build (args, LOGS_DIR=None, cpp_in=None, cpp_out=None, compile_cpp=None,
           cuda_in=None, cuda_out=None, compile_cuda=None, link_in=None,
           link_out=None, link=None
          ):
  import subprocess as sub
  import time
  from tee import tee
  from pathlib import Path

  LOGS_DIR = LOGS_DIR or Path ('logs')
  try:
    LOGS_DIR.mkdir ()
  except:
    pass


  cpp_in = cpp_in or [r'openbcsim_module.cpp']
  cpp_out = cpp_out or r'build\temp.win-amd64-3.6\Release\openbcsim_module.o'
  compile_cpp = compile_cpp or [
    r'clang++',
    r'-I C:\Users\spenc\Anaconda3\lib\site-packages\torch\lib\include',
    r'-I C:\Users\spenc\Anaconda3\lib\site-packages\torch\lib\include\TH',
    r'-I C:\Users\spenc\Anaconda3\lib\site-packages\torch\lib\include\THC',
    r'-I "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\include"',
    r'-I C:\Users\spenc\Anaconda3\include',
    r'-DTORCH_EXTENSION_NAME=openbcsim',
    r'-std=c++17',
    # r'-O3', # Costs 3 extra seconds
    r'-Wall',
    r'-o',
    cpp_out,
    r'-c',
    *cpp_in,
  ]

  cuda_in = cuda_in or ['openbcsim_kernel.cu']
  cuda_out = cuda_out or r'build\temp.win-amd64-3.6\Release\openbcsim_kernel.o'
  compile_cuda = compile_cuda or [
    r'nvcc',
    r'-arch=sm_61',
    r'-use_fast_math',
    r'-Xptxas="-v"',
    r'-o',
    cuda_out,
    r'-c',
    *cuda_in,
  ]

  link_in = link_in or [cpp_out, cuda_out]
  link_out = link_out or r'build\lib.win-amd64-3.6\openbcsim.cp36-win_amd64.pyd'
  link = link or [
    r'clang++',
    r'-shared',
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

  def format_msg (source, modified, stamp, force=False, show_ts=False,
                  msg_head='  Needs building:  ', path_length=30):
    src = str (source)
    if (show_ts or not force) and len (src) > path_length:
      src = '...' + src[-path_length+3:]
    msg = f'{msg_head}{src:{path_length}}'
    if (show_ts or not force):
      try: show_date = abs (modified - stamp) > 24 * 3600
      except: show_date = False
      msg += f'  [ edit: {format_ts (modified, show_date)}'
      msg += f' ][ built: {format_ts (stamp, show_date)} ]'
    print (msg)

  def needs_building (*files, enable=True):
    if needs_building.first:
        needs_building.first = False
        print ()
    if not enable:
      return
    file_pairs = []
    for file in files:
      source = Path (file)
      stamp_file = LOGS_DIR / source.with_suffix (source.suffix + '.timestamp').name
      stamp = modified = None
      if source.exists ():
        modified = source.stat ().st_mtime
      if stamp_file.exists ():
        stamp = float (stamp_file.read_text ())
        if stamp != modified or args.force:
          format_msg (source, modified, stamp, args.force, args.show_ts)
          file_pair = (stamp_file.resolve (), source.resolve ())
          file_pairs.append (file_pair)
        elif args.show_ts:
          format_msg (source, modified, stamp, args.force, args.show_ts,
              msg_head='  Timestamp only:  '
            )
      else:
        format_msg (source, modified, stamp, True, args.show_ts)
        file_pair = (stamp_file.resolve (), source.resolve ())
        file_pairs.append (file_pair)
    return file_pairs
  needs_building.first = True

  def record_built (built_list):
    for stamp_file, stamp in built_list:
      stamp_file.write_text (str (stamp.stat ().st_mtime))

  cpp_build_list = needs_building (*cpp_in, enable=args.cpp)
  cuda_build_list = needs_building (*cuda_in, enable=args.cuda)
  link_build_list = needs_building (*link_in, enable=args.link)

  start = time.time ()
  tic = start

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

  count = 0
  try:
    if args.cpp and cpp_build_list:
      cmd, step, log = compile_cpp, 'Compile C++', LOGS_DIR / 'cpp.log'
      tic = do_step (cmd, step, log, tic)
      record_built (cpp_build_list)
      count += 1
    if args.cuda and cuda_build_list:
      cmd, step, log = compile_cuda, 'Compile CUDA', LOGS_DIR / 'cuda.log'
      tic = do_step (cmd, step, log, tic)
      record_built (cuda_build_list)
      count += 1
    if args.link and (link_build_list or any ([cpp_build_list, cpp_build_list])):
      cmd, step, log = link, 'Link C++ and CUDA', LOGS_DIR / 'link.log'
      tic = do_step (cmd, step, log, tic)
      record_built (link_build_list)
      count += 1
  except sub.CalledProcessError as e:
    show_status (1, step, log, tic, after='')
    return

  if count == 0:
    print ('  Already up to date.')
  else:
    step = 'Build'
    tic = start
    show_status (0, step, None, tic, after='')


if __name__ == '__main__':
  from argparse import ArgumentParser

  description = 'Basically a Makefile to compile the Python module.'
  epilog = 'Defaults to build all if no build step flags given.'
  epilog += ' Also, can merge flags: `-clf` is equivalent to `-c -l -f`.'

  parser = ArgumentParser (description=description, epilog=epilog)
  parser.add_argument ('-c', '-cpp', '--cpp', '-c++', '--c++',
                       action='store_true',
                       help='Compile C++',
                       dest='cpp',
                      )
  parser.add_argument ('-u', '-cuda', '--cuda',
                       action='store_true',
                       help='Compile CUDA',
                       dest='cuda',
                      )
  parser.add_argument ('-l', '-link', '--link',
                       action='store_true',
                       help='Link C++ and CUDA',
                       dest='link',
                      )
  parser.add_argument ('-f', '-force', '--force',
                       action='store_true',
                       help='Force build even if source is up-to-date.',
                       dest='force',
                      )
  parser.add_argument ('-t', '-timestamp', '--timestamp',
                       action='store_true',
                       help='Show last edited/last built timestamp.',
                       dest='show_ts',
                      )
  args = parser.parse_args ()

  # Default to build all if no step flags set.
  exclude_flags = ['force', 'show_ts']
  if not any (v for k,v in args.__dict__.items () if k not in exclude_flags):
    for k in args.__dict__:
      if k not in exclude_flags:
        args.__dict__[k] = True

  build (args, link_out='openbcsim.cp36-win_amd64.pyd')

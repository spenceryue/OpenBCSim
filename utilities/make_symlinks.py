#!/usr/bin/env python

import os
import argparse

description = '''Make symlinks to all the needed DLLs from the `install/bin` \
directory in the build folder so you don't need to copy them manually all \
the time.'''

parser = argparse.ArgumentParser (description=description)
parser.add_argument ('--src', default='../install/bin', help='The `install/bin directory` containing the DLLs to symlink to.')
parser.add_argument ('--dest', default='../build/examples/Debug', help='The `build/<sub project>/Debug` folder to place the symlinks.')
args = parser.parse_args ()

if os.name == 'nt':
  # See https://stackoverflow.com/a/1447651/3624264
  import ctypes
  kdll = ctypes.windll.LoadLibrary ('kernel32.dll')

source = os.path.abspath (args.src)
dest = os.path.abspath (args.dest)

for file in os.listdir (source):
  if os.path.splitext (file)[1] == '.exe':
    continue

  link = os.path.join (dest, file)
  target = os.path.join (source, file)

  if os.name == 'nt':
    result = kdll.CreateSymbolicLinkW (link, target, 2)
  else:
    result = True
    os.symlink (target, link)

  if result:
    print ('Symbolic link created for:\n' +
           '  {}\n'.format (link) +
           '  <<===>>\n' +
           '  {}\n'.format (target)
    )
  else:
    print ('Symbolic link creation FAILED for:\n' +
           '  {}\n'.format (link) +
           '  <<===>>\n' +
           '  {}\n'.format (target)
    )


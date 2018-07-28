#!/usr/bin/env python

import os
import argparse

description = '''Make symlinks to all the needed DLLs from the `install/bin` \
directory in the build debug folder so you don't need to copy them manually \
all the time.'''

parser = argparse.ArgumentParser (description=description)
parser.add_argument ('--src',
  default='../install/bin',
  help='The folder containing the DLLs to symlink to. (default: `../install/bin`)')
parser.add_argument ('--dest',
  default='../build/examples/Debug',
  help='The folder folder to place the symlinks. (default: `../build/examples/Debug`)')
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
    print ('Symbolic link created at:\n' +
           '  {} --->\n'.format (link) +
           '  {}\n'.format (target)
    )
  else:
    print ('Symbolic link creation FAILED at:\n' +
           '  {} --->\n'.format (link) +
           '  {}\n'.format (target)
    )


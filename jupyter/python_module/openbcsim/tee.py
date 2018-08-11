#!/usr/bin/env python

def tee (cmd,
  file=None,
  check=False,
  mode='wt',
  stdout=None,
  prefix='',
  has_line_callback=None,
  bufsize=1,
  **kwargs
):
  import subprocess as sub
  import sys

  p = sub.Popen (
    cmd,
    stdout=sub.PIPE,
    stderr=sub.STDOUT,
    universal_newlines=True,
    bufsize=bufsize,
    **kwargs,
  )

  if stdout is None:
    stdout = sys.stdout

  def run (file=None):
    with p:
      has_line = False
      for line in p.stdout:
        has_line = True
        print (prefix + line, end='', file=stdout)
        if file is not None:
          print (line, end='', file=file)
      if has_line_callback is not None:
        has_line_callback (has_line)

  if file is not None:
    with open (file, mode) as f:
      run (f)
  else:
    run ()

  if check and p.returncode != 0:
    raise sub.CalledProcessError (p.returncode, cmd)

if __name__ == '__main__':
  import argparse

  parser = argparse.ArgumentParser ()
  parser.add_argument ('cmd', type=str, nargs='+')
  parser.add_argument ('-f', type=str, default=None, help='Output file', dest='file')
  parser.add_argument ('-s', action='store_true', dest='shell', help='Run a shell command')
  args = parser.parse_args ()
  tee (**args.__dict__)

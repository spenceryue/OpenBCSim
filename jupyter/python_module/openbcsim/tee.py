def tee (cmd,
  file=None,
  check=False,
  mode='wt',
  stdout=None,
  prefix='',
  has_line_callback=None,
  **kwargs
):
  import subprocess as sub
  import sys

  p = sub.Popen (
    cmd,
    stdout=sub.PIPE,
    stderr=sub.STDOUT,
    universal_newlines=True,
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
          print (line, file=file)
      if has_line_callback is not None:
        has_line_callback (has_line)

  if file is not None:
    with open (file, mode) as f:
      run (f)
  else:
    run ()

  if check and p.returncode != 0:
    raise sub.CalledProcessError (p.returncode, cmd)

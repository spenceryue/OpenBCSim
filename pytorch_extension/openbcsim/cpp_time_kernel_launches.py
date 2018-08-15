import os
from pathlib import Path

lib = []
lib.append (Path (r'C:\Users\spenc\Anaconda3\pkgs\pytorch-0.4.0-py36_cuda80_cudnn7he774522_1\Lib\site-packages\torch\lib'))
lib.append (Path (r'C:\Users\spenc\Anaconda3\pkgs\pytorch-0.4.0-py36_cuda80_cudnn7he774522_1\Lib\site-packages\torch'))
lib.append (Path (r'C:\Users\spenc\Anaconda3\libs'))
assert all (x.exists () for x in lib)

os.environ['PATH'] += ';' + ';'.join ([str (x) for x in lib])

exe = Path ('build/test_module.exe')
assert exe.exists ()

def run_once (num_elements):
  import tee
  import time
  tic = time.clock ()
  # 0 means don't use PyTorch
  with_pytorch = 0
  num_scatterers = int (1e6)
  tee.tee (f'{exe} {with_pytorch} {num_scatterers} {num_elements}')
  return time.clock () - tic

import time_kernel_launches as tm
tm.run_once = run_once
tm.main (out='logs/times.no_pytorch.grid_stride_loop.1M_scatterers.json')

r'''
Run the following commands first if executing directly from command line:

set path=%path%;C:\Users\spenc\Anaconda3\pkgs\pytorch-0.4.0-py36_cuda80_cudnn7he774522_1\Lib\site-packages\torch
set path=%path%;C:\Users\spenc\Anaconda3\pkgs\pytorch-0.4.0-py36_cuda80_cudnn7he774522_1\Lib\site-packages\torch\lib;C:\Users\spenc\Anaconda3\Library\bin
set path=%path%;C:\Users\spenc\Anaconda3\libs
'''

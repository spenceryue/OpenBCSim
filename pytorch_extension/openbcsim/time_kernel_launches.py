#!/usr/bin/env python

import sys; sys.path.append ('build') # To find openbcsim module
import torch
import Transducer, Simulator
from pathlib import Path
import json
import time

def run_once (num_elements):
  tx = Transducer.LinearTransducer (num_elements)
  sim = Simulator.Simulator (transmitter=tx)

  data_path = Path (r'C:\Users\spenc\Desktop\Experiments\ftp_files\cyst_phantom\pht_data.mat')
  assert data_path.exists ()
  sim.load_field_ii_scatterers (f'{data_path}', verbose=True)
  sim.stats ()

  result = sim.launch ()
  print (result.shape)
  return sim.toc

def main (start=0, stop=1001, step=50, out='logs/times.json', rename_collision=True):
  if out is not None:
    output_file = Path (out)
    while output_file.exists ():
      if rename_collision:
        output_file = output_file.parent / (output_file.stem + '.new' + output_file.suffix)
      else:
        raise NameError (f'  File already exists, not overwriting: {output_file}')
  tic = time.time ()
  times = {}
  try:
    for num_elements in range (start, stop, step):
      print ('\n' + '=' * 60 + '\n')
      times[num_elements] = run_once (num_elements)
      print ('\n' + '=' * 60 + '\n')
      print (f'  Num elements: {num_elements}  ({times[num_elements]:.1f} seconds)')
  except Exception as e:
    print (f'  Broke at {num_elements}.')
    print (e)
  except KeyboardInterrupt:
    pass
  finally:
    if out is not None:
      print (f'  Logging results to {output_file}')
      with output_file.open ('wt') as f:
        json.dump (times, f, indent=2)
    toc = time.time ()
    print (f'  Timing completed ({toc - tic:.1f} seconds)')

if __name__ == '__main__':
  from argparse import ArgumentParser

  parser = ArgumentParser ()
  parser.add_argument ('-start', default=0)
  parser.add_argument ('-stop', default=1001)
  parser.add_argument ('-step', default=50)
  parser.add_argument ('-out', default='logs/times.json')
  args = parser.parse_args ()
  main (**args)

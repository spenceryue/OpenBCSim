# Same as the notebook

import numpy as np


# Load data
import os, os.path

folder = '../install/bin/GPUExample2'
files = os.listdir (folder)

data = []
# for i, file in enumerate (files, 1):
from tqdm import tqdm
for file in tqdm (files, ascii=True):
  # print ('Line {} of {}'.format (i, len (files)), end='\r')
  import sys; sys.stdout.flush ()
  with open (os.path.join (folder, file), 'rt') as f:
    iq_line = []
    for i, line in enumerate (f):
      value = np.complex (*[float (x) for x in line.strip ().split (',')])
      iq_line.append (value)
  data.append (np.array (iq_line))
data = np.array (data)
print ()

# Plot results
from visualize import visualize
import matplotlib.pyplot as plt

ratio = data.shape[1]/data.shape[0]
# visualize (data, 'GPU Example 2 Output', save_path='../figures/gpu_example2.output.png', aspect_ratio=ratio, equalize=1)
visualize (data, 'GPU Example 2 Output', aspect_ratio=ratio, equalize=1)
# visualize (data, 'GPU Example 2 Output', aspect_ratio=ratio, min_dB=-30)
# visualize (data, 'GPU Example 2 Output', aspect_ratio=ratio)

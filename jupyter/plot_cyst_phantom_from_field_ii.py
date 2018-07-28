#!/usr/bin/env python

# See also the jupyter notebook that does the same thing.

import numpy as np


# Load cyst phantom scatterer data

import scipy.io as spio

file = r'C:\Users\spenc\Desktop\Experiments\ftp_files\cyst_phantom\pht_data.mat'
mat = spio.loadmat (file)

pos, amp = mat['phantom_positions'], mat['phantom_amplitudes']
scatterers = np.concatenate ([e.astype ('float32') for e in [pos, amp]], axis=-1)


# Generate the transducer apertures for send and receive

f0 = 3.5e6                 #  Transducer center frequency [Hz]
fs = 100e6                 #  Sampling frequency [Hz]
c = 1540                   #  Speed of sound [m/s]
lambd = c / f0             #  Wavelength [m]
width = lambd              #  Width of element
height = 5 / 1000          #  Height of element [m]
kerf = 0.05 / 1000         #  Kerf (gap between elements) [m]
pitch = kerf + width       #  Pitch (center-to-center distance between elements) [m]
N_elements = 192           #  Number of physical elements
no_sub_x = 1               #  Number of sub-divisions in x-direction of elements
no_sub_y = 10              #  Number of sub-divisions in y-direction of elements


# Create and configure GPU simulator

from pyrfsim import RfSimulator

sim = RfSimulator ('gpu')
sim.set_print_debug (True)

sim.set_parameter ('sound_speed', str (c))
sim.set_parameter ('radial_decimation', '1') # depth-direction downsampling factor
sim.set_parameter ('phase_delay', 'on') # improves subsample accuracy
sim.set_parameter ('use_elev_hack', 'off')


def subdivide (length, num):
  delta = length * 1/num
  divisions = np.arange ((-num//2 + 1) * delta, (num//2 + 1) * delta, delta)

  return divisions

# Arguments very similar to xdc_linear_array in Field II
def linear_transducer (N_elements, width, height, kerf, no_sub_x=1, no_sub_y=1,
                       as_array=False):
  '''Calculates the origin positions of the (sub-)elements in a linear
  array transducer.'''

  elem_x = subdivide (width, no_sub_x)
  elem_y = subdivide (height, no_sub_y)

  template_x = np.tile (elem_x, [1, N_elements])
  template_x = template_x.repeat (no_sub_y, axis=0)
  template_y = np.tile (elem_y.reshape(-1, 1), [1, N_elements])
  template_y = template_y.repeat (no_sub_x, axis=1)

  pitch = width + kerf # element center-to-center distance
  origins_x = np.arange (
    (-N_elements//2 + 1) * pitch,
    (N_elements//2 + 1) * pitch,
    pitch
  ).reshape (1, -1)
  origins_x = origins_x.repeat (no_sub_x, axis=1)
  origins_x = origins_x.repeat (no_sub_y, axis=0)

  transducer_x = template_x + origins_x
  transducer_y = template_y
  transducer_z = np.zeros (transducer_x.shape)

  if as_array:
    return np.stack ([transducer_x, transducer_y, transducer_z], axis=2)
  else:
    return {'x': transducer_x, 'y': transducer_y, 'z': transducer_z}


# Define a scan sequence

receive_aperture = linear_transducer (
  N_elements,
  width,
  height,
  kerf,
  no_sub_x,
  no_sub_y,
  as_array=True
)

origins = receive_aperture.reshape (-1, 3).astype ('float32')
N_subelements = origins.shape[0]
directions = np.tile (np.array ([0, 0, 1], dtype='float32'), [N_subelements, 1])
lateral_dirs = np.tile (np.array ([1, 0, 0], dtype='float32'), [N_subelements, 1])
timestamps = np.zeros (N_subelements, dtype='float32')
line_length = .09

sim.set_scan_sequence (origins, directions, line_length, lateral_dirs, timestamps)


# Plot transducer origins

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure (figsize=(12,8))
ax = fig.gca (projection='3d')
ax.scatter (
  origins[:,0],
  origins[:,1],
  origins[:,2],
  c=np.arange (origins.shape[0]),
  s=5,
  cmap='plasma'
)

ax.text (*origins[0], 'o Start', fontsize=14, weight='bold')
stride = N_elements * no_sub_x // 6
for i in range (stride, origins.shape[0], stride):
    ax.text (*origins[i], 'o {}'.format (i), fontsize=12, weight='ultralight')
ax.text (*origins[-1], 'o Stop', fontsize=14, weight='bold')

ax.set_xlabel ('x', color='red', fontsize=16, labelpad=10)
ax.set_ylabel ('y', color='red', fontsize=16, labelpad=10)
ax.set_zlabel ('z', color='red', fontsize=16, labelpad=10)
ax.set_title ('Transducer Origins', color='red', fontsize=22, pad=50)
ax.set_facecolor ('white')
ax.view_init (elev=60, azim=225)


# Define excitation signal

from scipy.signal import gausspulse

t = np.arange (-16/f0, 16/f0, 1/fs)
excitation, envelope  = gausspulse(t, fc=f0, bw=.5, retenv=True)
excitation = excitation.astype ('float32')
center_index = len (t) // 2
sim.set_excitation (excitation, center_index, fs, f0)


# Plot excitation signal

plt.figure ()
plt.plot(t, envelope, 'g--', t, excitation)
plt.title ('Excitation', fontsize=16, y=-.15)
plt.gca ().set_facecolor ('white')
plt.xticks ([])
plt.yticks ([])


# Record range of scatterer data

x_range = scatterers[:,0].min (), scatterers[:,0].max ()
y_range = scatterers[:,1].min (), scatterers[:,1].max ()
z_range = scatterers[:,2].min (), scatterers[:,2].max ()
a_range = scatterers[:,3].min (), scatterers[:,3].max ()


# Set scatterers

sim.clear_fixed_scatterers () # Make this cell idempotent

data = scatterers

sim.add_fixed_scatterers (data)


# Set the beam profile

sigma_lateral = 1e-3
sigma_elevational = 1e-3
sim.set_analytical_beam_profile (sigma_lateral, sigma_elevational)


# Simulate all scanlines

decimation = 1
sim.set_parameter ('radial_decimation', str (decimation))
sim.set_parameter ('use_elev_hack', 'off')
iq_lines = sim.simulate_lines ()
print (iq_lines.shape)


# Reshaping madness.
# Key to tip: The last axis gets read the fastest, the first axis the fastest.

shape = no_sub_y, N_elements, no_sub_x, -1
reshaped = iq_lines.T.reshape (shape)
data = reshaped.sum (axis=(0,2)).T / (no_sub_y*no_sub_x)


# Crop data

ROI = [z_range, x_range]
ROI = [[z*2/c*fs/decimation for z in z_range], [x/pitch for x in x_range]]

ROI[0] = [ROI[0][0], ROI[0][1]]
ROI[1] = [x+data.shape[1]/2 for x in ROI[1]]
ROI = [[int (e) for e in lims] for lims in ROI]
cropped = data[slice (*ROI[0]), slice (*ROI[1])]
print (cropped.shape, ROI[0])


# Plot results

from visualize import *

title = 'Cyst Phantom'

# Aspect ratio is height/width.
# The factor of 1/2 is the ratio to convert from units of time to
# depth because travel time is double the distance in the measured
# echo roundtrip.
# ratio = (c/fs/2) / (pitch/no_sub_x)
ratio = (c/fs/2) / (pitch)

# Scale back for decimation
ratio *= decimation

fig, axes = visualize (
  cropped,
  title,
  figsize=(18, 9),
  min_dB=-30,
  aspect_ratio=ratio,
  show=False
)

# Set tick marks with real distances
ticks, _ = plt.xticks ()
ticks = [x for x in ticks[:-1] if x >= 0]
labels = [x*pitch+x_range[0] for x in ticks]
labels = ['{:.1f}'.format (1000*l) for l in labels]
plt.xticks (ticks, labels)
plt.xlabel ('Width [mm]', fontsize=14, labelpad=15)

ticks, _ = plt.yticks ()
ticks = [t for t in ticks[:-1] if t >= 0]
labels = [decimation*t/(2*fs)*c+z_range[0] for t in ticks]
labels = ['{:.1f}'.format (1000*l) for l in labels]
plt.yticks (ticks, labels)
plt.ylabel ('Depth [mm]', fontsize=14, labelpad=15)

plt.sca (axes[0])
ticks, _ = plt.xticks ()
ticks = [t for t in ticks[:-1] if t >= 0]
labels = [decimation*t/(2*fs)*c+z_range[0] for t in ticks]
labels = ['{:.1f}'.format (1000*l) for l in labels]
plt.xticks (ticks, labels)
plt.xlabel ('Depth [mm]', fontsize=14, labelpad=15)

plt.show ()

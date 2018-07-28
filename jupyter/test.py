#!/usr/bin/env python

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
no_sub_x = 5               #  Number of sub-divisions in x-direction of elements
no_sub_y = 5              #  Number of sub-divisions in y-direction of elements


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
def linear_transducer (N_elements, width, height, kerf, no_sub_x=1, no_sub_y=1, as_array=False):
  '''Calculates the origin positions of the (sub-)elements in a linear array transducer.'''

  elem_x = subdivide (width, no_sub_x)
  elem_y = subdivide (height, no_sub_y)

  template_x = np.tile (elem_x, [1, N_elements])
  template_x = template_x.repeat (no_sub_y, axis=0)
  template_y = np.tile (elem_y.reshape(-1, 1), [1, N_elements])
  template_y = template_y.repeat (no_sub_x, axis=1)

  pitch = width + kerf # element center-to-center distance
  origins_x = np.arange ((-N_elements//2 + 1) * pitch, (N_elements//2 + 1) * pitch, pitch).reshape (1, -1)
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

receive_aperture = linear_transducer (N_elements, width, height, kerf, no_sub_x, no_sub_y, as_array=True)

origins = receive_aperture.reshape (-1, 3).astype ('float32')
N_subelements = origins.shape[0]
directions = np.tile (np.array ([0, 0, 1], dtype='float32'), [N_subelements, 1])
lateral_dirs = np.tile (np.array ([1, 0, 0], dtype='float32'), [N_subelements, 1])
timestamps = np.zeros (N_subelements, dtype='float32')
line_length = .09

sim.set_scan_sequence (origins, directions, line_length, lateral_dirs, timestamps)


# Define excitation signal

from scipy.signal import gausspulse

t = np.arange (-16/f0, 16/f0, 1/fs)
excitation, envelope  = gausspulse(t, fc=f0, bw=.5, retenv=True)
excitation = excitation.astype ('float32')
center_index = len (t) // 2
sim.set_excitation (excitation, center_index, fs, f0)


# Set scatterers

sim.clear_fixed_scatterers () # Make this cell idempotent

# transmit_aperture = linear_transducer (N_elements, width, height, kerf, no_sub_x, no_sub_y, as_array=True)
data = scatterers

sim.add_fixed_scatterers (data)


# Set the beam profile

sigma_lateral = 1e-3
sigma_elevational = 1e-3
sim.set_analytical_beam_profile (sigma_lateral, sigma_elevational)


# Simulate all scanlines

sim.set_parameter ('radial_decimation', '17') # test radial decimation by a number that does not divide m_rf_line_num_samples evenly
sim.set_parameter ('use_elev_hack', 'off')
iq_lines = sim.simulate_lines ()
print (iq_lines)
import sys; sys.stdout.flush()


# Test resizing the number of lines

no_sub_x = 3               #  Number of sub-divisions in x-direction of elements
no_sub_y = 5              #  Number of sub-divisions in y-direction of elements


# Define a scan sequence

receive_aperture = linear_transducer (N_elements, width, height, kerf, no_sub_x, no_sub_y, as_array=True)

origins = receive_aperture.reshape (-1, 3).astype ('float32')
N_subelements = origins.shape[0]
directions = np.tile (np.array ([0, 0, 1], dtype='float32'), [N_subelements, 1])
lateral_dirs = np.tile (np.array ([1, 0, 0], dtype='float32'), [N_subelements, 1])
timestamps = np.zeros (N_subelements, dtype='float32')
line_length = .09

sim.set_scan_sequence (origins, directions, line_length, lateral_dirs, timestamps)


# Simulate all scanlines

sim.set_parameter ('radial_decimation', '7') # test radial decimation by a number that does not divide m_rf_line_num_samples evenly
sim.set_parameter ('use_elev_hack', 'off')
iq_lines = sim.simulate_lines ()
print (iq_lines)
import sys; sys.stdout.flush()

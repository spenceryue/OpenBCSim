import torch
import openbcsim  # Must come after `import torch`
from Transducer import *


class Simulator:
  '''Data and parameters needed to run a simulation.'''

  def __init__ (self, sampling_frequency=100e6, decimation=10,
                scan_depth=9e-2, speed_of_sound=1540, attenuation=.7,
                tx=Transducer (), rx=None):
    '''The value of `sampling_frequency / decimation` tells how many
    samples to record per a second of simulation (Hz).

    `center_frequency` (Hz) describes the carrier wave frequency of the
    `tx` transducer.
    `attenuation` is the decrease in amplitude of the signal over
    distance travelled in dB/MHz/cm.
    `scan_depth` is how deep into the tissue (m) to simulate.
    `speed_of_sound` is the uniform speed assumed to be constant
    throughout the tissue (m/s).
    `tx` and `rx` are `Transducer` instances (possibly the same
    instance) that
    describe the element geometry and delay/apodization settings of the
    transducer(s) used to
    transmit and receive the ultrasound signal.

    `rx` defaults to the same instance as `tx` if not given.

    The `tensor_type` attribute (specifying the dtype and device of all
    tensors owned) of
    `tx` and `rx` must be the same.
    '''

    # Store parameters given.
    self.sampling_frequency = sampling_frequency
    self.decimation = decimation
    self.scan_depth = scan_depth
    self.speed_of_sound = speed_of_sound
    self.attenuation = attenuation

    # Save transducers.
    self.tx = tx
    self.rx = (rx) if (rx is not None) else (tx)

    # Check that the transducers have the same `tensor_type`.
    if not self.tx.has_same_tensor_type (self.rx):
      msg = 'tx and rx `Transducer` instances must have the ' \
          'same tensor_type (i.e. dtype and device).'
      msg += f'\ntx tensor_type:  {self.tx.tensor_type}'
      msg += f'\nrx tensor_type:     {self.rx.tensor_type}'
      raise ValueError (msg)
    else:
      self.dtype = self.tx.dtype
      self.device = self.tx.device
      self.tensor_type = self.tx.tensor_type

    # Compute number of time samples needed.
    self.num_time_samples = self.compute_time_samples (self.scan_depth,
                                                       self.speed_of_sound,
                                                       self.sampling_frequency)

    # Bind these tensor constructors for convenience.
    self.new_tensor = tx.new_tensor
    self.new_ones = tx.new_ones
    self.new_zeros = tx.new_zeros

  def check_shape (self, attributes, true_shape):
    '''Check that the list of `attributes` (given by their string names)
    have the desired `true_shape`.'''
    true_shape = tuple (true_shape)
    for attr in attributes:
      value = getattr (self, attr)
      if value is None:
        continue
      shape = tuple (value.shape)
      if shape != true_shape:
        msg = f'Shape of self.{attr} {shape} does not match desired true shape {true_shape}.'
        raise ValueError (msg)

  def set_gaussian_excitation (self, plot=False, bw=.5, bwr=-6, num_sigmas=3):
    '''Define the ultrasound excitation signal.
    `bw` is the fractional bandwidth (between 0 and 1) of the Gaussian
    modulating signal relative to `tx.center_frequency.
    `bwr` is the reference level at which the fractional bandwidth is
    calculated (dB).
    (See scipy.signal.gausspulse.)
    `num_sigmas` is how many standard deviations outwards from the
    center to sample in either direction.'''

    from scipy.signal import gausspulse
    import matplotlib.pyplot as plt

    fc = self.tx.center_frequency
    fs = self.sampling_frequency

    # https://github.com/scipy/scipy/blob/14142ff70d84a6ce74044a27919c850e893648f7/scipy/signal/waveforms.py#L232
    # exp(-a t^2) <->  sqrt(pi/a) exp(-pi^2/a * f^2)  = g(f)
    bwr = -6
    ref = pow (10.0, bwr / 20.0)
    a = -(np.pi * fc * bw) ** 2 / (4.0 * np.log (ref))
    time_sigma = 1 / np.sqrt (2 * a)
    time = np.arange (-num_sigmas * time_sigma,
                      num_sigmas * time_sigma, 1 / fs)

    if plot:
      # Plot the excitation signal
      excitation, envelope = gausspulse(
          time, fc=fc, bw=bw, bwr=bwr, retenv=True)
      plt.plot(envelope, 'g--', excitation)
      plt.title ('Excitation Signal', fontsize=20, pad=20)
      plt.xlabel (r'Time [$\mu$s]', fontsize=14, labelpad=10)
      plt.gca ().set_facecolor ('white')
      xticks = plt.xticks ()[0]
      plt.gca ().set_xticklabels ([f'{i*1e6/fs:.2f}' for i in xticks])
      plt.grid (False, axis='y')
      plt.grid (True, axis='x', linestyle=':', color='gray')
      plt.yticks ([])
      plt.show ()
    else:
      excitation = gausspulse(time, fc=fc, bw=bw, bwr=bwr)

    # Set the excitation signal
    self.excitation = self.new_tensor (excitation)

  @staticmethod
  def compute_time_samples (scan_depth, speed_of_sound, sampling_frequency):
    from math import ceil
    return ceil (2 * scan_depth / speed_of_sound * sampling_frequency)

  def load_field_ii_scatterers (self, filepath, verbose=True):
    '''Load scatterer data from a *.mat file from a Field II example.
    Save the data as `self.scatterer_x`, `self.scatterer_y`,
    `self.scatterer_z`, `self.scatterer_amplitude`.'''
    import scipy.io as spio
    from pathlib import Path

    filepath = Path (filepath)
    assert filepath.exists ()
    if verbose:
      print (f'Loading Field II scatterer data from "{filepath}"')
    mat = spio.loadmat (filepath)
    pos, amp = mat['phantom_positions'], mat['phantom_amplitudes']
    self.scatterer_x = self.new_tensor (pos[:, 0])
    self.scatterer_y = self.new_tensor (pos[:, 1])
    self.scatterer_z = self.new_tensor (pos[:, 2])
    self.scatterer_amplitude = self.new_tensor (amp.reshape (-1))
    self.num_scatterers = self.scatterer_x.nelement ()
    self.check_shape (
        ['scatterer_x', 'scatterer_y', 'scatterer_z', 'scatterer_amplitude'],
        [self.num_scatterers]
    )
    if verbose:
      print (
          f'Scatterer data loaded. Total # scatterers: {self.num_scatterers:,}')

  def select_by_type (self, *options):
    types = [torch.float32, torch.float64]
    for i, T in enumerate (types):
      if self.dtype is T:
        return options[i]
    raise TypeError (f'No matching type for: {self.dtype}')

  def to_struct (self):
    constructor = self.select_by_type (
        openbcsim.Simulator_float,
        openbcsim.Simulator_double
    )
    try:
      return constructor (
          sampling_frequency=self.sampling_frequency,
          decimation=self.decimation,
          scan_depth=self.scan_depth,
          speed_of_sound=self.speed_of_sound,
          attenuation=self.attenuation,
          tx=self.tx.to_struct (),
          rx=self.rx.to_struct (),
          num_time_samples=self.num_time_samples,
          scatterer_x=self.scatterer_x,
          scatterer_y=self.scatterer_y,
          scatterer_z=self.scatterer_z,
          scatterer_amplitude=self.scatterer_amplitude,
          num_scatterers=self.num_scatterers,
      )
    except Exception as e:
      if isinstance (e, AttributeError):
        raise RuntimeError ('Make sure to set the scatterer data first!\n'
                            '(Call `self.load_field_ii_scatterers()`.)'
                            )
      else:
        # Dump everything... (Look for a type error)
        print ({key: (type (getattr (self, key)), getattr (self, key)) for key in [
                'sampling_frequency', 'decimation', 'scan_depth', 'speed_of_sound',
                'attenuation', 'tx', 'rx', 'num_time_samples',
                'scatterer_x', 'scatterer_y', 'scatterer_z', 'scatterer_amplitude',
                'num_scatterers',
                ]})
        raise e

  def launch (self,
              scatterer_blocks_factor=32,
              rx_blocks=1,
              tx_blocks=1,
              convolve=True,
              demodulate=True,
              dry_run=False,
              silent=False,
              ):
    '''Launch CUDA kernel to simulate a frame.'''
    import time
    import os
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # To track the CUDA errors

    tic = time.clock ()
    try:
      self.grid = openbcsim.make_grid (
          scatterer_blocks_factor,
          rx_blocks,
          tx_blocks,
      )
      if not dry_run:
        self.result = openbcsim.launch (
            self.to_struct (),
            grid=self.grid,
        )
        if convolve:
          self.result = self.convolve (self.result, demodulate=demodulate)
        torch.cuda.synchronize ()
        [x for x in self.result if False]  # Read to trigger cudaMemcpy
    except Exception as e:
      msg = 'See CUDA documentation for list of error codes:\n'
      msg += 'https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g3f51e3575c2178246db0a94a430e0038'
      print (msg)
      raise e
    else:
      self.toc = time.clock () - tic
      props = openbcsim.DeviceProperties ()
      memoryClockRate = props.memoryClockRate * 1000  # convert from kHz to Hz
      memoryBusWidth = props.memoryBusWidth / 8  # bits to bytes
      sizeof_elem = 4 if (self.dtype is torch.float32) else 8  # bytes per element in `self.result`
      double_data_rate = 2
      num_transfers = 2
      self.perf = {
          'seconds': self.toc,
          'Theoretical memory bandwidth': memoryClockRate * memoryBusWidth * double_data_rate / 1e9,
          'Actual memory bandwidth': self.result.numel () * sizeof_elem * num_transfers / 1e9 / self.toc,
          'Threads per second': self.get_stats (as_dict=True)['CUDA thread count'] / self.toc,
      }
      if not silent:
        msg = '''\
{seconds:.2f} seconds
Theoretical memory bandwidth: {Theoretical memory bandwidth:.1f} GB/s
Actual memory bandwidth: {Actual memory bandwidth:.3f} GB/s
Threads per second: {Threads per second:.1f}\
'''.format (**self.perf)
        print (msg)

    return self.result

  def convolve (self, projected_data, demodulate=True):
    '''Apply excitation to projected time points via convolution (done
    in frequency domain).'''

    N = self.num_time_samples

    kernel = self.new_zeros ([N, 2])
    n = self.excitation.numel ()
    kernel[:n, 0] = self.excitation
    kernel = torch.fft (kernel, 1)

    projected_data = torch.fft (projected_data, 1)
    output = projected_data * kernel / N
    if demodulate:
      output = self.analytic_signal (output)

    output = torch.ifft (output, 1)
    return output

  @staticmethod
  def analytic_signal (spectrum):
    '''Returns `X[k] + i * H{X}[k]` where
    - `X[k]` is the frequency spectrum of the signal,
    - `i` is `sqrt(-1)`,
    - `H{X}[k]` is the frequency spectrum of the Hilbert transform of
    the signal.

    Effectively:
    - Positive frequency components (`i=1...N/2-1` of `i=0...N-1`, for N
    even) are doubled.
    - Negative frequency components (`i=N/2+1...N-1`, for N even) are
    zeroed.

    Operates on the second to last dimension of `spectrum`. Assumes last
    dimension (of length 2) is for the complex and imaginary components.

    Examples
    ===
    N=5
    component: 0 1 2 3 4
    "sign":    0 + + - -

    N=4
    component: 0 1 2 3
    "sign":    0 + 0 -

    N=3
    component: 0 1 2
    "sign":    0 + -
    '''

    N = spectrum.shape[-2]

    # Round odd N up
    positive = slice (1, (N + 1) // 2)  # ex: N=3 -> [1:2], N=4 -> [1:2], N=5 -> [1:3]
    # Round odd N down
    negative = slice (N // 2 + 1, N)    # ex: N=3 -> [2:3], N=4 -> [3:4]. N=5 -> [3:5]

    spectrum[..., positive, :] *= 2
    spectrum[..., negative, :] = 0

    return spectrum

  def get_stats (self,
                 scatterer_blocks_factor=32,
                 rx_blocks=1,
                 tx_blocks=1,
                 as_dict=False,
                 ):
    '''Launch stats.'''
    from functools import reduce

    shape = openbcsim.make_shape (self.to_struct ())
    if not hasattr (self, 'grid'):
      self.grid = openbcsim.make_grid (
          scatterer_blocks_factor,
          rx_blocks,
          tx_blocks,
      )
    elem_size = self.select_by_type (4, 8)
    memory = reduce (lambda a, b: a * b, shape) * elem_size / 1e6
    num_blocks = reduce (lambda a, b: a * b, self.grid)
    maxThreadsPerBlock = openbcsim.DeviceProperties ().maxThreadsPerBlock
    self.stats = {
        'Output buffer shape': shape,
        'Output buffer memory': memory,
        'Time samples': self.num_time_samples,
        'Transmitter subelements': self.tx.num_subelements,
        'Receiver subelements': self.rx.num_subelements,
        'Scatterer samples': self.num_scatterers,
        'CUDA grid blocks': self.grid,
        'CUDA thread count': maxThreadsPerBlock * num_blocks,
    }

    if not as_dict:
      msg = '''\
Output buffer shape        {Output buffer shape!s:<20} = \
[focal pts.] x [rx elems.] x [time pts.] x [real|imag]
Output buffer memory       {Output buffer memory:.1f} MB
Time samples               {Time samples:<20,}
Transmitter subelements    {Transmitter subelements:<20,}
Receiver subelements       {Receiver subelements:<20,}
Scatterer samples          {Scatterer samples:<20,}
CUDA grid blocks           {CUDA grid blocks!s:<20}
CUDA thread count          {CUDA thread count:<20,} = \
[{maxThreadsPerBlock} threads/block] x [Total grid blocks]
'''.format (**self.stats, maxThreadsPerBlock=maxThreadsPerBlock)
      print (msg)
    else:
      return self.stats

  @staticmethod
  def reset_device ():
    '''Doesn't work :(. (Trying to recover from error without restarting
    notebook.'''
    openbcsim.reset_device ()
    torch.cuda.empty_cache ()
    import importlib
    importlib.reload (torch)
    torch.cuda.init ()

  def __repr__ (self):
    '''Returns __repr__ of arguments needed to construct self.'''
    import inspect
    cls = type (self)
    constructor_args = inspect.signature (cls).parameters.keys ()
    message = cls.__name__ + ' ({})'
    # Note: `{{` escapes `{` and `}}` escapes `}`. (For reference: pyformat.info)
    # `!r` means convert with `repr(...)`.
    template = ', '.join (f'{arg}={{{arg}!r}}' for arg in constructor_args)

    return message.format (template).format (**vars (self))

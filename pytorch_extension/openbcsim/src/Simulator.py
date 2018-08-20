import torch
import openbcsim  # Must come after `import torch`
import Transducer


class Simulator:
  '''Data and parameters needed to run a simulation.'''

  def __init__ (self, sampling_frequency=100e6, decimation=10,
                scan_depth=9e-2, speed_of_sound=1540, attenuation=.7,
                transmitter=Transducer.Transducer (), receiver=None):
    '''The value of `sampling_frequency / decimation` tells how many
    samples to record per a second of simulation (Hz).

    `center_frequency` (Hz) describes the carrier wave frequency of the `transmitter`
    transducer.
    `attenuation` is the decrease in amplitude of the signal over distance travelled in dB/MHz/cm.
    `scan_depth` is how deep into the tissue (m) to simulate.
    `speed_of_sound` is the uniform speed assumed to be constant throughout the tissue (m/s).
    `transmitter` and `receiver` are `Transducer` instances (possibly the same instance) that
    describe the element geometry and delay/apodization settings of the transducer(s) used to
    transmit and receive the ultrasound signal.

    `receiver` defaults to the same instance as `transmitter` if not given.

    The `tensor_type` attribute (specifying the dtype and device of all tensors owned) of the
    transmitter and receiver must be the same.
    '''

    # Store parameters given.
    self.sampling_frequency = sampling_frequency
    self.decimation = decimation
    self.scan_depth = scan_depth
    self.speed_of_sound = speed_of_sound
    self.attenuation = attenuation

    # Save transducers.
    self.transmitter = transmitter
    self.receiver = (receiver) if (receiver is not None) else (transmitter)

    # Check that the transducers have the same `tensor_type`.
    if not self.transmitter.has_same_tensor_type (self.receiver):
      msg = 'Transmitter and receiver `Transducer` instances must have the ' \
          'same tensor_type (i.e. dtype and device).'
      msg += f'\nTransmitter tensor_type:  {self.transmitter.tensor_type}'
      msg += f'\nReceiver tensor_type:     {self.receiver.tensor_type}'
      raise ValueError (msg)
    else:
      self.dtype = self.transmitter.dtype
      self.device = self.transmitter.device
      self.tensor_type = self.transmitter.tensor_type

    # Compute number of time samples needed.
    self.num_time_samples = self.compute_time_samples (self.scan_depth,
                                                       self.speed_of_sound,
                                                       self.sampling_frequency)

    # Bind these tensor constructors for convenience.
    self.new_tensor = transmitter.new_tensor
    self.new_ones = transmitter.new_ones
    self.new_zeros = transmitter.new_zeros

  def check_shape (self, attributes, true_shape):
    '''Check that the list of `attributes` (given by their string names) have the desired `true_shape`.'''
    true_shape = tuple (true_shape)
    for attr in attributes:
      value = getattr (self, attr)
      if value is None:
        continue
      shape = tuple (value.shape)
      if shape != true_shape:
        msg = f'Shape of self.{attr} {shape} does not match desired true shape {true_shape}.'
        raise ValueError (msg)

  def set_gaussian_excitation (self, bw=.5, bwr=-6, num_sigmas=3, plot=True):
    '''Define the ultrasound excitation signal.
    `bw` is the fractional bandwidth (between 0 and 1) of the Gaussian modulating
    signal relative to the transmitter center frequency.
    `bwr` is the reference level at which the fractional bandwidth is calculated (dB).
    (See scipy.signal.gausspulse.)
    `num_sigmas` is how many standard deviations outwards from the center to sample in either direction.'''

    from scipy.signal import gausspulse
    import matplotlib.pyplot as plt

    fc = self.transmitter.center_frequency
    fs = self.sampling_frequency

    # https://github.com/scipy/scipy/blob/14142ff70d84a6ce74044a27919c850e893648f7/scipy/signal/waveforms.py#L232
    # exp(-a t^2) <->  sqrt(pi/a) exp(-pi^2/a * f^2)  = g(f)
    bwr = -6
    pi = np.pi
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
    return int (2 * scan_depth / speed_of_sound * sampling_frequency + .5)

  def load_field_ii_scatterers (self, filepath, verbose=True):
    '''Load scatterer data from a *.mat file from a Field II example.
    Save the data as `self.scatterer_x`, `self.scatterer_y`, `self.scatterer_z`,
    `self.scatterer_amplitude`.'''
    import scipy.io as spio
    if verbose:
      print (f'Loading Field II scatterer data from {filepath}')
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

  def select_by_type (self, *functions):
    types = [torch.float32, torch.float64]
    for i, T in enumerate (types):
      if self.dtype is T:
        return functions[i]
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
          transmitter=self.transmitter.to_struct (),
          receiver=self.receiver.to_struct (),
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
                'attenuation', 'transmitter', 'receiver', 'num_time_samples',
                'scatterer_x', 'scatterer_y', 'scatterer_z', 'scatterer_amplitude',
                'num_scatterers',
                ]})
        raise e

  def launch (self, silent=False):
    '''Launch CUDA kernel to simulate a frame.'''
    import time
    import os
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # To track the CUDA errors

    tic = time.clock ()
    simulate = self.select_by_type (
        openbcsim.launch_float, openbcsim.launch_double)
    try:
      result = simulate (self.to_struct ())
      torch.cuda.synchronize ()
      [x for x in result if False]  # Read to trigger cudaMemcpy
    except Exception as e:
      msg = 'See CUDA documentation for list of error codes:\n'
      msg += 'https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g3f51e3575c2178246db0a94a430e0038'
      print (msg)
      raise e
    else:
      self.toc = time.clock () - tic
      self.perf = {
          'seconds': self.toc,
          'Theoretical memory bandwidth': 4004e6 * (192 / 8) * 2 / 1e9,
          'Actual memory bandwidth': result.numel () * 4 * (1 + self.dtype is torch.float64) * 2 / toc,
          'Threads per second': stats (silent=True)['CUDA threads'] / toc,
      }
      if not silent:
        msg = '''\
{seconds:.1f} seconds
Theoretical memory bandwidth: {Theoretical memory bandwidth:.1f}
Actual memory bandwidth: {Actual memory bandwidth:.1f}
Threads per second: {Threads per second:.1f}\
'''.format (**self.perf)
        print (msg)

    return result

  def stats (self, silent=False):
    '''Launch stats.'''
    from math import ceil
    self.stats = {
        'Output buffer elements': 2 * self.num_time_samples * self.receiver.num_subelements,
        'Time samples': self.num_time_samples,
        'Transmitter subelements': self.transmitter.num_subelements,
        'Receiver subelements': self.receiver.num_subelements,
        'Scatterer samples': self.num_scatterers,
        'CUDA threads': 1024 * ceil (self.num_scatterers / 1024) * self.transmitter.num_subelements * self.receiver.num_subelements,
        'CUDA blocks': self.num_scatterers * self.transmitter.num_subelements * self.receiver.num_subelements // 1024,
    }

    if not silent:
      msg = '''\
Output buffer elements     {Output buffer elements:<15,} = \
2 x [Time samples] x [Receiver subelements] \
-- `2 x` because data is complex
Time samples               {Time samples:<15,}
Transmitter subelements    {Transmitter subelements:<15,}
Receiver subelements       {Receiver subelements:<15,}
Scatterer samples          {Scatterer samples:<15,}
CUDA threads               {CUDA threads:<15,} = \
1024 * ceil([Scatterer samples] / 1024) x [Transmitter subelements] x [Receiver subelements]
CUDA blocks                {CUDA blocks:<15,} = [CUDA threads] / 1024\
'''.format (**self.stats)
      print (msg)

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
    template = ', '.join (f'{arg}={{{arg}!r}}' for arg in constructor_args)

    return message.format (template).format (**self.__dict__)

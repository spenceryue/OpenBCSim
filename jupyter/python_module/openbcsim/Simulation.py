import torch
import openbcsim # Must come after `import torch`
import Transducer

class Simulation:
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
    if not self.transmitter.same_tensor_type (self.receiver):
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
    time_sigma = 1/np.sqrt (2*a)
    time = np.arange (-num_sigmas*time_sigma, num_sigmas*time_sigma, 1/fs)

    if plot:
      # Plot the excitation signal
      excitation, envelope  = gausspulse(time, fc=fc, bw=bw, bwr=bwr, retenv=True)
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
      excitation  = gausspulse(time, fc=fc, bw=bw, bwr=bwr)

    # Set the excitation signal
    self.excitation = self.new_tensor (excitation)

  @staticmethod
  def compute_time_samples (scan_depth, speed_of_sound, sampling_frequency):
    return int (2 * scan_depth / speed_of_sound * sampling_frequency + .5);

  def load_field_ii_scatterers (self, filepath, verbose=True):
    '''Load scatterer data from a *.mat file from a Field II example.
    Save the data as `self.scatterer_x`, `self.scatterer_y`, `self.scatterer_z`,
    `self.scatterer_amplitude`.'''
    import scipy.io as spio
    if verbose:
      print (f'Loading Field II scatterer data from {filepath}')
    mat = spio.loadmat (filepath)
    pos, amp = mat['phantom_positions'], mat['phantom_amplitudes']
    self.scatterer_x = self.new_tensor (pos[:,0])
    self.scatterer_y = self.new_tensor (pos[:,1])
    self.scatterer_z = self.new_tensor (pos[:,2])
    self.scatterer_amplitude = self.new_tensor (amp.reshape (-1))
    self.num_scatterers = self.scatterer_x.nelement ()
    self.check_shape (
        ['scatterer_x', 'scatterer_y', 'scatterer_z', 'scatterer_amplitude'],
        [self.num_scatterers]
      )
    if verbose:
      print (f'Scatterer data loaded. Total # scatterers: {self.num_scatterers:,}')


  def select_by_type (self, *functions):
    types = [torch.float32, torch.float64];
    for i,T in enumerate (types):
      if self.dtype is T:
        return functions[i]
    raise TypeError (f'No matching type for: {self.dtype}')

  def to_struct (self):
    constructor = self.select_by_type (
        openbcsim.Simulation_float,
        openbcsim.Simulation_double
      )
    try:
      return constructor (
          sampling_frequency = self.sampling_frequency,
          decimation = self.decimation,
          scan_depth = self.scan_depth,
          speed_of_sound = self.speed_of_sound,
          attenuation = self.attenuation,
          transmitter = self.transmitter.to_struct (),
          receiver = self.receiver.to_struct (),
          num_time_samples = self.num_time_samples,
          scatterer_x = self.scatterer_x,
          scatterer_y = self.scatterer_y,
          scatterer_z = self.scatterer_z,
          scatterer_amplitude = self.scatterer_amplitude,
          num_scatterers = self.num_scatterers,
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

  def launch (self, verbose=True):
    import time
    tic = time.clock ()
    simulate = self.select_by_type (openbcsim.launch_float, openbcsim.launch_double)
    result = simulate (self.to_struct ())
    if verbose:
      print (f'({time.clock () - tic:.1f} seconds)')
    return result

  def __repr__ (self):
    '''Returns __repr__ of arguments needed to construct self.'''
    import inspect
    cls = type (self)
    constructor_args = inspect.signature (cls).parameters.keys ()
    message = cls.__name__ + ' ({})'
    # Note: `{{` escapes `{` and `}}` escapes `}`. (For reference: pyformat.info)
    template = ', '.join (f'{arg}={{{arg}!r}}' for arg in constructor_args)

    return message.format (template).format (**self.__dict__)

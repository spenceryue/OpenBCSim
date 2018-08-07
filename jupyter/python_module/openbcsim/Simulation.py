import torch

class Simulation:
  '''Data and parameters needed to run a simulation.'''

  def __init__ (self, sampling_frequency=100e6, decimation=10,
                scan_depth=9e-2, speed_of_sound=1540, attenuation=.7,
                transmitter=Transducer (), receiver=None):
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

    # Compute number of time samples needed.
    self.num_time_samples = self.compute_time_samples (self.scan_depth,
                                                       self.speed_of_sound,
                                                       self.sampling_frequency)

    # Save transducers.
    self.transmitter = transmitter
    self.receiver = (receiver) if (receiver is not None) else (transmitter)

    # Check that the transducers have the same `tensor_type`.
    if not self.transmitter.same_tensor_type (self.receiver):
      msg = 'Transmitter and receiver `Transducer` instances must have the ' \
          'same tensor_type (i.e. dtype and device).'
      msg += '\nTransmitter tensor_type:  {}'.format (self.transmitter.tensor_type)
      msg += '\nReceiver tensor_type:   {}'.format (self.receiver.tensor_type)
      raise ValueError (msg)

    # Bind these tensor constructors for convenience.
    self.new_tensor = transmitter.new_tensor
    self.new_ones = transmitter.new_ones
    self.new_zeros = transmitter.new_zeros

  def load_field_ii_scatterers (filepath):
    '''Load scatterer data from a *.mat file from a Field II example.
    Save the data as `self.scatterer_x`, `self.scatterer_y`, `self.scatterer_z`,
    `self.scatterer_amplitude`.'''
    import scipy.io as spio
    mat = spio.loadmat (filepath)
    pos, amp = mat['phantom_positions'], mat['phantom_amplitudes']
    self.scatter_x = self.new_tensor (pos[:,0])
    self.scatter_y = self.new_tensor (pos[:,1])
    self.scatter_z = self.new_tensor (pos[:,2])
    self.scatter_amplitude = self.new_tensor (amp.reshape (-1))
    self.num_scatterers = self.scatter_x.nelement ()

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
      plt.gca ().set_xticklabels (['{:.2f}'.format (i*1e6/fs) for i in xticks])
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

  def run_simulation (self):
    '''
    import projection_kernel

    projection_kernel.run ()
    '''

  def __repr__ (self):
    '''Returns __repr__ of arguments needed to construct self.'''
    import inspect
    cls = type (self)
    constructor_args = inspect.signature (cls).parameters.keys ()
    message = cls.__name__ + ' ({})'
    # Note: `{{` escapes `{` and `}}` escapes `}`. (For reference: pyformat.info)
    template = ', '.join ('{0}={{{0}!r}}'.format (arg) for arg in constructor_args)

    return message.format (template).format (**self.__dict__)

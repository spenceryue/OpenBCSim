import torch
import numpy as np

class Transducer:
  '''Base class for describing a transducer.'''

  def __init__ (self, num_elements=1, num_subelements=1,
                x=None, y=None, z=None,
                delay=None, apodization=None,
                center_frequency=3.5e6,
                dtype=torch.float32, device='cuda'):
    '''Use one of the subclasses to fill in the arguments of this constructor
    with calculated positions for a particular aperture geometry more easily.

    Ensures all tensors owned by the transducer have the same dtype and device,
    recorded in `self.tensor_type`.

    `num_elements` is the number of physical transducer elements.
    `num_subelements` is the number of logical elements used in the simulation
    (for better accuracy).
    `x`, `y`, `z` are the arrays of coordinate positions of each subelement given
    in meters (m).
    `delay` is an array of per-element delays (seconds).
    `apodization` is an array of per-element apodization factors (i.e. amplitude
    scaling factor applied to the signal transmitted/received at a given element).
    `center_frequency` is the frequency of the carrier wave used to transmit or
    demodulate the ultrasound signal for the transducer (Hz).
    `dtype` is the torch data type for all tensors owned by this Transducer.
    `device` is the string identifier passed to torch.device () to determine the
    device all tensors owned by this Transducer will be placed on.
    '''

    # Length of arrays
    self.num_elements = num_elements
    self.num_subelements = num_subelements
    self.num_subdivisions = num_subelements // num_elements

    # Center frequency
    self.center_frequency = center_frequency

    # Device and dtype
    self.dtype = dtype
    self.device = device
    self.tensor_type = torch.empty (0, dtype=self.dtype, device=self.device)

    # Positions of transducer subelements
    self.x = self.new_tensor (x)
    self.y = self.new_tensor (y)
    self.z = self.new_tensor (z)

    # Per-element delay and apodization
    self.delay = self.new_tensor (delay)
    self.apodization = self.new_tensor (apodization)

    self.check_shape (['x', 'y', 'z'], [num_subelements])
    self.check_shape (['delay', 'apodization'], [num_elements])

  def new_tensor (self, data, **kwargs):
    '''Call `torch.Tensor.new_tensor` with `self.tensor_type` as the reference tensor,
    or return `None` if `data` is `None`.'''
    if data is None:
      return None
    else:
      return self.tensor_type.new_tensor (data, **kwargs)

  def new_ones (self, size, **kwargs):
    '''Shortcut to calling `self.tensor_type.new_ones (...)`'''
    return self.tensor_type.new_ones (size, **kwargs)

  def new_zeros (self, size, **kwargs):
    '''Shortcut to calling `self.tensor_type.new_zeros (...)`'''
    return self.tensor_type.new_zeros (size, **kwargs)

  def check_shape (self, attributes, true_shape):
    '''Check that the list of `attributes` (given by their string names) have the desired `true_shape`.'''
    true_shape = tuple (true_shape)
    for attr in attributes:
      value = getattr (self, attr)
      if value is None:
        continue
      shape = tuple (value.shape)
      if shape != true_shape:
        msg = 'Shape of self.{} {} does not match desired true shape {}.'.format (attr, shape, true_shape)
        raise ValueError (msg)

  def same_tensor_type (self, other):
    return (self.tensor_type.dtype == other.tensor_type.dtype) and \
         (self.tensor_type.device == other.tensor_type.device)

  def __repr__ (self):
    '''Returns __repr__ of arguments needed to construct self.'''
    import inspect
    cls = type (self)
    constructor_args = inspect.signature (cls).parameters.keys ()
    message = cls.__name__ + ' ({})'
    # Note: `{{` escapes `{` and `}}` escapes `}`. (For reference: pyformat.info)
    template = ', '.join ('{0}={{{0}!r}}'.format (arg) for arg in constructor_args)

    return message.format (template).format (**self.__dict__)


class LinearTransducer (Transducer):
  '''A linear transducer.'''

  def __init__ (self, num_elements=1, width=.44e-3, height=5e-3, kerf=0, num_sub_x=1, num_sub_y=1,
                center_frequency=3.5e6,
                dtype=torch.float32, device='cuda'):
    '''Initializes an `Transducer` with a linear geometry spanning lengthwise the x-direction.
    Also initializes per-element delays and apodizations to 0s and 1s respectively.'''

    # Calculate sub-element positions
    x, y, z = self.linear_array (num_elements, width, height, kerf, num_sub_x, num_sub_y)

    # Initialize `Transducer`
    num_subelements = num_elements * num_sub_x * num_sub_y
    super ().__init__ (
      num_elements = num_elements,
      num_subelements = num_subelements,
      x = x.reshape (-1),
      y = y.reshape (-1),
      z = z.reshape (-1),
      center_frequency = center_frequency,
      dtype = dtype,
      device = device
    )

    # Default to delay=0, apodization=1 for all elements
    self.delay = self.new_zeros (num_elements),
    self.apodization = self.new_ones (num_elements)

    # Record number of subelement divisions
    self.num_sub_x = num_sub_x
    self.num_sub_y = num_sub_y

    # Keep element geometry
    self.width = width
    self.height = height
    self.kerf = kerf

  @classmethod
  def linear_array (cls, num_elements, width, height, kerf, num_sub_x=1, num_sub_y=1):
    '''Calculates the positions of elements in a linear array aperture (i.e. transducer).
    Arguments and functionality is very similar to xdc_linear_array() in Field II.'''

    # Subelement division offsets
    offset_x = cls.subdivide (width, num_sub_x)
    offset_y = cls.subdivide (height, num_sub_y)

    # Tile the subelement offsets
    tiled_offset_x = np.tile (offset_x, [num_sub_y, num_elements])
    tiled_offset_y = np.tile (offset_y.reshape(-1, 1), [1, tiled_offset_x.shape[1]])

    # Center-to-center distance between elements
    pitch = width + kerf

    # Tile the element positions
    position_x = np.arange ((-num_elements//2 + 1) * pitch, (num_elements//2 + 1) * pitch, pitch)
    position_x = position_x.reshape (1, -1)
    position_x = position_x.repeat (num_sub_x, axis=1)
    position_x = position_x.repeat (num_sub_y, axis=0)

    # Sum positions and offsets
    x = position_x + tiled_offset_x
    y = tiled_offset_y
    z = np.zeros (x.shape)

    return x, y, z

  @staticmethod
  def subdivide (length, num_divisions):
    '''Calculate a range of `num_divisions` positions centered around zero that have the given total `length`.'''
    step = length / num_divisions        # ex: step = 1
    start = (-num_divisions//2 + 1) * step   # ex: start = -3//2 + 1 = (-2) + 1 = -1
    stop = (num_divisions//2 + 1) * step     # ex: stop = 3//2 + 1 = (1) + 1 = 2 (but stop is excluded)
    divisions = np.arange (start, stop, step)  # ex: divisions = [-1, 0, 1]

    return divisions

  def plot (self, true_scale=False, show=True):
    '''Plot the aperture elements in 3D.'''
    from matplotlib import pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    # Size figure and make scatter plot.
    fig = plt.figure (figsize=(12,8))
    ax = fig.gca (projection='3d')
    ax.scatter (self.x, self.y, self.z, c=np.arange (self.num_subelements), s=5, cmap='plasma')

    if self.num_elements > 1:
      # Label first and last transducer element ('Start' and 'Stop').
      ax.text (self.x[0], self.y[0], self.z[0], 'o Start', fontsize=14, weight='bold', color='red')
      if true_scale:
        # Scale axes to true proportions (if `true_scale` set).
        limits = np.array ([getattr (ax, 'get_{}lim'.format(axis)) () for axis in 'xyz'])
        ax.auto_scale_xyz(*[(limits.min (), limits.max ())]*3)
      else:
        # Label transducer elements.
        stride = self.num_elements * self.num_sub_x // 6
        stride = max (stride, 1)
        for i in range (stride, self.num_subelements, stride):
          ax.text (self.x[i], self.y[i], self.z[i], 'o {}'.format (i), fontsize=12, color='red')
        # Warn that axes are not to scale.
        ax.text2D (1, 0, '(Axes not plotted to scale.)', transform=ax.transAxes,
               ha='right', fontsize=12, color='red')
      ax.text (self.x[-1], self.y[-1], self.z[-1], 'o Stop', fontsize=14, weight='bold', color='red')
    else:
      # Special case when there's only one element
      ax.scatter (self.x, self.y, self.z, c='red', s=50)

    # Label axes and convert tick values to millimeters.
    ax.set_xlabel ('x [mm]', color='red', fontsize=16, labelpad=10)
    xticks = ax.get_xticks ()
    ax.set_xticklabels (['{:.1f}'.format (e*1000) for e in xticks])
    ax.set_ylabel ('y [mm]', color='red', fontsize=16, labelpad=10)
    yticks = ax.get_yticks ()
    ax.set_yticklabels (['{:.1f}'.format (e*1000) for e in yticks])
    ax.set_zlabel ('z [mm]', color='red', fontsize=16, labelpad=10)
    zticks = ax.get_zticks ()
    ax.set_zticklabels (['{:.1f}'.format (e*1000) for e in zticks])

    # Give the plot a title
    ax.set_title ('Transducer Element Positions', color='red', fontsize=22, pad=50)

    # Set the background color and view angle
    ax.set_facecolor ('white')
    ax.view_init (elev=45, azim=225)

    if show:
      plt.show()
    else:
      return fig, ax
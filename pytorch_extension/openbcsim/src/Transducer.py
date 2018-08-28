import torch
import openbcsim  # Must come after `import torch`
import numpy as np
from mpl_utils import rcParams


class Transducer:
  '''Base class for describing a transducer.'''

  def __init__ (self, num_elements=1, num_subelements=1,
                x=None, y=None, z=None,
                delays=None, apodization=None,
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
    `delays` is an array of per-element delays (seconds).
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
    self.subdivision_factor = num_subelements // num_elements
    # self.num_focal_points defined through @property

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

    # Per-element delays and apodization
    self.delays = self.new_tensor (delays)
    self.apodization = self.new_tensor (apodization)

    # Fill with a default value if None
    if self.x is None:
      self.x = self.new_zeros (num_subelements)
    if self.y is None:
      self.y = self.new_zeros (num_subelements)
    if self.z is None:
      self.z = self.new_zeros (num_subelements)
    if self.delays is None:
      self.delays = self.new_zeros (num_elements)
    if self.apodization is None:
      self.apodization = self.new_ones (num_elements)

    self.check_shape (['x', 'y', 'z'], [num_subelements])
    self.check_shape (['delays', 'apodization'], [num_elements])

  @property
  def num_focal_points(self):
    try:
      return len (self.focal_points)
    except AttributeError:
      return 1

  @num_focal_points.setter
  def num_focal_points(self, _):
    msg = 'Don\'t try to set the number of scans through this property.\n'
    msg += 'The number of scans is implicitly determined from the first ' \
           'dimension of `self.delays` (or equivalently of self.apodization).'
    raise AttributeError (msg)

  def has_same_tensor_type (self, other):
    return (self.tensor_type.dtype == other.tensor_type.dtype) and \
        (self.tensor_type.device == other.tensor_type.device)

  def is_compatible (self, tensor):
    try:
      return (self.dtype is tensor.dtype) and (self.device == tensor.device)
    except (AttributeError, TypeError):  # i.e. `tensor` is not an instance of `torch.Tensor`
      return False

  def new_tensor (self, data, **kwargs):
    '''Call `torch.Tensor.new_tensor` with `self.tensor_type` as the
    reference tensor if `data` is not `None`.
    If the data is already a tensor compatible with tensor_type then do
    not make a new copy.'''
    if data is not None:
      if self.is_compatible (data):
        return data
      else:
        return self.tensor_type.new_tensor (data, **kwargs)

  def new_ones (self, size, **kwargs):
    '''Shortcut to calling `self.tensor_type.new_ones (...)`.'''
    return self.tensor_type.new_ones (size, **kwargs)

  def new_zeros (self, size, **kwargs):
    '''Shortcut to calling `self.tensor_type.new_zeros (...)`.'''
    return self.tensor_type.new_zeros (size, **kwargs)

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

  def select_by_type (self, *functions):
    types = [torch.float32, torch.float64]
    for i, dtype in enumerate (types):
      if self.dtype is dtype:
        return functions[i]
    raise TypeError (f'No matching type for: {self.dtype}')

  def to_struct (self):
    constructor = self.select_by_type (
        openbcsim.Transducer_float,
        openbcsim.Transducer_double
    )
    try:
      return constructor (
          num_elements=self.num_elements,
          num_subelements=self.num_subelements,
          subdivision_factor=self.subdivision_factor,
          num_focal_points=self.num_focal_points,
          x=self.x,
          y=self.y,
          z=self.z,
          delays=self.delays,
          apodization=self.apodization,
          center_frequency=self.center_frequency,
      )
    except Exception as e:
      # Dump everything... (Look for a type error)
      print ({key: (type (getattr (self, key)), getattr (self, key)) for key in [
              'num_elements', 'num_subelements', 'subdivision_factor', 'num_focal_points',
              'x', 'y', 'z', 'delays', 'apodization', 'center_frequency'
              ]})
      raise e

  def __repr__ (self):
    '''Returns __repr__ of arguments needed to construct self.'''
    import inspect
    cls = type (self)
    constructor_args = inspect.signature (cls).parameters.keys ()
    message = cls.__name__ + ' ({})'
    # Note: `{{` escapes `{` and `}}` escapes `}`. (For reference: pyformat.info)
    # `!r` means convert with `repr(...)`.
    template = ', '.join (f'{arg}={{{arg}!r}}' for arg in constructor_args)

    return message.format (template).format (**self.__dict__)

  def set_delays (self, *, focal_points=None, map_func=None, speed_of_sound=1540):
    '''Calculates the delays needed for each focal point.
    Or specipy your own `map_func`. (See `map_focal_points()` for details.)
    Give a set of focal points to scan for. The total number of focal
    points determines the value of `num_focal_points`.
    Each focal point should be a triple (x,y,z) in the same coordinate
    system as the Transducer elements:
      x is lateral -- corresponds to width-direction of image
      y is elevational -- perpendicular to the imaging plane
      z is depthwise -- height-direction of image plane
    `focal_points` defaults to the last seen set of focal points if not
    set. (See `map_focal_points()`.)
    `speed_of_sound` is in meters/sec.
    This function accepts keyword arguments only.
    A negative delay is equivalent to an "advance" in time.
    '''

    if map_func is None:
      def map_func (X, Y, Z, focus):
        distance = ((X - focus[0])**2 + (Y - focus[1])**2 + (Z - focus[2])**2)**.5
        # Note: This returns non-positive delays, which corresponds to "advances" in time.
        delays = -(distance - distance.min ()) / speed_of_sound
        return delays

    self.delays = self.map_focal_points (map_func, focal_points)

    return self.delays

  def set_apodization (self, *, focal_points=None, map_func=None):
    '''Passing no arguments will yield uniform apodization matching the shape of
    `self.delays`.
    This function accepts keyword arguments only.
    '''
    if map_func is None:
      # Uniform apodization, match shape of self.delays
      self.apodization = self.new_ones (self.delays.shape)
    else:
      self.apodization = self.map_focal_points (map_func, focal_points)

    return self.apodization

  def map_focal_points (self, map_func, focal_points=None):
    '''Applies `map_func` to arguments:
        X = array of transducer element position x-coordinates
        Y = array of transducer element position y-coordinates
        Z = array of transducer element position z-coordinates
        focus = position of a focal point: (x,y,z)
    Equivalent to the following:
        result[scan_idx] = map_func (X, Y, Z, focal_points[scan_idx])
    `map_func` should output a Tensor with shape (len (X),).
    `focal_points` defaults to the last seen set of focal points if
    `None`.
    `result` returned is a Tensor (or numpy array) with shape:
      (len (focal_points), self.num_elements).

    Essentially, this is a helper function that indexes into transducer
    *element* positions instead of *subelement* positions (which would
    be what one gets if one loops over the positions without a stride).
    '''

    if focal_points is not None:
      self.focal_points = focal_points

    II = slice (self.subdivision_factor // 2, self.num_subelements, self.subdivision_factor)
    result = [*map (lambda point:
                    self.new_tensor (
                        map_func (self.x[II], self.y[II], self.z[II], point).reshape (1, -1)
                    ),
                    self.focal_points
                    )]
    result = torch.cat (result, dim=0)

    return result

  @rcParams ({'font.family': 'calibri'})
  def plot_delays (self, show=True, stride=1):
    from matplotlib import pyplot as plt
    lines = plt.plot (np.arange (self.delays.shape[1]), self.delays.cpu ().numpy ()[::stride].T)

    # Formatting
    plt.title ('Delays', fontsize=20, pad=10)
    plt.xlabel ('Element index', fontsize=14, labelpad=10)
    plt.xticks (color='gray', fontsize=12)
    plt.yticks (color='gray', fontsize=12)
    plt.ylabel ('Time (sec)', fontsize=14, labelpad=10)
    plt.grid (axis='y', color='gray', ls=':')
    plt.gca ().set_facecolor ('white')

    if show:
      plt.show ()
    else:
      return lines

  @rcParams ({'font.family': 'calibri'})
  def plot_apodization (self, show=True, stride=1):
    from matplotlib import pyplot as plt
    lines = plt.plot (np.arange (
        self.apodization.shape[1]), self.apodization.cpu ().numpy ()[::stride].T)

    # Formatting
    plt.title ('Apodization', fontsize=20, pad=10)
    plt.xlabel ('Element index', fontsize=14, labelpad=10)
    plt.xticks (color='gray', fontsize=12)
    plt.yticks (color='gray', fontsize=12)
    plt.ylabel ('Scale Factor', fontsize=14, labelpad=10)
    plt.grid (axis='y', color='gray', ls=':')
    plt.gca ().set_facecolor ('white')

    if show:
      plt.show ()
    else:
      return lines

  def copy (self):
    from copy import copy
    return copy (self)


class LinearTransducer (Transducer):
  '''A linear transducer.'''

  def __init__ (self, num_elements=1, width=.44e-3, height=5e-3, kerf=0.05e-3,
                num_sub_x=1, num_sub_y=1, center_frequency=3.5e6,
                dtype=torch.float32, device='cuda'):
    '''Initializes an `Transducer` with a linear geometry spanning lengthwise the x-direction.
    Also initializes per-element delays and apodizations to 0s and 1s respectively.'''

    # Calculate sub-element positions
    x, y, z = self.linear_array (
        num_elements, width, height, kerf, num_sub_x, num_sub_y)

    # Initialize `Transducer`
    num_subelements = num_elements * num_sub_x * num_sub_y
    super ().__init__ (
        num_elements=num_elements,
        num_subelements=num_subelements,
        x=x.reshape (-1, order='F'),  # Read elements in Fortran-style, first axis changes fastest
        y=y.reshape (-1, order='F'),  # Read elements in Fortran-style, first axis changes fastest
        z=z.reshape (-1, order='F'),  # Read elements in Fortran-style, first axis changes fastest
        center_frequency=center_frequency,
        dtype=dtype,
        device=device
    )

    # Default to delays=0, apodization=1 for all elements
    self.delays = self.new_zeros (num_elements)
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
    offset_x = cls.centered_range (num_sub_x, width / num_sub_x)
    offset_y = cls.centered_range (num_sub_y, height / num_sub_y)

    # Tile the subelement offsets
    tiled_offset_x = np.tile (offset_x, [num_sub_y, num_elements])
    tiled_offset_y = np.tile (offset_y.reshape(-1, 1),
                              [1, tiled_offset_x.shape[1]])

    # Center-to-center distance between elements
    pitch = width + kerf

    # Tile the element positions
    position_x = cls.centered_range (num_elements, pitch)
    position_x = position_x.reshape (1, -1)
    position_x = position_x.repeat (num_sub_x, axis=1)
    position_x = position_x.repeat (num_sub_y, axis=0)

    # Sum positions and offsets
    x = position_x + tiled_offset_x
    y = tiled_offset_y
    z = np.zeros (x.shape)

    return x, y, z

  @staticmethod
  def centered_range (count, step, center=0):
    '''Calculate a range of `count` positions centered around `center` spaced at
    `step` units apart.'''
    divisions = np.arange (count, dtype=float)
    divisions -= (count - 1) / 2  # Shift center to 0
    divisions *= step  # Adjust spacing
    divisions += center  # Re-center
    return divisions

  @rcParams ({'font.family': 'calibri'})
  def plot (self, true_scale=False, show=True, plot_focal_points=True):
    '''Plot the aperture elements in 3D.'''
    from matplotlib import pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.patheffects as PathEffects

    x, y, z = self.x, self.y, self.z

    if plot_focal_points and not hasattr (self, 'focal_points'):
      msg = '`self.focal_points` not set.'
      raise AttributeError (msg)

    # If plotting focal points, need to swap y/z axes and negate x axis.
    if plot_focal_points:
      x = -x
      y, z = z, y

    # Size figure and make scatter plot.
    fig = plt.figure (figsize=(12, 8))
    ax = fig.gca (projection='3d')
    ax.scatter (x, y, z, s=5, color='gray', alpha=.3)

    # Plot `self.focal_points`
    if plot_focal_points:
      f_x, f_y, f_z = zip (*self.focal_points)
      f_x = [-e for e in f_x]
      ax.scatter (f_x, f_z, f_y, c='red', s=25, marker='d', label='Focal Points')
      txt = ax.text (f_x[-1], f_z[-1], f_y[-1], '  Focus',
                     fontsize=14, weight='bold', color='red')
      txt.set_path_effects ([PathEffects.withStroke (linewidth=2, foreground='w')])

    if self.num_elements > 1:
      # Label first and last transducer element ('Start' and 'Stop').
      txt = ax.text (x[0], y[0], z[0], '  Start',
                     fontsize=14, weight='bold', color='dodgerblue')
      txt.set_path_effects ([PathEffects.withStroke (linewidth=2, foreground='w')])
      if true_scale:
        # Scale axes to true proportions (if `true_scale` set).
        limits = np.array (
            [getattr (ax, f'get_{axis}lim') () for axis in 'xyz'])
        ax.auto_scale_xyz(*[(limits.min (), limits.max ())] * 3)
      else:
        # Label transducer elements.
        stride = self.num_elements * self.num_sub_x // 5
        stride = max (stride, 1)

        # Make `x`, `y`, `z` back into the shape as returned by `linear_array()` using
        # `.reshape (...).transpose (...)`.
        # Then flatten in row-major (C-style) order, where the last dimension changes the fastest.
        # (Was originally flattened in Fortran-style order.)
        x, y, z = (e.reshape (-1, self.num_sub_y).transpose (0, 1).reshape (-1)
                   for e in [x, y, z])
        for i in range (stride, self.num_subelements - 1, stride):
          txt = ax.text (x[i], y[i], z[i], f'  {i}', fontsize=10, color='dodgerblue')
          txt.set_path_effects ([PathEffects.withStroke (linewidth=2, foreground='w')])

        # Highlight corresponding sampled elements in blue
        highlight = slice (stride, self.num_subelements - 1, stride)
        ax.scatter (x[0], y[0], z[0], label='Transducer Subelements',
                    s=50, color='dodgerblue', edgecolor='w', alpha=1, linewidth=1)
        ax.scatter (x[highlight], y[highlight], z[highlight],
                    s=50, color='dodgerblue', edgecolor='w', alpha=1, linewidth=1)
        ax.scatter (x[-1], y[-1], z[-1],
                    s=50, color='dodgerblue', edgecolor='w', alpha=1, linewidth=1)

        # Warn that axes are not to scale.
        ax.text2D (1, 0, '(Axes not plotted to scale.)', transform=ax.transAxes,
                   ha='right', fontsize=12, color='gray')
      txt = ax.text (x[-1], y[-1], z[-1], '  Stop',
                     fontsize=14, weight='bold', color='dodgerblue')
      txt.set_path_effects ([PathEffects.withStroke (linewidth=2, foreground='w')])
    else:
      # Special case when there's only one element
      ax.scatter (x, y, z, c='red', s=50)

    # Label axes and convert tick values to millimeters.
    ax.set_xlabel ('x [mm]', fontsize=16, labelpad=10)
    xticks = ax.get_xticks ()
    if plot_focal_points:
      xticks = [-e for e in xticks]
    ax.set_xticklabels ([f'{e*1000:.1f}' for e in xticks], fontsize=12, color='gray')
    ylabel, zlabel = 'y [mm]', 'z [mm]'
    if plot_focal_points:
      ylabel, zlabel = zlabel, ylabel
    ax.set_ylabel (ylabel, fontsize=16, labelpad=10)
    yticks = ax.get_yticks ()
    ax.set_yticklabels ([f'{e*1000:.1f}' for e in yticks], fontsize=12, color='gray')
    ax.set_zlabel (zlabel, fontsize=16, labelpad=10)
    zticks = ax.get_zticks ()
    ax.set_zticklabels ([f'{e*1000:.1f}' for e in zticks], fontsize=12, color='gray')

    # Give the plot a title
    ax.set_title ('Transducer Element Positions', fontsize=24, pad=50)

    # Set the background color and view angle
    plt.grid (axis='y', color='gray', ls=':')
    ax.set_facecolor ('white')
    ax.view_init (elev=45, azim=225)

    # Format legend
    legend = plt.legend (fontsize=12, frameon=True, edgecolor='k', facecolor='w', loc=3,
                         bbox_to_anchor=(-.05, -.05))
    plt.setp (legend.get_texts ()[-1], color='dodgerblue', alpha=.8)

    # Change view, title padding, legend text if plotting focal points
    if plot_focal_points:
      ax.set_title ('Transducer Element Positions', fontsize=24, pad=45)
      ax.view_init (elev=25, azim=45)
      plt.setp (legend.get_texts ()[0], color='red', alpha=.8)

    if show:
      plt.show()
    else:
      return fig, ax

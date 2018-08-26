import unittest


class TestDeviceProperties (unittest.TestCase):
  def test_dict (self):
    '''Test __dict__ member.'''
    import sys
    sys.path.append ('../install')
    import torch
    import openbcsim
    a = openbcsim.DeviceProperties ()

    # __dict__ starts off empty
    self.assertFalse (a.__dict__)

    # __dict__ populates after calling __repr__() once
    self.assertTrue (str (a) == str (a.__dict__))

    # __dict__ should contain all member variables
    dict_ = {}
    for key in dir (a):
      if not key.startswith ('__'):
        dict_[key] = getattr (a, key)
        if len (dict_) <= 5 or len (dict_) > len (a.__dict__) - 5:
          print (key, ': ', dict_[key])
        if len (dict_) == 5:
          print ('...')
    self.assertTrue (dict_ == a.__dict__)

  def test_getitiem (self):
    import sys
    sys.path.append ('../install')
    import torch
    import openbcsim as openbcsim
    a = openbcsim.DeviceProperties ()

    # Initialize __dict__ by calling __repr__()
    str (a)

    values = [a[key] for key in a.__dict__]
    print (values)
    self.assertEqual (values, list (a.__dict__.values ()))


class TestSimulator (unittest.TestCase):
  def test_launch (self):
    import sys
    sys.path.append ('../src')
    import Simulator as bc
    tx = bc.Transducer ()
    sim = bc.Simulator (tx=tx)
    sim.load_field_ii_scatterers ('../data/Field II/cyst_phantom/pht_data.mat')
    sim.set_gaussian_excitation ()
    print (bc.openbcsim.make_shape (sim.to_struct ()))
    print (bc.openbcsim.make_grid ())
    result = sim.launch ()
    print (result.shape)
    print (result)
    self.assertTrue (True)

  def test_linear_transducer (self):
    import sys
    sys.path.append ('../src')
    import Simulator as bc
    tx = bc.LinearTransducer (num_elements=192)
    print ('tx.num_focal_points:', tx.num_focal_points)
    sim = bc.Simulator (tx=tx)
    sim.load_field_ii_scatterers ('../data/Field II/cyst_phantom/pht_data.mat')
    sim.set_gaussian_excitation ()
    print (bc.openbcsim.make_shape (sim.to_struct ()))
    print (bc.openbcsim.make_grid ())
    result = sim.launch ()
    print (result.shape)
    print (result)
    self.assertTrue (True)


if __name__ == '__main__':
  import os
  from pathlib import Path
  scripts_dir = Path (__file__).parent
  os.chdir (scripts_dir)
  unittest.main ()

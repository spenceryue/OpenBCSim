import unittest
import torch
import openbcsim as bc


class TestDeviceProperties (unittest.TestCase):
  def test_dict (self):
    '''Test __dict__ member.'''
    a = bc.DeviceProperties ()

    # __dict__ starts off empty
    self.assertFalse (a.__dict__)

    # __dict__ populates after calling __repr__() once
    self.assertTrue (str (a) == str (a.__dict__))

    # __dict__ should contain all member variables
    dict_ = {}
    for key in dir (a):
      if not key.startswith ('__'):
        dict_[key] = getattr (a, key)
        print (key, ': ', dict_[key])
    self.assertTrue (dict_ == a.__dict__)

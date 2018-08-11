from unittest import TestCase
import torch
import openbcsim as bc

class Test (TestCase):
  def test_A (self):
    a = torch.ones ([1], device='cuda')
    t = bc.Transducer_float (1,1,1,a,a,a,a,a,1)
    args = bc.Simulator_float (1,1,1,1,1,t,t,1,a,a,a,a,1)
    bc.launch_float (args);
    self.assertTrue (1)

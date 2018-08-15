from unittest import TestCase
import torch
import axpy

class TestAxpy (TestCase):
  def test_hello (self):
    print (axpy.hello ())
    self.assertTrue (1)
  def test_hello2 (self):
    axpy.hello (axpy.TestClass (2,None,None))
    # axpy.hello (axpy.TestClass (2.3, torch.arange (10,5,-1), torch.ones (2)))
    self.assertTrue (1)

# TestAxpy ().test_hello ()
# TestAxpy ().test_hello2 ()
# axpy.TestClass (2.3, torch.ones ([1]), torch.arange (5,10))
# print (axpy.TestClass ())
print (axpy.TestClass (2.3,None,None))
# print (axpy.TestClass (2.3, torch.empty ([0]), torch.empty ([0])))
# print (axpy.hello ())
# axpy.hello (axpy.TestClass (2.3, torch.zeros ([0]), torch.empty ([1])))

# import sys; sys.path.append (r'C:\Users\spenc\Desktop\OpenBCSim\jupyter\python_module\axpy\build\lib.win-amd64-3.6'); import none_arg
# print (none_arg.function_taking_optional (torch.ones(5)))
# print (none_arg.function_taking_optional (None))
# print (none_arg.function_taking_optional ())

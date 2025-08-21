import math
import numpy as np


class Value:
  def __init__(self, data, _children=(), label=''):
    self.data = data
    self._prev = _children
    self._op = ''
    self.label = label
    self.grad = 0 
  
  def __repr__(self):
    return f"Data([{self.data}])"


if __name__ == '__main__':
  a = Value(2)
  a = a+2
  print(a)

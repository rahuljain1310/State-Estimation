#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : control.py
# Author : Rahul Jain
# Email  : rahuljain13101999@gmail.com
# Date   : 23/03/2021

import numpy as np
from math import sin, cos, pi

class NoControlPolicy():
  def __init__(self):
    pass

  def input(self):
    return np.array([[0], [0]])
    
class SineCosineControlPolicy():
  def __init__(self, Tx, Ty):
    self.wx, self.wy = 2*pi/Tx, 2*pi/Ty
    self.t = -1

  def input(self):
    self.t += 1
    return np.array([[sin(self.wx*self.t)], [cos(self.wy*self.t)]])

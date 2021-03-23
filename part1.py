#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : part1.py
# Author : Rahul Jain
# Email  : rahuljain13101999@gmail.com
# Date   : 23/03/2021

import numpy as np
from model import AirPlaneModel
from config import A, B, C, R, Q
import utils as utl

## Plotting Trajectory of Actual and Observed State of Airplan
## No Control Input, Inital Position (0,0), Initial Velocity (1, 1)

X0 = np.array([[0], [0], [1], [1]])
U = np.array([[0], [0]])
airplane = AirPlaneModel(A, B, C, R, Q, X0)
for T in range(200):
  _, _Z = airplane.step(U)

legends = ['Actual', 'Observed']
utl.display_trajectories([airplane.actual_trajectory, airplane.observed_trajectory], legends)
utl.display_XY_trajectories([airplane.actual_trajectory, airplane.observed_trajectory], legends)
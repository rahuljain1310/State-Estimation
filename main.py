#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : main.py
# Author : Rahul Jain
# Email  : rahuljain13101999@gmail.com
# Date   : 23/03/2021

import json
import numpy as np
from math import sin, cos, pi
from filter import KalmanFilter
from model import AirPlaneModel
from config import A, B, C, R, Q
import utils as utl

X_mean = np.array([10., 10., 1., 1.])
X_cov = np.diag([100**2, 100**2, 0.01, 0.01])
X0 = np.random.multivariate_normal(X_mean, X_cov, 1).reshape(-1, 1)

airplane = AirPlaneModel(A, B, C, R, Q, X0)
estimator = KalmanFilter(A, B, C, R, Q, X_mean, X_cov)

# U = np.array([[0], [0]])
Tx, Ty = 30, 30 
wx, wy = 2*pi/Tx, 2*pi/Ty

for T in range(200):
  U = np.array([[sin(wx*T)], [cos(wy*T)]])
  _ , zt = airplane.step(U)
  estimator.step(U, zt)

legends = ['Actual', 'Estimated', 'Observed']
trajectories = [airplane.actual_trajectory , estimator.estimated_trajectory, airplane.observed_trajectory]
utl.display_trajectories(airplane.actual_trajectory , estimator.estimated_trajectory, airplane.observed_trajectory)
utl.display_XY_trajectories(trajectories, legends)
utl.display_estimation_error(airplane.actual_trajectory, estimator.estimated_trajectory)
print("Press Key To Quit .."); input()
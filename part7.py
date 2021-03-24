#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : part7.py
# Author : Rahul Jain
# Email  : rahuljain13101999@gmail.com
# Date   : 23/03/2021

import json
import numpy as np
from math import sin, cos, pi

from numpy.lib.function_base import append
from filter import KalmanFilter
from model import AirPlaneModel
from control import SineCosineControlPolicy
from config import A, B, C, R, Q
import utils as utl

X_mean = np.array([10., 10., 1., 1.])
X_cov = np.diag([0.01**2, 0.01**2, 0.01**2, 0.01**2])
X0 = np.random.multivariate_normal(X_mean, X_cov, 1).reshape(-1, 1)

airplane = AirPlaneModel(A, B, C, R, Q, X0)
estimator = KalmanFilter(A, B, C, R, Q, X_mean, X_cov)
control = SineCosineControlPolicy(30, 30)

for T in range(0,45):
  U = control.input()
  _ , zt = airplane.step(U)
  if (T >=10 and T<20) or (T>=30 and T<40):
    estimator.step(U)
  else:
    estimator.step(U, zt)

# Display Trajectories
actual = airplane.actual_trajectory
estimated = estimator.estimated_trajectory
observed = airplane.observed_trajectory
Means = estimator.belief_mean_list
Covariances = estimator.belief_cov_list
utl.display_uncertainity_ellipse(estimated, actual,
  Means[10:20]+Means[30:40], Covariances[10:20]+Covariances[30:40],
  legends=['Estimated Trajectory', 'Actual Trajectory'])
utl.display_estimation_error(actual, estimated)
utl.display_trajectories(actual , estimated, observed)
utl.display_XY([actual , estimated, observed], ['Actual', 'Estimated', 'Observed'])
print("Press Key To Quit .."); input()
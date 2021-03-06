#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : main.py
# Author : Rahul Jain
# Email  : rahuljain13101999@gmail.com
# Date   : 23/03/2021

import numpy as np
import utils as utl
from filter import KalmanFilter
from model import AirPlaneModel
from control import NoControlPolicy, SineCosineControlPolicy
from config import A, B, C, R, Q

X_mean = np.array([10., 10., 1., 1.])
X_cov = np.diag([0.01**2, 0.01**2, 0.01**2, 0.01**2])
X0 = np.random.multivariate_normal(X_mean, X_cov, 1).reshape(-1, 1)

airplane = AirPlaneModel(A, B, C, R, Q, X0)
estimator = KalmanFilter(A, B, C, R, Q, X_mean, X_cov)
control = NoControlPolicy()

for T in range(0,100):
  U = control.input()
  _ , zt = airplane.step(U)
  estimator.step(U, zt)

actual = airplane.actual_trajectory
estimated = estimator.estimated_trajectory
observed = airplane.observed_trajectory
Means = estimator.belief_mean_list
Covariances = estimator.belief_cov_list

utl.display_estimation_error(actual, estimated)
utl.display_trajectories(actual , estimated, observed)
utl.display_velocity(airplane.actual_velocity, estimator.estimated_velocity)
utl.display_XY([airplane.actual_velocity, estimator.estimated_velocity], ['Actual', 'Estimator'], title='Velocity')
utl.display_XY([actual , estimated, observed], ['Actual', 'Estimated', 'Observed'], title='Displacement')
utl.display_uncertainity_ellipse(estimated, actual, Means, Covariances, legends=['Estimated', 'Actual'])
print("Press Key To Quit .."); input()
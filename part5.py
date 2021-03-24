#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : part5.py
# Author : Rahul Jain
# Email  : rahuljain13101999@gmail.com
# Date   : 23/03/2021

import json
import numpy as np
from math import sin, cos, pi
from filter import KalmanFilter
from model import AirPlaneModel
from config import A, B, C, R
from matplotlib import pyplot as plt
import utils as utl

ACT = list()
EST = list()
listQ = [2, 10, 50]

for Q in listQ:
  Q = (Q**2)*np.identity(2) 
  X_mean = np.array([10., 10., 1., 1.])
  X_cov = 0.01*np.identity(4)
  X0 = np.random.multivariate_normal(X_mean, X_cov, 1).reshape(-1, 1)
  airplane = AirPlaneModel(A, B, C, R, Q, X0)
  estimator = KalmanFilter(A, B, C, R, Q, X_mean, X_cov)
  Tx, Ty = 30, 30 
  wx, wy = 2*pi/Tx, 2*pi/Ty
  for T in range(200):
    U = np.array([[sin(wx*T)], [cos(wy*T)]])
    _ , zt = airplane.step(U)
    estimator.step(U, zt)
  ACT.append(airplane.actual_trajectory)
  EST.append(estimator.estimated_trajectory)


fig, axs = plt.subplots(len(listQ))
for i in range(len(listQ)):
  legends = ['Actual', f'Estimated σ = {listQ[i]}']
  x_act, y_act = utl.get_xy_trajectory(ACT[i])
  axs[i].plot(x_act, y_act)
  x_est, y_est = utl.get_xy_trajectory(EST[i])
  axs[i].plot(x_est, y_est)
  axs[i].set_ylabel('Displacement')
  axs[i].legend(legends)
plt.xlabel('Timestep, T')
plt.show()

fig, axs = plt.subplots()
for i in range(len(listQ)):
  e = [np.linalg.norm(ACT[i][k]-EST[i][k]) for k in range(len(ACT[i])) ]
  t = list(range(len(e)))
  axs.plot(t, e)
  axs.set_title('Actual and Estimated Trajectory')
  axs.legend([f'σ = {q}' for q in listQ])
  axs.set_title('Estimation Error')
  axs.set_xlabel('Timestep, T')
  axs.set_ylabel('Error')
plt.show()

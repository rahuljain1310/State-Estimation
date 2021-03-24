#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : filter.py
# Author : Rahul Jain
# Email  : rahuljain13101999@gmail.com
# Date   : 23/03/2021

import numpy as np

class KalmanFilter(object):
  def __init__(self, A, B, C, R, Q, belief_mean, belief_cov):
    self.A = A
    self.B = B
    self.C = C
    self.R = R
    self.Q = Q
    self.belief_mean = belief_mean.reshape(-1, 1)
    self.belief_cov = belief_cov
    self.belief_mean_list=  list()
    self.belief_cov_list = list()
    self.estimated_trajectory = list()
    self.estimated_velocity = list()

  def get_kalman_gain(self, cov):
    a = np.matmul(cov, self.C.T)
    b = np.matmul(np.matmul(self.C, cov), self.C.T) + self.Q
    K = a.dot(np.linalg.inv(b))
    return K
  
  def record(self, mean, cov):
    pos = mean.reshape((-1,))[:2]
    vel = mean.reshape((-1,))[2:]
    self.estimated_trajectory.append(pos)
    self.estimated_velocity.append(vel)
    self.belief_mean_list.append(mean)
    self.belief_cov_list.append(cov)

  def step(self, U, Z = None):
    mean = np.matmul(self.A, self.belief_mean) + np.matmul(self.B, U)
    cov = np.matmul(self.A, np.matmul(self.belief_cov, self.A.T)) + self.R

    if Z is not None:
      K = self.get_kalman_gain(cov)
      mean = mean + np.matmul(K , Z - self.C.dot(mean))
      cov = (np.identity(K.shape[0]) - K.dot(self.C)).dot(cov)

    self.record(mean, cov)
    self.belief_mean = mean
    self.belief_cov = cov
    return mean, cov

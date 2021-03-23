#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : motion.py
# Author : Rahul Jain
# Email  : rahuljain13101999@gmail.com
# Date   : 23/03/2021

import numpy as np

class MotionModel(object):
    def __init__(self, A, B, R):
        """
            Motion Model
            x' = Ax + Bu + er
            where, 
                x = [x, y, vx, vy]
            input:
            (A) - N x N Matrix
            (B) - N x U Matrix
            (R) - N x N Covariance Matrix
        """
        assert A.shape[0] == B.shape[0] == R.shape[0]
        self.A = A.copy()
        self.B = B.copy()
        self.R = R.copy()
        self.dim = A.shape[0]
        self.mean = np.zeros(self.dim)

    def get_noise_sample(self):
        x =  np.random.multivariate_normal(self.mean, self.R, 1)
        return x.reshape(-1, 1)

    def __call__(self, X, U):
        _X = np.matmul(self.A, X)
        _X += np.matmul(self.B, U)
        _X += self.get_noise_sample()
        return _X

class SensorModel(object):
    def __init__(self, C, Q):
        """
            Sensor Model
            z = Cx + eq
            where, 
                x = [x, y, vx, vy]
                z = [x`, y`]
            input:
            (C) - Z x N Matrix
            (Q) - Z x Z Covariance Matrix
        """
        assert C.shape[0] == Q.shape[0]
        self.C = C.copy()
        self.Q = Q.copy()
        self.dim = C.shape[0]
        self.mean = np.zeros(self.dim)

    def get_noise_sample(self):
        x = np.random.multivariate_normal(self.mean, self.Q, 1)
        return x.reshape(-1, 1)

    def __call__(self, X):
        _Z = np.matmul(self.C, X)
        _Z += self.get_noise_sample()
        return _Z

class AirPlaneModel():
    def __init__(self, A, B, C, R, Q, X0):
        self.motion_model = MotionModel(A, B, R)
        self.sensor_model = SensorModel(C, Q)
        self.actual_trajectory = list()
        self.observed_trajectory = list()
        X0 = X0.reshape(-1, 1)
        self.X_prev = X0
        self.X0 = X0
    
    def record(self, X, Z):
        cur_pos = X.reshape((-1,))[:2]
        obs_pos = Z.reshape((-1,))
        self.actual_trajectory.append(cur_pos)
        self.observed_trajectory.append(obs_pos)

    def step(self, U):
        _X = self.motion_model(self.X_prev, U)
        _Z = self.sensor_model(_X)
        self.record(_X, _Z)
        self.X_prev = _X
        return _X, _Z






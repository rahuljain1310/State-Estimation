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
    self.estimated_trajectory = list()

  def get_kalman_gain(self, _cov):
    a = np.matmul(_cov, self.C.T)
    b = np.matmul(np.matmul(self.C, _cov), self.C.T) + self.Q
    K = a.dot(np.linalg.inv(b))
    return K
  
  def record(self, X):
    pos = X.reshape((-1,))[:2]
    self.estimated_trajectory.append(pos)

  def step(self, U, Z):
    _mean = np.matmul(self.A, self.belief_mean) + np.matmul(self.B, U)
    _cov = np.matmul(self.A, np.matmul(self.belief_cov, self.A.T)) + self.R
    K = self.get_kalman_gain(_cov)
    mean = _mean + np.matmul(K , Z - self.C.dot(_mean))
    cov = (np.identity(K.shape[0]) - K.dot(self.C)).dot(_cov)
    self.record(mean)
    self.belief_mean = mean
    self.belief_cov = cov
    return mean, cov

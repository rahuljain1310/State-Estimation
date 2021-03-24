#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : utils.py
# Author : Rahul Jain
# Email  : rahuljain13101999@gmail.com
# Date   : 23/03/2021

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

def get_xy(trajectory):
  x = [s[0] for s in trajectory]
  y = [s[1] for s in trajectory]
  return x,y

def display_trajectories(act, est, obs):
  fig, axs = plt.subplots(1, 2)
  axs[0].set_title('Actual And Estimated')
  for t in [act, est]:
    x, y = get_xy(t)
    axs[0].plot(x, y)
  axs[0].legend(['Actual', 'Estimated'])
  
  axs[1].set_title('Actual and Observed')
  for t in [act, obs]:
    x, y = get_xy(t)
    axs[1].plot(x, y)
  axs[1].legend(['Actual', 'Observed'])
  for ax in axs.flat: ax.set(xlabel='X', ylabel='Y')
  plt.show(block=False)
  

def display_XY(trajectories, legends, title='Magnitude'):
  X = list()
  Y = list()
  for t in trajectories:
    x, y = get_xy(t)
    X.append(x); Y.append(y)
  t = list(range(len(X[0])))
  fig, axs = plt.subplots(1, 2)
  axs[0].set_title('X-Axis')
  for x in X:
    axs[0].plot(t, x)
  axs[0].legend(legends)
  axs[1].set_title('Y-Axis')
  for y in Y:
    axs[1].plot(y)
  axs[1].legend(legends)
  for ax in axs.flat: ax.set(xlabel='Timestep, T', ylabel=title)
  plt.show(block=False)

def display_estimation_error(act_traj, est_trac):
  assert len(act_traj) == len(est_trac)
  e = [np.linalg.norm(act_traj[i]-est_trac[i]) for i in range(len(act_traj)) ]
  t = list(range(len(e)))
  fig1, ax1 = plt.subplots()
  ax1.plot(t, e)
  ax1.set_title('Estimation Error')
  ax1.set_xlabel('Timestep, T')
  ax1.set_ylabel('Error (Euclidean Distance)')
  plt.show(block=False)

def plot_uncertainity_ellipse(mean, cov, ax, edgecolor='red'):
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2, facecolor='none', edgecolor=edgecolor)
    scale_x, scale_y  = np.sqrt(cov[0, 0]), np.sqrt(cov[1, 1])
    mean_x, mean_y = mean[0], mean[1]
    transf = transforms.Affine2D().rotate_deg(45).scale(scale_x, scale_y).translate(mean_x, mean_y)
    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

def display_uncertainity_ellipse(est_trac, act_trac, Means, Covariances, legends):
  fig1, ax1 = plt.subplots()
  x_est, y_est = get_xy(est_trac)
  x_act, y_act = get_xy(act_trac)
  ax1.plot(x_est, y_est)
  ax1.plot(x_act, y_act)
  for i in range(len(Means)):
    mean = Means[i].flatten()[:2]
    cov = Covariances[i][:2, :2]
    plot_uncertainity_ellipse(mean[:2], cov[:2, :2], ax1)  
  ax1.set_title('Trajectory')
  ax1.set_xlabel('X')
  ax1.set_ylabel('Y')
  ax1.legend(legends)
  plt.show(block=False)

def display_velocity(act_vel, est_vel):
  fig1, ax1 = plt.subplots()
  x_est, y_est = get_xy(est_vel)
  x_act, y_act = get_xy(act_vel)
  ax1.plot(x_est, y_est)
  ax1.plot(x_act, y_act)
  ax1.set_title('Velocity')
  ax1.set_xlabel('X')
  ax1.set_ylabel('Y')
  ax1.legend(['Estimated', 'Actual'])
  plt.show(block=False)

  
  
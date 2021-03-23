from matplotlib import pyplot as plt
import numpy as np

def get_xy_trajectory(trajectory):
  x = [s[0] for s in trajectory]
  y = [s[1] for s in trajectory]
  return x,y

def display_trajectories(act, est, obs):
  fig, axs = plt.subplots(1, 2)
  axs[0].set_title('Actual And Estimated')
  for t in [act, est]:
    x, y = get_xy_trajectory(t)
    axs[0].plot(x, y)
  axs[0].legend(['Actual', 'Estimated'])
  
  axs[1].set_title('Actual and Observed')
  for t in [act, obs]:
    x, y = get_xy_trajectory(t)
    axs[1].plot(x, y)
  axs[1].legend(['Actual', 'Observed'])
  for ax in axs.flat: ax.set(xlabel='X', ylabel='Y')
  plt.show(block=False)
  

def display_XY_trajectories(trajectories, legends):
  X = list()
  Y = list()
  for t in trajectories:
    x, y = get_xy_trajectory(t)
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
  for ax in axs.flat: ax.set(xlabel='Timestep, T', ylabel='Displacement')
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


  
  
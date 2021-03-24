#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : hmm.py
# Author : Pratyush Garg
# Date   : 23/03/2021

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)

class Grid:
	def __init__(self, size):
		self.size = size
		self.robot = None
		self.robot_path = []
		self.viterbi_path = []


		self.sensors = []

		self.Pr_up = 0.4
		self.Pr_down = 0.1
		self.Pr_left = 0.2
		self.Pr_right = 0.3

		self.Cum_Pr_up = self.Pr_up
		self.Cum_Pr_down = self.Pr_up + self.Pr_down
		self.Cum_Pr_left = self.Pr_up + self.Pr_down + self.Pr_left
		self.Cum_Pr_right = self.Pr_up + self.Pr_down + self.Pr_left + self.Pr_right

		self.sensor_Pr_map = np.array([[0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5],
									   [0.5,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.5],
									   [0.5,0.6,0.7,0.7,0.7,0.7,0.7,0.6,0.5],
									   [0.5,0.6,0.7,0.8,0.8,0.8,0.7,0.6,0.5],
									   [0.5,0.6,0.7,0.8,0.9,0.8,0.7,0.6,0.5],
									   [0.5,0.6,0.7,0.8,0.8,0.8,0.7,0.6,0.5],
									   [0.5,0.6,0.7,0.7,0.7,0.7,0.7,0.6,0.5],
									   [0.5,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.5],
									   [0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]])

		self.sensor_measurements = []
		self.measurement_pr = []

		self.current_estimate = np.ones((self.size,self.size))/(self.size*self.size)
		self.forward_estimates = []

		self.beta = np.ones((self.size,self.size))		
		self.backward_estimates = []

		self.mu_max = []
		self.mu_argmax = []


	def add_sensor(self,x,y):
		self.sensors.append((x,y))

	def add_robot(self,x,y):
		self.robot = (x,y)
		self.robot_path.append(self.robot)

	def move_up(self):
		x = self.robot[0]
		y = self.robot[1]

		if y + 1 < self.size:
			self.robot = (x,y+1)

	def move_down(self):
		x = self.robot[0]
		y = self.robot[1]

		if y - 1 > -1:
			self.robot = (x,y-1)

	def move_left(self):
		x = self.robot[0]
		y = self.robot[1]

		if x - 1 > -1:
			self.robot = (x - 1, y)

	def move_right(self):
		x = self.robot[0]
		y = self.robot[1]

		if x + 1 < self.size:
			self.robot = (x + 1, y)

	def move_robot(self):
		x = np.random.rand()

		if x < self.Cum_Pr_up:
			self.move_up()
		elif x < self.Cum_Pr_down:
			self.move_down()
		elif x < self.Cum_Pr_left:
			self.move_left()
		else:
			self.move_right()

		self.robot_path.append(self.robot)

	def measure(self,sensor):

		x = self.robot[0] - sensor[0] + 4
		y = self.robot[1] - sensor[1] + 4

		if x < 0 or y < 0 or x > 8 or y >8:
			return 0
		else:
			num = np.random.rand()
			if num < self.sensor_Pr_map[y,x]:
				return 1
			else:
				return 0

	def measure_presence(self):

		presence = []
		for i in range(len(self.sensors)):
			val = self.measure((self.sensors[i]))
			presence.append(val)

		self.sensor_measurements.append(presence)

		return presence

	def plot_manhattan(self):

		distances = []

		for i in range(self.time):
			x1,y1 = self.robot_path[i]
			Pr_Map = self.forward_estimates[i]

			x2,y2 = np.unravel_index(np.argmax(Pr_Map, axis=None), Pr_Map.shape)

			dist = abs(x2-x1) + abs(y2-y1)
			distances.append(dist)

		plt.clf()
		plt.plot(distances, 'b', label = "Forward Estimate")

		distances = []

		for i in range(self.time):
			x1,y1 = self.robot_path[i]
			Pr_Map = self.backward_estimates[i]

			x2,y2 = np.unravel_index(np.argmax(Pr_Map, axis=None), Pr_Map.shape)

			dist = abs(x2-x1) + abs(y2-y1)
			distances.append(dist)

		plt.plot(distances, 'r', label = "Backward Estimate")
		plt.ylabel('Manhattan Distance')
		plt.xlabel('Time Step')
		plt.legend(loc="upper left")
		plt.savefig("Manhattan.jpg")


	def plot_manhattan_viterbi(self):

		distances = []

		for i in range(self.time):
			x1,y1 = self.robot_path[i]
			x2,y2 = self.viterbi_path[i]

			dist = abs(x2-x1) + abs(y2-y1)
			distances.append(dist)

		plt.clf()
		plt.plot(distances)
		plt.ylabel('Manhattan Distance')
		plt.xlabel('Time Step')
		plt.savefig("Manhattan_Viterbi.jpg")

		


	def viterbi(self):

		"""
		-1: Stay
		 0: UP
		 1: Down
		 2: Right
		 3: Left

		"""

		if self.mu_max == []:
			mu_max = self.current_estimate
			mu_argmax = np.ones((self.size,self.size))*(-1)
			self.mu_max.append(mu_max)
			self.mu_argmax.append(mu_argmax)
		else:
			mu_max = np.zeros((self.size,self.size))
			mu_argmax = np.zeros((self.size,self.size))

			for i in range(self.size):
				for j in range(self.size):

					probs = []
					args = []

					if j - 1 < 0:
						probs.append(self.mu_max[-1][j,i]*self.Pr_up*self.measurement_pr[-1][j,i])
						args.append(-1)
					else:
						probs.append(self.mu_max[-1][j-1,i]*self.Pr_up*self.measurement_pr[-1][j-1,i])
						args.append(0)

					if j + 1 >= self.size:
						probs.append(self.mu_max[-1][j,i]*self.Pr_down*self.measurement_pr[-1][j,i])
						args.append(-1)
					else:
						probs.append(self.mu_max[-1][j+1,i]*self.Pr_down*self.measurement_pr[-1][j+1,i])
						args.append(1)

					if i - 1 < 0:
						probs.append(self.mu_max[-1][j,i]*self.Pr_right*self.measurement_pr[-1][j,i])
						args.append(-1)
					else:
						probs.append(self.mu_max[-1][j,i-1]*self.Pr_right*self.measurement_pr[-1][j,i-1])
						args.append(2)

					if i + 1 >= self.size:
						probs.append(self.mu_max[-1][j,i]*self.Pr_left*self.measurement_pr[-1][j,i])
						args.append(-1)
					else:
						probs.append(self.mu_max[-1][j,i+1]*self.Pr_left*self.measurement_pr[-1][j,i+1])
						args.append(3)

					probs = np.array(probs)
					mu_max[j,i] = np.max(probs)
					mu_argmax[j,i] = args[np.argmax(probs)]


			self.mu_argmax.append(mu_argmax)
			self.mu_max.append(mu_max)


	def construct_path(self):
		path = []
		mu_max = self.mu_argmax[-1]
		loc = np.unravel_index(np.argmax(mu_max, axis=None), mu_max.shape)
		for i in range(self.time-1,-1,-1):
			path = [(loc[1],loc[0])] + path
			# path.append((loc[1],loc[0]))
			mu_argmax = self.mu_argmax[i]
			move = mu_argmax[loc]
			if move == -1:
				loc = (loc[0],loc[1])
			elif move == 0:
				loc = (loc[0],loc[1]-1)
			elif move == 1:
				loc = (loc[0],loc[1]+1)
			elif move == 2:
				loc = (loc[0]-1,loc[1])
			else:
				loc = (loc[0]+1,loc[1])

		self.viterbi_path = path
		self.plot_path()


	def plot_path(self):

		plt.clf()

		X_true = []
		Y_true = []

		X_viterbi = []
		Y_viterbi = []

		for i in range(self.time):
			X_true.append(self.robot_path[i][0])
			Y_true.append(self.robot_path[i][1])
			X_viterbi.append(self.viterbi_path[i][0])
			Y_viterbi.append(self.viterbi_path[i][1])

		plt.plot(X_true,Y_true,'b', label='Actual Path')
		plt.plot(X_viterbi,Y_viterbi,'r', label='Most Likely Path')
		plt.plot(X_true[0],Y_true[0],'bX')
		plt.plot(X_viterbi[0],Y_viterbi[0],'rX')

		plt.legend(loc="upper left")

		plt.savefig("Path.jpg")




	def simulate(self,time):

		self.time = time

		for i in range(time):
			presence = self.measure_presence()
			self.incorporate_evidence(presence)
			self.viterbi()
			self.Markov_Model_Forward()
			self.move_robot()


	def plot_forward_estimate(self):

		plt.clf()

		for i in range(self.time):
			robot = np.zeros((self.size,self.size))
			robot[self.robot_path[i][1],self.robot_path[i][0]] += 1

			f, axarr = plt.subplots(1,2)
			f.suptitle("Time Step: " + str(i + 1))
			axarr[0].imshow(np.flip(robot,axis=0))
			axarr[0].set_title("Actual Position")
			axarr[1].imshow(np.flip(self.forward_estimates[i],axis=0))
			axarr[1].set_title("Forward Estimate")
			plt.savefig("Forward_" + str(i + 1) + ".jpg")

	def plot_backward_estimate(self):

		plt.clf()

		for i in range(self.time):
			robot = np.zeros((self.size,self.size))
			robot[self.robot_path[i][1],self.robot_path[i][0]] += 1

			f, axarr = plt.subplots(1,2)
			f.suptitle("Time Step: " + str(i + 1))
			axarr[0].imshow(np.flip(robot,axis=0))
			axarr[0].set_title("Actual Position")
			axarr[1].imshow(np.flip(self.backward_estimates[i],axis=0))
			axarr[1].set_title("Backward Estimate")
			plt.savefig("Backward_" + str(i + 1) + ".jpg")


	def predict(self,time):

		plt.clf()

		estimate = self.current_estimate

		for i in range(time):

			plt.imshow(np.flip(estimate,axis = 0))
			plt.title("Prediction Time Step: " + str(i + self.time + 1))
			plt.savefig("Prediction_" + str(i+self.time +1) + '.jpg')

			estimate = self.Markov_Model_Predict(estimate)


	def Markov_Model_Predict(self,estimate):
		next_step = np.zeros((self.size,self.size))

		for i in range(self.size):
			for j in range(self.size):

				if j - 1 < 0:
					next_step[j,i] += estimate[j,i]*self.Pr_up
				else:
					next_step[j,i] += estimate[j-1,i]*self.Pr_up

				if j + 1 >= self.size:
					next_step[j,i] += estimate[j,i]*self.Pr_down
				else:
					next_step[j,i] += estimate[j+1,i]*self.Pr_down

				if i - 1 < 0:
					next_step[j,i] += estimate[j,i]*self.Pr_right
				else:
					next_step[j,i] += estimate[j,i-1]*self.Pr_right

				if i + 1 >= self.size:
					next_step[j,i] += estimate[j,i]*self.Pr_left
				else:
					next_step[j,i] += estimate[j,i+1]*self.Pr_left

		estimate = next_step/np.sum(next_step)

		return estimate


	def Markov_Model_Forward(self):
		next_step = np.zeros((self.size,self.size))

		for i in range(self.size):
			for j in range(self.size):

				if j - 1 < 0:
					next_step[j,i] += self.current_estimate[j,i]*self.Pr_up
				else:
					next_step[j,i] += self.current_estimate[j-1,i]*self.Pr_up

				if j + 1 >= self.size:
					next_step[j,i] += self.current_estimate[j,i]*self.Pr_down
				else:
					next_step[j,i] += self.current_estimate[j+1,i]*self.Pr_down

				if i - 1 < 0:
					next_step[j,i] += self.current_estimate[j,i]*self.Pr_right
				else:
					next_step[j,i] += self.current_estimate[j,i-1]*self.Pr_right

				if i + 1 >= self.size:
					next_step[j,i] += self.current_estimate[j,i]*self.Pr_left
				else:
					next_step[j,i] += self.current_estimate[j,i+1]*self.Pr_left

		self.current_estimate = next_step/np.sum(next_step)

	def incorporate_evidence(self, presence):

		measurement_pr = self.Sensor_Model(presence)
		self.measurement_pr.append(measurement_pr)

		self.current_estimate = self.current_estimate*measurement_pr
		self.current_estimate = self.current_estimate/np.sum(self.current_estimate)

		self.forward_estimates.append(self.current_estimate)


	def Sensor_Model(self,presence):

		measurement_pr = np.zeros((self.size,self.size))

		for i in range(self.size):
			for j in range(self.size):
				pr = 1
				for k in range(len(self.sensors)):
					x = i - self.sensors[k][0] + 4
					y = j - self.sensors[k][1] + 4

					if x < 0 or y < 0 or x > 8 or y >8:
						if presence[k] == 0:
							pr *= 1
						else:
							pr *= 0
					else:
						if presence[k] == 0:
							pr *= (1 - self.sensor_Pr_map[y,x])
						else:
							pr *= self.sensor_Pr_map[y,x]

				measurement_pr[j,i] = pr

		return measurement_pr


	def Markov_Model_Backward(self,presence):
		next_step = np.zeros((self.size,self.size))
		measurement_pr = self.Sensor_Model(presence)

		for i in range(self.size):
			for j in range(self.size):

				if j - 1 < 0:
					next_step[j,i] += self.beta[j,i]*self.Pr_down*measurement_pr[j,i]
				else:
					next_step[j,i] += self.beta[j-1,i]*self.Pr_down*measurement_pr[j-1,i]

				if j + 1 >= self.size:
					next_step[j,i] += self.beta[j,i]*self.Pr_up*measurement_pr[j,i]
				else:
					next_step[j,i] += self.beta[j+1,i]*self.Pr_up*measurement_pr[j+1,i]

				if i - 1 < 0:
					next_step[j,i] += self.beta[j,i]*self.Pr_left*measurement_pr[j,i]
				else:
					next_step[j,i] += self.beta[j,i-1]*self.Pr_left*measurement_pr[j,i-1]

				if i + 1 >= self.size:
					next_step[j,i] += self.beta[j,i]*self.Pr_right*measurement_pr[j,i]
				else:
					next_step[j,i] += self.beta[j,i+1]*self.Pr_right*measurement_pr[j,i+1]

		self.beta = next_step/np.sum(next_step)

	def smoothing(self):

		for i in range(self.time-1,-1,-1):
			presence = self.sensor_measurements[i]
			forward_estimate = self.forward_estimates[i]

			backward_estimate = self.beta * forward_estimate
			backward_estimate = backward_estimate/np.sum(backward_estimate)

			self.backward_estimates = [backward_estimate] + self.backward_estimates

			self.Markov_Model_Backward(presence)



grid = Grid(30)
grid.add_robot(15,15)
grid.add_sensor(8,15)
grid.add_sensor(15,15)
grid.add_sensor(22,15)
grid.add_sensor(15,22)

grid.simulate(25)
grid.plot_forward_estimate()
grid.smoothing()
grid.plot_backward_estimate()

grid.plot_manhattan()


grid.predict(25)

grid.construct_path()
grid.plot_manhattan_viterbi()
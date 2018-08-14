'''
Environment for robot arm reaching movement. The arm tries to reach the object on the tavle.
The environment will return a geographic (distance) information for the arm to learn.
The far away from target the less reward;

--vrep: True, training in the vrep env
		False, training without simulator

Requirement:
python 2.7
pypot 2.11.0
tensorflow 1.4.0

'''

# Import libraries
from pypot.vrep import from_vrep, close_all_connections
from pypot.robot import from_config
import time
import numpy as np
import itertools
import random
import pypot.dynamixel

# if in the vrep env
vrep = False 

if vrep:
    close_all_connections()
    poppy = from_vrep('poppy.json', scene = 'experiment.ttt') 

#initial motor angle
if vrep:
	arm_theta1 = 0
	arm_theta2 = 0 
	arm_theta3 = -10
	for m in poppy.motors:
		if m.id == 41:
			motor_41 = m
		if m.id == 42:
			motor_42 = m
		if m.id == 44:
			motor_44 = m
else:
	arm_theta1 = 90
	arm_theta2 = 85 
	arm_theta3 = 0

def initial_pose():
	move(motor_41, motor_41.present_position, arm_theta1)
	move(motor_42, motor_41.present_position, arm_theta2)
	move(motor_44, motor_44.present_position, arm_theta3)

class Env(object):
 	# action will be angle move between [-1,1]
 	# states bound [20, 140], [15, 105], [-100, -10]
	state_dim = 9	# theta1 & theta2 & theta3, distance to goal,get_point
	action_dim = 3
	get_point = False
	grab_counter = 0
	vrep_matrix = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]])
	i = 0
	if vrep:
		poppy.reset_simulation()		
		arm1l = 18.5
		arm2l = 21.5  #21.5 + 12
		theta_bound = np.array([[-75, 55], [-19, 70], [-100, -10]])
		action_bound = [-5, 5]
		point_bound = np.array([[-23, -33], [-10,6]]) # YZ
		point_l = 2
		arm_theta1 = 0
		arm_theta2 = 0 
		arm_theta3 = -10
		initial_pose()
		motor_41.compliant = False
		motor_42.compliant = False
		motor_44.compliant = False
		motor_41.torque_limit = 15
		motor_42.torque_limit = 15
		motor_44.torque_limit = 15
		motor_41.moving_speed = 10
		motor_42.moving_speed = 10
		motor_44.moving_speed = 10

	else:
		# table place:(20,0, 0, -66.5), table size:(33.5,31.5,34)
		arm1l = 15
		arm2l = 11.5 + 29.5
		theta_bound = np.array([[20, 140], [15, 105], [-100, -10]]) 
		action_bound = [-10, 10]
		point_bound = np.array([[30, 38], [0, 35]]) #XY
		point_l = 5
		motor_id = np.array([41, 42, 44])
		table_bound = np.array([[20, 51.5], [-2.5, 32]]) # x_bound, y_bound
		epislon = 3
		constraints = False

	def __init__(self, point_info = np.array([32.5, 0, -25 ]) ):
		self.arm_info = np.zeros(3)
		self.EE = np.zeros(3)
		self.arm_info[0] = arm_theta1
		self.arm_info[1] = arm_theta2
		self.arm_info[2] = arm_theta3
		if vrep:
			point_info = self.vrep_matrix.dot(point_info)
		self.point_info = point_info
		self.point_info_init = self.point_info.copy()
		self.EE = self.get_EE(self.arm_info)




	def step(self, action):
		done = False
		action_ = action  #* 180 / np.pi 	# action is np.array(2,)
		collision = False
		if not vrep:
			action_[np.abs(action_) < 1] = 1 # in the real robot should consider the case where action is smaller than 1deg??????? 
		goal_position_1 = np.clip((self.arm_info + action_)[0], self.theta_bound[0, 0] , self.theta_bound[0, 1] )
		goal_position_2 = np.clip((self.arm_info + action_)[1], self.theta_bound[1, 0],  self.theta_bound[1, 1] )
		goal_position_3 = np.clip((self.arm_info + action_)[2], self.theta_bound[2, 0],  self.theta_bound[2, 1] )
		if vrep:
			move(motor_41, motor_41.present_position, goal_position_1)
			move(motor_42, motor_42.present_position, goal_position_2)
			move(motor_44, motor_44.present_position, goal_position_3)

			self.arm_info[0] = motor_41.present_position 
			self.arm_info[1] = motor_42.present_position 
			self.arm_info[2] = motor_44.present_position
		else:
			######### collosion avoidance #########				
			# adaptive time sleep
			pred_arm = np.array([goal_position_1, goal_position_2, goal_position_3])
			pred_EE = self.get_EE(pred_arm)
			print('i:', self.i, 'pred_EE: ', pred_EE , 'distance: %.2f'% np.linalg.norm(pred_EE - self.point_info))
			collision = (pred_EE[2] < (-32.5+4) ) & ( (self.table_bound[0,0]-self.epislon) <pred_EE[0]< (self.table_bound[0,1]+self.epislon) ) & ( (self.table_bound[1,0]-self.epislon) < pred_EE[1] < (self.table_bound[1,1]+self.epislon) )
			self.constraints = collision
			if collision:
				self.i += 1
				s = self.get_state()
				r = - .1
				done = False
				print('I will collide with tableeeeeeeeeeeeeeee')
				return s, r, done, collision
			if (goal_position_1  > 80) & (goal_position_2  > 90) :
				self.i += 1
				s = self.get_state()
				r = - .1
				done = False
				collision = True                
				print('I will collide with myself')
				return s, r, done, collision

			self.arm_info[:] = np.array([goal_position_1, goal_position_2, goal_position_3])

		self.i += 1
# 		print('i:', self.i, ' go_pos1: %.2f' % goal_position_1, '   go_pos2: %.2f' % goal_position_2 , '   go_pos3: %.2f' % goal_position_3)
		self.EE = self.get_EE(self.arm_info)
		s = self.get_state()
		r = self._r_func(s[6])
		done = self.get_point
		return s, r, done, collision

	def _r_func(self, distance):
		t = 50
		abs_distance = distance
		r = -abs_distance/200
		# print('point_l : ', self.point_l, 'get_point: ',  self.get_point, ' abs_dis: %.2f'% abs_distance)
		if abs_distance < 8. and (not self.get_point):
			print('******************r+0.2**************************,| grab_counter: ', self.grab_counter)
			r += .2
			if abs_distance < self.point_l and (not self.get_point):
				print('******************r+1**************************,| grab_counter: ', self.grab_counter)
				r += 1.
				self.grab_counter += 1

				if self.grab_counter > t:
					r += 10.	
					self.get_point = True
					print('******************r+10**************************')
			elif abs_distance > self.point_l:
				self.grab_counter = 0
				self.get_point = False
		elif abs_distance > self.point_l:
			self.grab_counter = 0
			self.get_point = False
            
		return r

	def reset(self):
		self.get_point = False
		self.i = 0         
		if vrep:
			poppy.reset_simulation()
			self.point_info[1] = np.clip(self.point_bound[0, 0] + 10*np.random.random(), self.point_bound[0,0], self.point_bound[0,1])
			self.point_info[2] = np.clip(self.point_bound[1, 0] + 16*np.random.random(), self.point_bound[1,0], self.point_bound[1,1])
			self.point_info[0] = 23
			# initial random position
			self.arm_info[0] = np.clip(self.theta_bound[0, 0] + 130*np.random.random(), -30, 30)
			self.arm_info[1] = np.clip(self.theta_bound[1, 0] + 89*np.random.random(), 0, 10)
			self.arm_info[2] = np.clip(self.theta_bound[2, 0] + 90*np.random.random(), -40, -10)
			move(motor_41, motor_41.present_position, self.arm_info[0])
			move(motor_42, motor_42.present_position, self.arm_info[1])
			move(motor_44, motor_44.present_position, self.arm_info[2])
			self.EE = self.get_EE(self.arm_info)
			print('initial random point: ', self.point_info)
			print('initial random state: ', self.arm_info)
			self.arm_info[0] = motor_41.present_position  # initial state should be observation ???????
			self.arm_info[1] = motor_42.present_position		
			self.arm_info[2] = motor_44.present_position
		else: 
			self.point_info[0] = np.clip(self.point_bound[0, 0] + 8*np.random.random(), self.point_bound[0,0], self.point_bound[0,1]) # 37
			self.point_info[1] = np.clip(self.point_bound[1, 0] + 35*np.random.random(), self.point_bound[1,0], self.point_bound[1,1]) # 4
			self.point_info[2] = -25
			# initial robot arm configuration
			self.arm_info[0] = arm_theta1 #np.clip(self.theta_bound[0, 0] + 130*np.random.random(), -30, 30)
			self.arm_info[1] = arm_theta2 #np.clip(self.theta_bound[1, 0] + 89*np.random.random(), 0, 10)
			self.arm_info[2] = arm_theta3 #np.clip(self.theta_bound[2, 0] + 90*np.random.random(), -40, -10)
			self.EE = self.get_EE(self.arm_info)
			print('initial random point: ', self.point_info)
			print('initial random state: ', self.arm_info)

        # if vrep:
		# 	self.point_info = self.vrep_matrix.dot(self.point_info)
		self.EE = self.get_EE(self.arm_info)

		print(" \n -----------------reset--------------- \n")
		return self.get_state()




	def get_state(self):
		state_ = np.zeros(9)
		state_[:3] = self.arm_info
		state_[3] = self.EE[0] - self.arm_info[0] 
		state_[4] = self.EE[1] - self.arm_info[1]
		state_[5] = self.EE[2] - self.arm_info[2]
		state_[6] = np.linalg.norm(self.point_info - self.EE)
		state_[7] = 1 if self.grab_counter > 0 else 0
		state_[8] = 1 if self.constraints == True else 0
		return state_ 


	def rotation_matrix(self, theta, axis):
		R = np.zeros((4,4))
		theta_ = - theta*np.pi/180
		R[3, 3] = 1
		if axis == 0:   # axis x
			R[0, 0] = 1
			R[1, 1] = np.cos(theta_)
			R[1, 2] = -np.sin(theta_)
			R[2, 1] = np.sin(theta_)
			R[2, 2] = np.cos(theta_)
		elif axis == 1:   # axis y
			R[0, 0] = np.cos(theta_)
			R[0, 2] = -np.sin(theta_)
			R[1, 1] = 1
			R[2, 0] = np.sin(theta_)
			R[2, 2] = np.cos(theta_)
		elif axis == 2:
			R[0, 0] = np.cos(theta_)
			R[0, 1] = -np.sin(theta_)
			R[1, 0] = np.sin(theta_)
			R[1, 1] = np.sin(theta_)
			R[2, 2] = 1

		return R

	def translation_matrix(self, length, axis):
		T = np.zeros((4,4))
		T[:3,:3] = np.eye(3)
		T[3, 3] = 1
		if axis == 0:
			T[0, 3] = length
		elif axis == 1:
			T[1, 3] = length
		elif axis == 2:
			T[2, 3] = length

		return T

	def get_EE(self, arm_info):
		if vrep:
			R1 = self.rotation_matrix(self.arm_info[0], 1) # theta1
			R2 = self.rotation_matrix(self.arm_info[1], 0) # theta2 			
		else:
			R1 = self.rotation_matrix(self.arm_info[0]-90, 1) # theta1
			R2 = self.rotation_matrix(self.arm_info[1]-90, 0) # theta2 

		R3 = self.rotation_matrix(self.arm_info[2], 1) # theta3
		T1 = self.translation_matrix(-self.arm1l, 2) 
		T2 = self.translation_matrix(-self.arm2l, 2)


		EE_full = R1.dot(R2.dot(T1.dot(R3.dot(T2))))
		EE = EE_full[:3, -1]
		EE += np.array([0, 0, -5])

		if vrep:
			EE = self.vrep_matrix.dot(EE)

		return EE

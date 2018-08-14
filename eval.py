# Import libarary
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D, proj3d
from IPython import display
from itertools import product, combinations
import tensorflow as tf
import numpy as np
import os
import shutil
import argparse
import time
from ENV import Env

# Import custom libraries
from ENV import Env
from ddpg import Actor, Critic, Memory

# np.random.seed(1)
# tf.set_random_seed(1)

VAR_MIN = 0.001

# Parameters  
parser = argparse.ArgumentParser()
parser.add_argument('--memory', type=int, default=15000,
                     help='memory size for algorithm')
parser.add_argument('--batch_size', type=int, default=200,
                     help='minibatch size')
parser.add_argument('--lr_actor', type=float, default=1e-4,
                     help='learning rate for Actor network')
parser.add_argument('--lr_critic', type=float, default=1e-4,
                     help='learning rate for Critic network')
parser.add_argument('--target_update_a', type=int, default=1100,
                     help='update frequency for Actor target network')
parser.add_argument('--target_update_c', type=int, default=1000,
                     help='update frequency for Critic target network')
parser.add_argument('--gamma', type=float, default=0.9,
                     help='reward discount factor')
args = parser.parse_args()

MEMORY_CAPACITY = args.memory
BATCH_SIZE = args.batch_size
LR_A = args.lr_actor 
LR_C = args.lr_critic
REPLACE_ITER_A = args.target_update_a
REPLACE_ITER_C = args.target_update_c
GAMMA = args.gamma

env = Env()
STATE_DIM = env.state_dim
ACTION_DIM = env.action_dim
ACTION_BOUND = env.action_bound

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement=True
sess = tf.Session(config=config)

# Create actor and critic.
actor = Actor(sess, ACTION_DIM, ACTION_BOUND[1], LR_A, REPLACE_ITER_A)
critic = Critic(sess, STATE_DIM, ACTION_DIM, LR_C, GAMMA, REPLACE_ITER_C, actor.a, actor.a_)
actor.add_grad_to_graph(critic.a_grads)

M = Memory(MEMORY_CAPACITY, dims=2 * STATE_DIM + ACTION_DIM + 1)

saver = tf.train.Saver()
path = './checkpoints'

saver.restore(sess, tf.train.latest_checkpoint(path))

def eval():
    s = env.reset()
    while True:
        a = actor.choose_action(s)
        s_, r, done, collision = env.step(a)
        s = s_

def eval_count():
    test_num = 100
    succ_count = 0
    succ_point = []
    fail_point = []
    for i in range(test_num):
        s = env.reset()
        point = env.point_info.copy()
        print('env.point_info: ', env.point_info, 'point: ', point)
        for j in range(150):
            # if RENDER:
            #     env.render()
            a = actor.choose_action(s)
            s_, r, done, collision = env.step(a)
            
            if done :#& (not collision):
                succ_count += 1
                succ_point.append(point)
                break
            else:
                s = s_
    
        if not done:
            fail_point.append(point) 

        print('succ_point: ', succ_point)
        print('fail_point: ', fail_point)

    print('success cases/total test : %i/%i '%( succ_count , test_num))

def eval_plot():
	pos_x = []
	pos_y = []
	pos_z = []
	s = env.reset()
	pos = env.EE.copy()
	pos_x.append( env.EE.copy()[0])
	pos_y.append( env.EE.copy()[1])
	pos_z.append( env.EE.copy()[2])
	tar = env.point_info.copy() 
	for j in range(150):
		a = actor.choose_action(s)
		s_, r, done,collision = env.step(a)
		s = s_
		pos_x.append( env.EE.copy()[0])
		pos_y.append( env.EE.copy()[1])
		pos_z.append( env.EE.copy()[2])

	fig = plt.figure(figsize=(12, 8))
	ax = fig.gca(projection='3d')
	txt1 = ax.plot(pos_x, pos_y, pos_z, c='r' , linewidth=6, label='predicted')
	txt2 = ax.scatter(tar[0], tar[1], tar[2], c='k', s = 120, label='target')
	plt.title('End-effector trajectory in simulation', fontsize = 20)
	ax.set_xlabel('x [cm]')
	ax.set_ylabel('y [cm]')
	ax.set_zlabel('z [cm]')

	#draw cube
	r1 = [20, 54]
	r2 = [0,34]
	r3 = [-66, -32]

	for s, e in combinations(np.array(list(product(r1,r2,r3))), 2):
		if np.sum(np.abs(s-e)) == r1[1]-r1[0]:
			ax.plot3D(*zip(s,e), color="b" , linewidth=3, label = 'tabel')
	plt.show()


if __name__ == '__main__':
    eval_plot()
    # eval()
    # eval_count()
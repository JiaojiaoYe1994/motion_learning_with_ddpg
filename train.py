# coding: utf-8

# Import python libraries
import tensorflow as tf
import numpy as np
import os
import shutil
import argparse
import time

# Import custom libraries
from ENV import Env
from  ddpg import Actor, Critic, Memory

np.random.seed(1)
tf.set_random_seed(1)

VAR_MIN = 0.001

env = Env()
STATE_DIM = env.state_dim
ACTION_DIM = env.action_dim
ACTION_BOUND = env.action_bound

# Parameters  
parser = argparse.ArgumentParser()
parser.add_argument('--max_episodes', type=int, default=2000,
                     help='number of epochs')
parser.add_argument('--max_ep_steps', type=int, default=200,
                     help='number of steps per episode')
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

MAX_EPISODES = args.max_episodes
MAX_EP_STEPS = args.max_ep_steps
MEMORY_CAPACITY = args.memory
BATCH_SIZE = args.batch_size
LR_A = args.lr_actor 
LR_C = args.lr_critic
REPLACE_ITER_A = args.target_update_a
REPLACE_ITER_C = args.target_update_c
GAMMA = args.gamma

# txt file that saves the rewards and average rewards values during training
with open("reward/reward.txt","w+") as f_re:
    pass
with open("reward/reward_ave.txt","w+") as f_re:
    pass

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
sess.run(tf.global_variables_initializer())

def train():
    var = 15.  # control exploration
    re_sum = 0.
    reward_list = []
    reave_list = [] 
    board_writer = tf.summary.FileWriter(
        'summary'+ "/" + str(int(time.time())), 
        sess.graph
        )

    for ep in range(MAX_EPISODES):
        s = env.reset()
        ep_reward = 0

        for t in range(MAX_EP_STEPS):

            # Added exploration noise
            a = actor.choose_action(s)
            # add randomness to action selection for exploration
            a = np.clip(np.random.normal(a, var), *ACTION_BOUND)    
            s_, r, done, collision,  = env.step(a)
            M.store_transition(s, a, r, s_)

			# reply memory buffer
            if M.pointer > MEMORY_CAPACITY:
            	# decay the action randomness
                var = max([var*.9999, VAR_MIN])   
                b_M = M.sample(BATCH_SIZE)
                b_s = b_M[:, :STATE_DIM]
                b_a = b_M[:, STATE_DIM: STATE_DIM + ACTION_DIM]
                b_r = b_M[:, -STATE_DIM - 1: -STATE_DIM]
                b_s_ = b_M[:, -STATE_DIM:]

                # print('it: ', t, '--learn--')
                critic.learn(b_s, b_a, b_r, b_s_)
                actor.learn(b_s)
                summary1 = critic.summary
                board_writer.add_summary( summary1, t + ep*MAX_EP_STEPS)

            s = s_
            ep_reward += r

            if t == MAX_EP_STEPS-1 or done : 
                result = '| done' if done else '| ----'
                print('Ep:', ep,
                      result,
                      '| R: %i' % int(ep_reward),
                      '| Explore: %.3f' % var,
                      )
                break
               
        re_sum += ep_reward
        re_ave = re_sum / (ep+1)
        reward_list.append(ep_reward)
        reave_list.append(re_ave)
        
        if ep%50 ==0 :
            if os.path.isdir(path): shutil.rmtree(path)
            os.mkdir(path)
            # ckpt_path = os.path.join('./'+MODE[n_model], 'DDPG.ckpt')
            ckpt_path = os.path.join('./checkpoints', 'DDPG.ckpt')
            save_path = saver.save(sess, ckpt_path, write_meta_graph=False)
            print("\nSave Model %s\n" % save_path)

        with open("reward/reward.txt","a+") as f_re:
        	f_re.write("%.3f, " % (reward_list[-1]))
        with open("reward/reward_ave.txt","a+") as f_re:
        	f_re.write("%.3f, " % (reave_list[-1]))


def eval():
    s = env.reset()
    while True:
        a = actor.choose_action(s)
        s_, r, done, collision = env.step(a)
        s = s_

if __name__ == '__main__':
    print('### Begin train ###')
    time.sleep(1)
    train()
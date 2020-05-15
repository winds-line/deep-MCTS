from MCTS.MCTS import MCTS

from config.config import *
from utils.utils import *
from PIL import Image
from io import BytesIO
import base64
from network.avp import AvpNet
from network.vsp import VspNet
from game.game import Game
import tensorflow as tf
import numpy as np
import time
import os
from gym_torcs import TorcsEnv
import cv2


class Coach:
    def __init__(self, game, avp_net):
        self.train_interval = FLAGS.train_interval
        self.game = game
        self.avp_net = avp_net
        self.mcts = MCTS(self.game, self.avp_net)
        self.train_examples = []
        self.action_num = FLAGS.action_num
        self.step = 0

        self.last_state = None

        self.last_pi = 0
        self.last_a = 0


sess = tf.Session()
avpNet = AvpNet(sess)
vspNet = VspNet(sess)
sess.run(tf.global_variables_initializer())
game_ = Game(sess, avpNet, vspNet)
coach = Coach(game_, avpNet)

env = TorcsEnv(vision=True, throttle=False)
obs = env.reset()
steer_angle = 0.0
reward = 0.0
max_eps_steps = 10000
episode_count = 2000
for i in range(episode_count):
    if np.mod(i, 3) == 0:
        obs = env.reset(relaunch=True)   #relaunch TORCS every 3 episode because of the memory leak error
    else:
        obs = env.reset()

    for _ in range(max_eps_steps):
        coach.step += 1
        image = obs.img
        image = np.reshape(image, (64,  64, 3))

        image = process_img(image)/255

        pos = (0, 0)
        w = coach.game.format_steer_angle(steer_angle)
        r = reward
        if coach.step > 1:
            coach.train_examples.append([coach.last_state, coach.last_pi, r, coach.last_a])
            if len(coach.train_examples) >= coach.train_interval:
                coach.game.train_net(coach.train_examples)
                coach.train_examples = []

        # store current state waiting for next state to get reward
        state = np.reshape(image, (FLAGS.image_size,))
        coach.last_state = state

        coach.mcts = MCTS(coach.game, coach.avp_net)
        pi = coach.mcts.get_action_prob(state, w, 1)

        action = np.argmax(pi)
        np.set_printoptions(precision=4)
        pi = np.array(pi)
        coach.last_a = action
        coach.last_pi = pi
        steer_angle = -1.0 + action/(coach.action_num-1)*2.0
        a_t = np.zeros((1, ))
        a_t[0] = steer_angle
        obs, reward, done, _ = env.step(a_t)
        if reward < 0:
            reward = -1
        else:
            reward = 0.1
        print('eps', i, 'step', coach.step, 'r', r, 'action', a_t, 'pi', pi)
        if done:
            print('reset！'*10)
            obs = env.reset()

from MCTS.MCTS import MCTS
import socketio
import eventlet.wsgi
from flask import Flask
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


class Coach:
    def __init__(self, game, avp_net):
        self.train_interval = FLAGS.train_interval
        self.game = game
        self.avp_net = avp_net
        self.mcts = MCTS(self.game, self.avp_net)
        self.train_examples = []
        self.action_num = FLAGS.action_num
        self.step = 0

        self.avoidance_step = 0
        self.max_step = 0
        self.max_step_list = []

        self.last_state = None
        self.last_o = 0
        self.last_w = 0
        self.last_d = 0
        self.last_pi = 0
        self.last_a = 0

        self.last_steer = 0.0
        self.steer_punish = 0.0


sess = tf.Session()
avpNet = AvpNet(sess)
vspNet = VspNet(sess)
game_ = Game(sess, avpNet, vspNet)
coach = Coach(game_, avpNet)

# start socketio
sio = socketio.Server()
app = Flask(__name__)


@sio.on('telemetry')
def execute_action(sid, data):
    coach.step += 1
    coach.avoidance_step += 1
    image = data['image']
    image = Image.open(BytesIO(base64.b64decode(image)))
    image_p = np.asarray(image)
    image = process_img(image_p)/255

    steer_angle = float(data['steering_angle'])
    reward_flag = float(data['reward'])
    # reward_flag = 1 means the collision occurs, otherwise, reward_flag = 0
    if reward_flag > 0.5:
        reward = -1
    else:
        reward = 0.1
    if reward < -0.1:
        if coach.avoidance_step > coach.max_step and coach.step > 10000:
            coach.max_step = coach.avoidance_step
            coach.game.save_ckpt()
            coach.max_step_list.append(coach.max_step)
            np.save('max_step.npy', coach.max_step_list)
        coach.avoidance_step = 0

    steer_angle /= 25
    w = coach.game.format_steer_angle(steer_angle)

    r = reward + coach.steer_punish
    if coach.step > 1:
        coach.train_examples.append([coach.last_state, coach.last_pi, r, coach.last_a])
        if len(coach.train_examples) >= coach.train_interval:
            coach.game.train_net(coach.train_examples)
            coach.train_examples = []

    # store current state waiting for next state to get reward
    state = image
    coach.last_state = state

    coach.mcts = MCTS(coach.game, coach.avp_net)
    pi = coach.mcts.get_action_prob(state, w, FLAGS.tau)

    action = np.random.choice(len(pi), p=pi)
    np.set_printoptions(precision=4)
    pi = np.array(pi)
    print('step', coach.step, 'r', r, 'action', action, 'pi', pi)
    coach.last_a = action
    coach.last_pi = pi
    steer_angle = -1.0 + action/(coach.action_num-1)*2.0
    coach.last_steer = steer_angle
    throttle = 1.0
    send_control(steer_angle, throttle)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit(
        "steer",
        data={
            'steering_angle': steering_angle.__str__(),
            'throttle': throttle.__str__(),
            'reset': '0'
        },
        skip_sid=True)


app = socketio.Middleware(sio, app)
eventlet.wsgi.server(eventlet.listen(('', 4569)), app)

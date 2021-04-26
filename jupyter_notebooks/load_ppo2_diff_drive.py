import numpy as np

import gym
from gym import Env
from gym.spaces import Box
from gym.utils import seeding

import math


class DifferentialDriveEnv(Env):
  """Custom Environment that follows gym interface"""
  # metadata = {'render.modes': ['human']}
  metadata = {'render.modes': ['console']}

  def __init__(self, L, r, delta_t = 0.01, init_position=None, goal_position=[0,0], min_action=-5, max_action=5, min_position=[-10,-10], max_position=[10,10]):
    super(DifferentialDriveEnv, self).__init__()

    #Define model parameters
    self.L = L 
    self.r = r

    self.delta_t = delta_t

    # Define action and observation space
    # They must be gym.spaces objects
    self.min_action = min_action
    self.max_action = max_action
    self.min_position = min_position
    self.max_position = max_position

    self.min_orientation = -math.pi
    self.max_orientation = math.pi
    
    self.init_position = init_position
    self.goal_position = goal_position

    # self.goal_velocity = goal_velocity
    # self.goal_orientation = goal_orientation

    self.max_duration = 500
    
    self.low_state = np.array(
        self.min_position+[self.min_orientation], dtype=np.float32
    )

    self.high_state = np.array(
        self.max_position+[self.max_orientation], dtype=np.float32
    )

    self.viewer = None

    self.action_space = Box(
        low=self.min_action,
        high=self.max_action,
        #low=-1,
        #high=1,
        shape=(2,),
        dtype=np.float32
    )

    self.observation_space = Box(
        low=self.low_state,
        high=self.high_state,
        shape=(3,),
        dtype=np.float32
    )

    self.seed()
    self.reset()

  def seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]

  def reset(self):
    # Reset the state of the environment to an initial state

    if self.init_position is None:
      self.state = np.array([self.np_random.uniform(low=-3, high=-2), self.np_random.uniform(low=-2, high=-1), -math.pi/2])
      # self.state = np.array([self.np_random.uniform(low=self.min_position[0], high=self.max_position[0]), self.np_random.uniform(low=self.min_position[0], high=self.max_position[0]), math.pi/2])
    elif isinstance(self.init_position, list):
      if len(self.init_position) == 3:
        self.state = np.array(self.init_position)
      else:
        raise Exception("Initial position must be size 3: [x, y, theta]")
    else:
      raise Exception("Initial position must be a list: [x, y, theta]")
    self.max_duration = 500

    return np.array(self.state)

  def render(self, mode='console', close=False):
    if mode is 'console':
      print("========================================================")
      print(">> Pos: x = ",self.state[0],"; y = ",self.state[1])
      print(">> Ori: ",self.state[2])
      print("========================================================")

  def step(self, action):
    x = self.state[0]
    y = self.state[1]
    theta = self.state[2]

    v = (action[1] + action[0]) * self.r / 2
    w = (action[1] - action[0]) * self.r / self.L

    x = x + v * self.delta_t * math.cos(theta)
    y = y + v * self.delta_t * math.sin(theta)
    theta = theta + w * self.delta_t

    threshold = 0.01
    
    done = bool(
        np.linalg.norm(np.array(self.goal_position)-np.array([x, y])) <= threshold
    )

    reward = 0
    if done:
        reward = 100.0
    # reward -= math.pow(action[0], 2) * 0.1
    reward -= float(np.linalg.norm(np.array(self.goal_position)-np.array([x, y]))/10)

    if self.max_duration <= 0: 
            done = True
    else:
            done = False

    info = {}

    self.state = np.array([x, y, theta])

    self.max_duration -= 1

    return self.state, reward, done, info

  def close(self):
    pass

from stable_baselines.common.env_checker import check_env
axle_length = 50
wheel_radius = 15
#axle_length = 0.5
#wheel_radius = 0.15
#init_pose = [-1,1,-np.pi/4]
init_pose = [-2.38,-1.075,-np.pi/2] #ok
#init_pose = [-2.38,-1.075,np.pi/4] #noooo swirl
#init_pose = [-2.38,-1.075,3*np.pi/4] #converges in a far position [-4,-2]
env = DifferentialDriveEnv(L=axle_length, r=wheel_radius,init_position=init_pose)
print("INITPOSE_after env created {}".format(env.init_position))
#env = DifferentialDriveEnv(L=axle_length, r=wheel_radius)
# If the environment don't follow the interface, an error will be thrown
check_env(env, warn=True)

print("Env checked!")

from stable_baselines import PPO2

model = PPO2.load("ppo2_diff_drive") #paolo trained model
#model = PPO2.load("ppo2_diff_drive_fra_test") # my trained model with paolo conf
#model = PPO2.load("ppo2_diff_drive_fra_test_meters") # my trained model with paolo conf
obs = env.reset()
print("INITPOSE_after env reset {}".format(env.init_position))
print("OBS after resest {}".format(obs))
max_steps = 1500
x_values = []
y_values = []
theta_values = []
for _ in range(max_steps):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    x_values.append(obs[0])
    y_values.append(obs[1])
    theta_values.append(obs[2])
    #print("Current x: {} current y: {}".format(obs[0],obs[1]))
    #env.render(mode = 'console')

import matplotlib.pyplot as plt
print("Starting point: x:{}, y:{}".format(x_values[0],y_values[0]))
print("End point: x:{}, y:{}".format(x_values[-1],y_values[-1]))

plt.scatter(x_values, y_values,color='blue')
plt.scatter(x_values[0],y_values[0],color='purple')
plt.scatter(x_values[-1],y_values[-1],color='green')
plt.scatter(0,0,color='red',marker='x')
plt.show()


   
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

  def __init__(self, L, r, delta_t = 0.01, goal_position=[0,0], min_action=-10, max_action=10, min_position=[-100,-100], max_position=[100,100]):
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
    
    self.goal_position = goal_position
    # self.goal_velocity = goal_velocity
    # self.goal_orientation = goal_orientation

    
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
    self.state = np.array([self.np_random.uniform(low=self.min_position[0], high=self.max_position[0]), self.np_random.uniform(low=self.min_position[0], high=self.max_position[0]), math.pi/2])
    return np.array(self.state)

  def render(self, mode='human', close=False):
    # Render the environment to the screen
    pass

  def step(self, action):
    x = self.state[0]
    y = self.state[1]
    theta = self.state[2]

    v = (action[1] + action[0]) * self.r / 2
    w = (action[1] - action[0]) * self.r / self.L

    x = x + v * self.delta_t * math.cos(theta)
    y = y + v * self.delta_t * math.sin(theta)
    theta_next = theta + w * self.delta_t

    done = bool(
        [x, y] == self.goal_position
    )

    reward = 0
    if done:
        reward = 100.0
    reward -= math.pow(action[0], 2) * 0.1

    self.state = np.array([x, y, theta])
    return self.state, reward, done, {}
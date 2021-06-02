import numpy as np
import gym
from gym import Env
from gym.spaces import Box
from gym.utils import seeding
import math

#ppo2_model_name = "ppo2_meters_redesigned_1"
#axle_length = 0.5
#wheel_radius = 0.15

class DifferentialDriveEnvV1(Env):
  """Custom Environment that follows gym interface"""
  metadata = {'render.modes': ['console']}

  def __init__(self, L, r, delta_t = 0.01, init_position=None, goal_position=[0,0], min_action=-3, max_action=3, min_position=[-1,-1], max_position=[1,1]):
    super(DifferentialDriveEnvV1, self).__init__()

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
    
    #self.init_position = init_position
    self.init_position = self.set_init_position(init_position)
    
    self.goal_position = goal_position

    # self.goal_velocity = goal_velocity
    # self.goal_orientation = goal_orientation
    self.goal_reached_count = 0

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
  
  def set_init_position(self,init_position):
    if init_position == None:
      self.init_position = None
      return
    if isinstance(init_position, list):
      if len(init_position) == 3:
        self.init_position = init_position
      else:
        raise Exception("Initial position must be size 3: [x, y, theta]")
    else:
      raise Exception("Initial position must be a list: [x, y, theta]")

  def reset(self):
    # Reset the state of the environment to an initial state

    if self.init_position is None:
       self.state = np.array([self.np_random.uniform(low=-0.1, high=0.1), self.np_random.uniform(low=-0.5, high=0.5), -math.pi/2+self.np_random.uniform(low=-0.01, high=0.01)])
    else:
       self.state = np.array(self.init_position)
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

    v = (action[1] + action[0]) * self.r / 2      #max vlin = 2 * 3 *0.15 /2 = 3*0.15 = 0.45 m/s (was 0.75 wmax = 5)
    w = (action[1] - action[0]) * self.r / self.L #max omega = 3 - (-3) * 0.15 /0.5 = 0.9 /0.5  1,8 rad/s ()

    x = x + v * self.delta_t * math.cos(theta)
    y = y + v * self.delta_t * math.sin(theta)
    theta = theta + w * self.delta_t

    threshold = 0.1
    distance_to_target = np.linalg.norm(np.array(self.goal_position)-np.array([x, y]))

    goal_reached = bool(distance_to_target <= threshold)
    too_far = bool(distance_to_target > 0.1)

    reward = 0
    if goal_reached:
      reward = 100.0
      self.goal_reached_count += 1
    reward -= distance_to_target *0.1
    done = goal_reached or bool(self.max_duration <= 0)
    info = {}

    self.state = np.array([x, y, theta])

    self.max_duration -= 1

    return self.state, reward, done, info

  def close(self):
    pass
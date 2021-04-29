 
import numpy as np

import gym
from gym import Env
from gym.spaces import Box
from gym.utils import seeding

import math

ppo2_model_name = "ppo2_meters_redesigned_1"
axle_length = 0.5
wheel_radius = 0.15
class DifferentialDriveEnv(Env):
  """Custom Environment that follows gym interface"""
  metadata = {'render.modes': ['console']}

  def __init__(self, L, r, delta_t = 0.01, init_position=None, goal_position=[0,0], min_action=-3, max_action=3, min_position=[-1,-1], max_position=[1,1]):
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

  def reset(self):
    # Reset the state of the environment to an initial state

    if self.init_position is None:
       self.state = np.array([self.np_random.uniform(low=-0.1, high=0.1), self.np_random.uniform(low=-0.5, high=0.5), -math.pi/2+self.np_random.uniform(low=-0.01, high=0.01)])
    elif isinstance(self.init_position, list):
      if len(self.init_position) == 3:
        self.state = np.array(self.init_position)
      else:
        raise Exception("Initial position must be size 3: [x, y, theta]")
    else:
      raise Exception("Initial position must be a list: [x, y, theta]")
    #self.max_duration = 500
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

from stable_baselines.common.env_checker import check_env
from stable_baselines import PPO2
def check_diff_drive_env():
    env = DifferentialDriveEnv(L=axle_length, r=wheel_radius)
    # If the environment don't follow the interface, an error will be thrown
    check_env(env, warn=True)
    print("Env checked!")

def load_and_run_model(model_name,n_steps,init_pose=None):
    env = None
    model = None
    model = PPO2.load(model_name)
    env = DifferentialDriveEnv(L=axle_length, r=wheel_radius,init_position=init_pose)
    print("INITPOSE_after env created {}".format(env.init_position))
    obs = env.reset()
    print("INITPOSE_after env reset {}".format(env.init_position))
    print("OBS after resest {}".format(obs))
    obs_list = [obs]
    action_list = []
    for _ in range(n_steps):
        action, _states = model.predict(obs,deterministic=True)
        obs, rewards, done, info = env.step(action)
        action_list.append(action)
        obs_list.append(obs)
        if done:
            print("Arrived in {} steps".format(len(obs_list)))
            break  
        #print("Current x: {} current y: {}".format(obs[0],obs[1]))
        #env.render(mode = 'console')
    return obs_list, action_list

import matplotlib.pyplot as plt
    
def robot_linear_velocity(u_l,u_r,L=axle_length, r=wheel_radius):
    v = (u_l + u_r)* w_radius/2
    w = (u_r - u_l)* w_radius/a_length
    return v, w

def show_rl_trajectory(obs_list,act_list):
    x_values = list(map(lambda obs: obs[0], obs_list))
    y_values = list(map(lambda obs: obs[1], obs_list))

    theta_values = list(map(lambda obs: obs[2], obs_list))

    u_l_values = list(map(lambda act: act[0], act_list))
    u_l_values = list(map(lambda act: act[1], act_list))
    
    v_values = []
    w_values = []

    for a in act_list:
        v,w = robot_linear_velocity(a[0],a[1])
        v_values.append(v)
        w_values.append(w)

    print("Starting point: x:{}, y:{} -PURPLE-".format(x_values[0],y_values[0]))
    print("End point: x:{}, y:{} -GREEN-".format(x_values[-1],y_values[-1]))
    def on_close(event):
        print('Closed Figure!')
        
    fig, ax = plt.subplots(4)
    fig.canvas.mpl_connect('close_event', on_close)
    ax[0].scatter(x_values, y_values,color='blue')
    ax[0].set_title("Path")
    ax[0].scatter(x_values[0],y_values[0],color='purple')
    ax[0].scatter(x_values[-1],y_values[-1],color='green')
    ax[0].scatter(0,0,color='red',marker='x')
    ax[0].axis("equal")
    ax[0].grid()

    ax[1].plot(range(len(theta_values)), theta_values)
    ax[1].set_title("Orientation")
    ax[1].grid()

    ax[2].plot(range(len(v_values)), v_values)
    ax[2].set_title("Linear velocity")
    ax[2].grid()

    ax[3].plot(range(len(w_values)), w_values)
    ax[3].set_title("Angular velocity")
    ax[3].grid()

    plt.show()

def robot_linear_velocity(u_l,u_r,L=axle_length, r=wheel_radius):
    v = (u_l + u_r)* r/2
    w = (u_r - u_l)* r/L
    return v, w

def main():
    check_diff_drive_env()
    #init_pose  = [-0.05, -0.25, -np.pi/2]
    init_pose  = [0.12, -0.25, -np.pi/2]
    obss, actions = load_and_run_model(ppo2_model_name,1000,init_pose)
    print("I have {} observations and {} actions ".format(len(obss),len(actions)))
    show_rl_trajectory(obss,actions)

if __name__ == "__main__":
    main()
import numpy as np

import gym
from gym import Env
from gym.spaces import Box
from gym.utils import seeding

import math

#conf cm paolo
min_pos_val = -10
max_pos_val = 10
min_act_val = -5
max_act_val = 5
reward_threshold = 0.1

#right
#min_init_x = 2
#max_init_x = 3
#left
min_init_x = -3
max_init_x = -2
min_init_y = -2
max_init_y = -1

axle_length = 50
wheel_radius = 15

#conf meters fra 1

#min_pos_val = -5
#max_pos_val = 5
#min_act_val = -5
#max_act_val = 5
#reward_threshold = 0.5
#reward_threshold = 0.1

#right
#min_init_x = -3
#max_init_x = -2

#larger
#min_init_x = -3
#max_init_x = 3

#left
#min_init_x = 2
#max_init_x = 3

#min_init_y = -2
#max_init_y = -1

#axle_length = 0.5
#wheel_radius = 0.15


#conf maters fra 2
#min_pos_val = -0.10
#max_pos_val = 0.10
#min_act_val = -5
#max_act_val = 5
#reward_threshold = 0.001

#right
#min_init_x = 0.02
#max_init_x = 0.03
#left
#min_init_x = -0.03
#max_init_x = -0.02
#min_init_y = -0.02
#max_init_y = -0.01

#axle_length = 0.50
#wheel_radius = 0.15

#init poses for paolo cm and conf meters 1 by fra
#init_robot_pose = {'x': -0.5, 'y': 0.0, 'theta': np.pi/2} #target on the side robot along y axis
#init_robot_pose = {'x': -1, 'y': 0.0, 'theta': 0} #ok target in front robot along y axis
#init_robot_pose = {'x': -1, 'y': 0.0, 'theta': -np.pi} #ok target behind robot along y axis
#init_robot_pose = {'x': -1, 'y': 1, 'theta': -np.pi/4} # target in front
#init_robot_pose = {'x': -1, 'y': 1, 'theta': -np.pi/4} # target in front cm
#init_robot_pose = {'x': -2.38, 'y': -1.075, 'theta': -np.pi/2} 
init_robot_pose = {'x': -2.38, 'y': -1.5, 'theta': -np.pi/2} 
#init_robot_pose = {'x': 2.38, 'y': -1.5, 'theta': -np.pi/2} 

#init_robot_pose = {'x':-0.0238,'y':-0.0107,'theta':-np.pi/2}
class DifferentialDriveEnv(Env):
  """Custom Environment that follows gym interface"""
  # metadata = {'render.modes': ['human']}
  metadata = {'render.modes': ['console']}

  def __init__(self, L, r, delta_t = 0.01, init_position=None, goal_position=[0,0], min_action=min_act_val, max_action=max_act_val, min_position=[min_pos_val,min_pos_val], max_position=[max_pos_val,max_pos_val]):
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
      self.state = np.array([self.np_random.uniform(low=min_init_x, high=max_init_x), self.np_random.uniform(low=min_init_y, high=max_init_y), -math.pi/2])
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

    threshold = reward_threshold
    
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
    obs_list = []
    for _ in range(n_steps):
        action, _states = model.predict(obs,deterministic=True)
        obs, rewards, dones, info = env.step(action)
        obs_list.append(obs)
        if (np.linalg.norm(np.array(env.goal_position)-np.array([obs[0], obs[1]])))<=reward_threshold:
            print("Arrived")
            break
            
        #print("Current x: {} current y: {}".format(obs[0],obs[1]))
        #env.render(mode = 'console')
    return obs_list

import matplotlib.pyplot as plt
    
def show_rl_trajectory(obs_list):
    x_values = list(map(lambda obs: obs[0], obs_list))
    y_values = list(map(lambda obs: obs[1], obs_list))
    theta_values = list(map(lambda obs: obs[2], obs_list))
    print("Starting point: x:{}, y:{} -PURPLE-".format(x_values[0],y_values[0]))
    print("End point: x:{}, y:{} -GREEN-".format(x_values[-1],y_values[-1]))
    def on_close(event):
        print('Closed Figure!')
        
    fig = plt.figure()
    fig.canvas.mpl_connect('close_event', on_close)
    plt.scatter(x_values, y_values,color='blue')
    plt.scatter(x_values[0],y_values[0],color='purple')
    plt.scatter(x_values[-1],y_values[-1],color='green')
    plt.scatter(0,0,color='red',marker='x')
    plt.axis("equal")
    plt.grid()
    plt.show()

def main():
    ppo2_model = "ppo2_diff_drive" #paolo trained model
    #ppo2_model = "ppo2_diff_drive_cm_bottom_right"
    #ppo2_model = "ppo2_diff_drive_fra_test" # my trained model with paolo conf
    #ppo2_model = "ppo2_diff_drive_fra_test_meters" #my model in meters 
    #ppo2_model = "ppo2_diff_drive_fra_test_meters_tr0_1" #my model in meters
    #ppo2_model = "ppo2_diff_drive_fra_test_meters_tr0_1bis" #my model in meters
    #ppo2_model="ppo2_diff_drive_fra_test_meters_tr0_5_init_bottom_right" #my model meters starts at bottom right
    #ppo2_model = "ppo2_diff_drive_from_paolo_meter_bl"
    #ppo2_model = "ppo2_diff_drive_from_paolo_meter_bl_rwscaled"
    #ppo2_model = "ppo2_diff_drive_from_paolo_meter_bl_rwscaled_correct"
    #ppo2_model = "ppo2_diff_drive_from_paolo_meter_bl_rwscaled_correct_th001_2"
    #ppo2_model = "ppo2_diff_drive_from_paolo_meter_bl_rwscaled_correct_th001_stop_at_goal"
    #ppo2_model = "ppo2_diff_drive_from_paolo_meter_bl_rwscaled_correct_th001_stop_at_goal_and_too_far"
    check_diff_drive_env()
    init_pose = list(init_robot_pose.values())
    print("INIT POSE!!!!!!!!!!!!! {}".format(init_pose))
    obss = load_and_run_model(ppo2_model,500,init_pose)
    show_rl_trajectory(obss)

if __name__ == "__main__":
    main()



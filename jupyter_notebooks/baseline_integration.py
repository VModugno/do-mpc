 
import numpy as np

import gym
from gym import Env
from gym.spaces import Box
from gym.utils import seeding
import matplotlib.pyplot as plt
from casadi import *
from casadi.tools import *
from differential_drive_env_v2 import DifferentialDriveEnvV2
from differential_drive_env_v1 import DifferentialDriveEnvV1

import math

ppo2_model_name = "ppo2_meters_redesigned_1"
axle_length = 0.5
wheel_radius = 0.15

from stable_baselines.common.env_checker import check_env
from stable_baselines import PPO2
def check_diff_drive_env(env_class):
    env = env_class(L=axle_length, r=wheel_radius)
    # If the environment don't follow the interface, an error will be thrown
    check_env(env, warn=True)
    print("Env checked!")

def setup_model_execution_on_env(model_name,a_length,w_radius,init_pos,env_class,model_wrapper_class=None,model_wrapper_params={}):
  model = PPO2.load(model_name)
  if model_wrapper_class:
    model = model_wrapper_class(model, **model_wrapper_params)
  env = env_class(L=a_length, r=w_radius, init_position=init_pos)
  return {'policy':model,'env':env}

def run_model(model,env,n_steps,init_pos):
  print("run_model() init_pose param = {}".format(init_pos))
  env.set_init_position(init_pos)
  obs = env.reset()
  print("env.init_position after env reset() = {}".format(env.init_position))
  print("obs returned by env.reset() = {}".format(obs))
  obs_list = [obs]
  action_list = []
  for i in range(n_steps):
      action, _states = model.predict(obs, deterministic=False)
      #print("Run model iteration i {}, state {}, pred_action {}".format(i, obs, action))
      obs, rewards, done, info = env.step(action)
      action_list.append(action)
      obs_list.append(obs)
      if done:
          print("Arrived in {} steps".format(len(obs_list)))
          break  
      #print("Current x: {} current y: {}".format(obs[0],obs[1]))
      #env.render(mode = 'console')
  return obs_list, action_list

def load_and_run_model(model_name,n_steps,a_length,w_radius,env_class,init_pose=None,model_wrapper_class=None, model_wrapper_params={}):
    print("LOAD AND RUN init_pose before setup {}".format(init_pose))
    ppo2_model_env_dict = setup_model_execution_on_env(model_name,a_length,w_radius,init_pose,env_class,model_wrapper_class,model_wrapper_params)
    print("LOAD AND RUN init_pose before run model {}".format(init_pose))
    obs_list,action_list = run_model(ppo2_model_env_dict['policy'],ppo2_model_env_dict['env'],n_steps,init_pose)
    return obs_list, action_list


def compare_trajectory_with_predicted_trajectory(gt_path,gt_actions,gt_final_state,model,env):
  assert isinstance(gt_path, list) and isinstance(gt_actions, list)
  assert len(gt_path)==len(gt_actions), 'gt_path and gt_actions must have the same length: len of gt_actions is: {}, len of gt_path is: {}'.format(len(gt_path),len(gt_actions))
  complete_path = list(gt_path)
  complete_path.append(gt_final_state)
  comparison_list = []
  for i in range(len(complete_path)-1):
    obs_list, action_list = run_model(model,env,1,complete_path[i])
    c_info = {'state':obs_list[0].tolist(),
              'gt_next_state':complete_path[i+1],
              'gt_action':gt_actions[i],
              'predicted_action':action_list[0].tolist(),
              'next_predicted_state':obs_list[1].tolist()
    }
    c_info['cartesian_error'] = sqrt((c_info['next_predicted_state'][0]-c_info['gt_next_state'][0])**2+
                                     (c_info['next_predicted_state'][1]-c_info['gt_next_state'][1])**2)
    c_info['orientation_error'] = atan2(sin(c_info['next_predicted_state'][2]-c_info['gt_next_state'][2]),
                                        cos(c_info['next_predicted_state'][2]-c_info['gt_next_state'][2]))
    comparison_list.append(c_info)
  return comparison_list
    
def show_policy_comparison_vs_gt(comparison_list):
  gt_x_values = list(map(lambda c: c['gt_next_state'][0], comparison_list))
  gt_y_values = list(map(lambda c:c['gt_next_state'][1],comparison_list))
  predicted_x_values = list(map(lambda c:c['next_predicted_state'][0],comparison_list))
  predicted_y_values = list(map(lambda c:c['next_predicted_state'][1],comparison_list))
  fig, ax = plt.subplots(3,figsize=(15,15))
  plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=2.0, hspace=0.5)
  ax[0].scatter(gt_x_values, gt_y_values,color='blue')
  ax[0].set_title("Path")
  ax[0].scatter(predicted_x_values,predicted_y_values,color='red')
  ax[0].axis("equal")
  
  cartesian_errors = list(map(lambda comp:comp['cartesian_error'],comparison_list))
  ax[1].plot(range(len(cartesian_errors)), cartesian_errors)
  ax[1].set_title("Cartesian error")

  orientation_errors = list(map(lambda comp:comp['orientation_error'],comparison_list))
  ax[2].plot(range(len(orientation_errors)), orientation_errors,color='blue')
  ax[2].set_title("Orientation error")
  
  plt.show()

def from_commands_to_robot_velocity(u_l,u_r,L, r):
    v = (u_l + u_r)* r/2
    w = (u_r - u_l)* r/L
    return v, w

def from_robot_velocity_to_commands(v,w,L,r):
    #From the previous equations: summing we have the expression of 2 u_r
    #Subtracting we have the espression of u_l
    u_r = (2*v+L*w)/(2*r)
    u_l = (2*v-L*w)/(2*r) 
    return u_l,u_r

def compute_trajectories_x_eq_y_x_eq_min_y(n_samples,ax_len1,ax_len2,wheel_r,init_pos=[0,0],abs_max_ul=3,abs_max_ur=3,delta_t=0.01):
    max_robot_lin_vel_len1 = from_commands_to_robot_velocity(3,3,L=ax_len1,r=wheel_r)[0]
    max_robot_lin_vel_len2 = from_commands_to_robot_velocity(3,3,L=ax_len2,r=wheel_r)[0]
    max_robot_lin_vel = np.min([max_robot_lin_vel_len1,max_robot_lin_vel_len2])
    max_ds = delta_t * max_robot_lin_vel
    path_x_eq_y = [[init_pos[0],init_pos[1],np.pi/4]]
    controls_x_eq_y = []
    path_x_eq_min_y =[[init_pos[0],init_pos[1],3*np.pi/4]]
    controls_x_eq_min_y = []
    curr_pos = init_pos
    for i in range(n_samples):
      #curr_ds = np.random.uniform(low=0.0,high=max_ds)
      curr_ds = max_ds/2
      curr_lin_vel = curr_ds/delta_t
      curr_commands_l1 = from_robot_velocity_to_commands(curr_lin_vel,0.0,L=ax_len1,r=wheel_r)
      curr_dx = curr_ds/np.sqrt(2)
      curr_pos = [curr_pos[0]+curr_dx, curr_pos[1]+curr_dx]
      path_x_eq_y.append([curr_pos[0],curr_pos[1],np.pi/4])
      controls_x_eq_y.append(curr_commands_l1)
      path_x_eq_min_y.append([-curr_pos[0],curr_pos[1],3*np.pi/4])
      curr_commands_l2 = from_robot_velocity_to_commands(curr_lin_vel,0.0,L=ax_len2,r=wheel_r)
      controls_x_eq_min_y.append(curr_commands_l2)
    return [{'path':path_x_eq_y,'actions':controls_x_eq_y},{'path':path_x_eq_min_y,'actions':controls_x_eq_min_y}]

def diffdrive_evolution_from_commands(axle_l, wheel_r,init_pose,commands,delta_t=0.01):
  x = init_pose['x']
  y = init_pose['y']
  theta = init_pose['theta']
  state_seq = [[x,y,theta]]
  for c in commands:
    v,w = from_commands_to_robot_velocity(c[0],c[1],axle_l,wheel_r)
    x = x + v * delta_t * cos(theta)
    y = y + v * delta_t * sin(theta)
    theta = theta + w * delta_t
    state_seq.append([x,y,theta])
    #print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!! c is {} state is {}".format(c,[x,y,theta]))
  return state_seq
      
def compute_cost_of_tracking_along_the_horizon(cost_expression,mpc,init_robot_pose,ref_trajectories,command_values):
  
    n_coll_pts = mpc.n_total_coll_points
    assert n_coll_pts == 0, "This test is only for discrete opt problems: num of collocation poits is {} must be 0".format(n_coll_pts)
    #_x is a SX_sym of dim horizon_steps * num_scenarios * num_of_collocation_points
    horizon_steps = len(mpc.opt_x['_x'])-1
    num_scenarios = len(mpc.opt_x['_x'][0])
    assert len(ref_trajectories) == num_scenarios, "The reference trajectiories number {} is different from the scenarios number {}".format(len(ref_trajectories),num_scenarios)
    assert len(command_values)>= horizon_steps
    assert mpc.n_robust <= 1, "This test currently assumes that n_robust is equal to 1 (this inpacts on the x symbols retrieving)"
    print("Obj expression {} ".format(cost_expression))
    print("Num of coll points {} ".format(n_coll_pts))
    cost_value = cost_expression
    for s in range(num_scenarios):
      print("\nXXXXXXXXXXX SCENARIO: {} [axle length = {} wheel radius = {}] XXXXXXXXXXXXXXXXXXXXX".format(s,ref_trajectories[s]['L'],ref_trajectories[s]['r']))
      ref_obss = ref_trajectories[s]['path']
      ref_actions = ref_trajectories[s]['actions']
      s_L = ref_trajectories[s]['L']
      s_r = ref_trajectories[s]['r']
      ref_path_along_horizon = ref_obss[0:horizon_steps+1]
      # PATH of the state along the horizon: replace this section with the computation of the state with command_values
      #epsilon_path = 0.0005
      #state_along_horizon = list(map(lambda st: [0,(st[1]-epsilon_path),init_robot_pose['theta']],ref_path_along_horizon))
      #state_along_horizon[0] = list(init_robot_pose.values())
      #####
      state_along_horizon = diffdrive_evolution_from_commands(s_L,s_r,init_robot_pose,command_values)
      print("Path to track along the horizon: {}".format(ref_path_along_horizon))
      print("State along the horizon {}".format(state_along_horizon))
      print("Opt TVP {}".format(mpc.opt_p["_tvp"]))
      print("Opt X of _x variables (state): {}".format(mpc.opt_x['_x']))
      print("Opt X of _x variables along horizon, scenario {}, coll {}: {}".format(s,n_coll_pts, mpc.opt_x['_x',:,s,0]))
      x_x_horizon = mpc.opt_x['_x',:,s,n_coll_pts,'x']
      x_y_horizon = mpc.opt_x['_x',:,s,n_coll_pts,'y']
      x_theta_horizon = mpc.opt_x['_x',:,s,n_coll_pts,'theta']
      print("Opt X of _x.x along horizon, scenario {}, coll {}: {}".format(s,n_coll_pts,x_x_horizon))
      print("Opt X of _x.y along horizon, scenario {}, coll {}: {}".format(s,n_coll_pts,x_y_horizon))
      if mpc.scenario_tvp :
        tvp_horizon_x_ref = mpc.opt_p['_tvp',s,:,'x_ref']
        tvp_horizon_y_ref = mpc.opt_p['_tvp',s,:,'y_ref']
        tvp_horizon_theta_ref = mpc.opt_p['_tvp',s,:,'theta_ref']
      else:
        tvp_horizon_x_ref = mpc.opt_p['_tvp',:,'x_ref']
        tvp_horizon_y_ref = mpc.opt_p['_tvp',:,'y_ref']
        tvp_horizon_theta_ref = mpc.opt_p['_tvp',:,'theta_ref']
      print("Opt TVP of x_ref along the horizon: {}".format(tvp_horizon_x_ref))
      print("Opt TVP of y_ref along the horizon: {}".format(tvp_horizon_y_ref))
      
      for i in range(0,horizon_steps+1):
          cost_value = substitute(cost_value,tvp_horizon_x_ref[i],ref_path_along_horizon[i][0])
          cost_value = substitute(cost_value,tvp_horizon_y_ref[i],ref_path_along_horizon[i][1])
          cost_value = substitute(cost_value,tvp_horizon_theta_ref[i],ref_path_along_horizon[i][2])
          cost_value = substitute(cost_value,x_x_horizon[i],state_along_horizon[i][0])
          cost_value = substitute(cost_value,x_y_horizon[i],state_along_horizon[i][1])
          cost_value = substitute(cost_value,x_theta_horizon[i],state_along_horizon[i][2])
      print("Cost function at s {} is {}".format(s,cost_value))    
    print("Cost function value: {}".format(cost_value))
    return cost_value

def show_rl_trajectory(obs_list,act_list,a_length,w_radius):
    x_values = list(map(lambda obs: obs[0], obs_list))
    y_values = list(map(lambda obs: obs[1], obs_list))

    theta_values = list(map(lambda obs: obs[2], obs_list))

    u_l_values = list(map(lambda act: act[0], act_list))
    u_l_values = list(map(lambda act: act[1], act_list))
    
    v_values = []
    w_values = []

    for a in act_list:
        v,w = from_commands_to_robot_velocity(a[0],a[1],a_length,w_radius)
        v_values.append(v)
        w_values.append(w)

    print("Starting point: x:{}, y:{} -PURPLE-".format(x_values[0],y_values[0]))
    print("End point: x:{}, y:{} -GREEN-".format(x_values[-1],y_values[-1]))
    def on_close(event):
        print('Closed Figure!')
        
    fig, ax = plt.subplots(4,figsize=(15,15))
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=2.0, hspace=0.5)

    #left  = 0.125  # the left side of the subplots of the figure
    #right = 0.9    # the right side of the subplots of the figure
    #bottom = 0.1   # the bottom of the subplots of the figure
    #top = 0.9      # the top of the subplots of the figure
    #wspace = 0.2   # the amount of width reserved for blank space between subplots
    #hspace = 0.2   # the amount of height reserved for white space between subplots
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

def main():
    check_diff_drive_env(DifferentialDriveEnvV1)
    #init_pose  = [-0.05, -0.25, -np.pi/2]
    init_pose  = [0.12, -0.25, -np.pi/2]
    obss, actions = load_and_run_model(ppo2_model_name,1000,axle_length, wheel_radius,DifferentialDriveEnvV1,init_pose)
    print("I have {} observations and {} actions ".format(len(obss),len(actions)))
    show_rl_trajectory(obss,actions)

if __name__ == "__main__":
    main()
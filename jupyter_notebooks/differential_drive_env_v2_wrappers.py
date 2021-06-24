import math
import numpy as np

from differential_drive_env_v2 import DifferentialDriveEnvV2

class DifferentialDriveEnvV2Unscaled(DifferentialDriveEnvV2):
  def _scaled_to_real_state(self, state):
    # deep copying to avoid modifying the variables of the caller
    state = list(state)
    if isinstance(state, list) or isinstance(state, np.ndarray):
      state[2] *= self.orientation_map
    return state

  def _real_to_scaled_state(self, state):
    # deep copying to avoid modifying the variables of the caller
    state = list(state)
    if isinstance(state, list) or isinstance(state, np.ndarray):
      state[2] /= self.orientation_map
    return state

  def _real_to_scaled_action(self, action):
    action = np.array(action)
    return action / self.action_map

  def set_init_position(self, init_position):
    # the caller here pass the true/real init_position, but the super() class expects it scaled
    if init_position:
      init_position = self._real_to_scaled_state(init_position)
    return super().set_init_position(init_position)

  def reset(self):
    state = super().reset()
    return self._scaled_to_real_state(state)

  def step(self, action):
    # here action is given unscaled, but the super class expects it scaled
    # (it will convert it again inside its step() for computing the dynamics)
    action = self._real_to_scaled_action(action)
    state, reward, done, info = super().step(action)
    # super()'s step() methods returns the unscaled state, but the caller expects it real
    return self._scaled_to_real_state(state), reward, done, info

class RLAgentUnscalingWrapper:
  def __init__(self, scaled_agent, state_scaling_factors=np.array([1.0]), action_scaling_factors=np.array([1.0])):
    self.scaled_agent = scaled_agent
    self.state_scaling_factors = np.array(state_scaling_factors)
    self.action_scaling_factors = np.array(action_scaling_factors)

  def predict(self, state, deterministic=False):
    #print(f"RLAgentUnscalingWrapper.predict(): state given as parameter = {state}")
    state = np.array(state) / self.state_scaling_factors
    #print(f"RLAgentUnscalingWrapper.predict(): state passed to the internal agent = {state}")
    action, _states = self.scaled_agent.predict(state, deterministic=deterministic)
    #print(f"RLAgentUnscalingWrapper.predict(): internal agent returned action = {action}")
    action = action * self.action_scaling_factors
    #print(f"RLAgentUnscalingWrapper.predict(): returning action = {action}")
    return action, _states
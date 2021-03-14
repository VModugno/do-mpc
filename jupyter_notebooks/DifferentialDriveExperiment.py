import numpy as np
from casadi import *
import do_mpc

class DifferentialDriveExperiment:
    def __init__(self,axle_lenghts,wheel_radii,true_axle_length=None,true_wheel_radius=None):
        if len(axle_lenghts)==1 and true_axle_length or len(wheel_radii)==1 and true_wheel_radius:
            raise Exception("You cannot specify true parameters in a non-scenario setting: in non-scenario setting simulator uses the model values")
        self.delta_t = 0.01 # Euler sampling interval and controller interval
        self.axle_lengths = axle_lenghts
        self.wheel_radii = wheel_radii
        self.true_axle_length = true_axle_length  if true_axle_length else axle_lenghts[0] #axle length used in the simulator and in the visualization
        self.true_wheel_radius =  true_wheel_radius if true_wheel_radius else wheel_radii[0] #wheel radius used in the simulator and in the visualization 
        self.min_wheel_ang_vel = -2
        self.max_wheel_ang_vel = 2
        self.n_horizon = 50 # prediction horizion
        self.n_robust = 1 # robust horizon (if scenario based experiment)
        self._model = None
        self._mpc = None
        self._estimator = None
        self._simulator = None
        
    @property
    def model(self):
        if not self._model:
            self._setup_differential_drive_model()
        return self._model
    
    @property
    def mpc(self):
        if not self._mpc:
            self._setup_differential_drive_model_mpc_controller()
        return self._mpc

    @property
    def estimator(self):
        if not self._estimator:
            self._setup_differential_drive_estimator()
        return self._estimator

    @property
    def simulator(self):
        if not self._simulator:
            self._setup_differential_drive_simulator()
        return self._simulator
    
    @property
    def is_axle_lenght_param(self):
        return len(self.axle_lengths)>1
    
    @property
    def is_wheel_radii_param(self):
        return len(self.wheel_radii)>1

    @property
    def is_scenario_based(self):
        return self.is_axle_lenght_param or self.is_wheel_radii_param

    def _setup_differential_drive_model(self):
        model_type = 'discrete'
        model = do_mpc.model.Model(model_type)
        # Euler sampling interval
        delta_t = self.delta_t
        # Verify if L, distance between wheels, is an uncertain parameter
        if self.is_axle_lenght_param: 
            L = model.set_variable('_p', 'L')
        else:
            # Single hypothesis for L intrawheels 
            L = self.axle_lengths[0]
        # Verify if r, distance between wheels is an uncertain parameter
        if self.is_wheel_radii_param:
            r = model.set_variable('_p', 'r')
        else:
            # Single hypothesis for r, wheel radius
            r = self.wheel_radii[0]
    
        # input angular velocities of the left (u_l) and right (u_r) wheels
        u_l = model.set_variable('_u', 'u_l')
        u_r = model.set_variable('_u', 'u_r')
    
        # linear (v) and angular (w) velocities of the center of the robot axle
        v = (u_l + u_r) * r / 2
        w = (u_r - u_l) * r / L

        # auxiliary expressions for the robot center visualization
        model.set_expression('v', v)
        model.set_expression('w', w)
        
        # state variables : position of the center of the robot axle and heading wrt x-axis
        x = model.set_variable('_x', 'x')
        y = model.set_variable('_x', 'y')
        theta = model.set_variable('_x', 'theta')
        
        # right-hand side of the dynamics
        x_next = x + v * delta_t * cos(theta)
        y_next = y + v * delta_t * sin(theta)
        theta_next = theta + w * delta_t
        model.set_rhs('x', x_next)
        model.set_rhs('y', y_next)
        model.set_rhs('theta', theta_next)

        # auxiliary expressions for the quadratic distance to the origin
        model.set_expression('distance', x**2+y**2)
        model.set_expression('zero',x-x)
        model.setup()

        self._model = model

    def _setup_differential_drive_model_mpc_controller(self):
        mpc = do_mpc.controller.MPC(self.model)
        n_robust = 0
        if self.is_scenario_based:
            n_robust = self.n_robust
        setup_mpc = {
        'n_horizon': self.n_horizon, # prediction horizion
        'n_robust': n_robust, # robust horizon
        'open_loop': 0, # if set to false, for each time step and scenario an individual control input it computed
        't_step': self.delta_t,  # timestep of the mpc
        'store_full_solution': True, # choose whether to store the full solution of the optimization problem
        # Use MA27 linear solver in ipopt for faster calculations:
        'nlpsol_opts': {'ipopt.linear_solver': 'mumps'}
        }
        mpc.set_param(**setup_mpc)

        mterm = self.model.aux['distance'] # "naive" terminal cost
        lterm = self.model.aux['zero']
        mpc.set_objective(mterm=mterm,lterm=lterm) # "naive" cost function
        mpc.set_rterm(u_l=0, u_r=0) # smooth factor for the penalization for the module of the input: currently no penalization are taking into account (to be defined)

        # set the constraints of the control problem
        mpc.bounds['lower','_u','u_l'] = self.min_wheel_ang_vel
        mpc.bounds['upper','_u','u_l'] = self.max_wheel_ang_vel
        mpc.bounds['lower','_u','u_r'] = self.min_wheel_ang_vel
        mpc.bounds['upper','_u','u_r'] = self.max_wheel_ang_vel

        if self.is_axle_lenght_param:
            if self.is_wheel_radii_param:
                mpc.set_uncertainty_values(L=self.axle_lengths, r=self.wheel_radii)
            else:
                mpc.set_uncertainty_values(L=self.axle_lengths)
        else:
            if self.is_wheel_radii_param:
                mpc.set_uncertainty_values(r=self.wheel_radii)
        mpc.setup()
        self._mpc = mpc

    def _setup_differential_drive_estimator(self):
        # We assume that all states can be directly measured
        self._estimator = do_mpc.estimator.StateFeedback(self.model)
    
    def _setup_differential_drive_simulator(self):
        # Create simulator in order to run MPC in a closed-loop
        simulator = do_mpc.simulator.Simulator(self.model)
        simulator.set_param(t_step=self.delta_t)
        if self.is_scenario_based:
            p_template = simulator.get_p_template()
            def p_fun(t_now):
                if 'L' in p_template.keys():
                    p_template['L'] = self.true_axle_length
                if 'r' in p_template.keys():
                    p_template['r'] = self.true_wheel_radius
                return p_template
            simulator.set_p_fun(p_fun)
        simulator.setup()
        self._simulator = simulator
    
    def setup_experiment(self,initial_pose):
        # Define the initial state of the system and set for all parts of the closed-loop configuration:
        self.simulator.x0['x'] = initial_pose['x']
        self.simulator.x0['y'] = initial_pose['y']
        self.simulator.x0['theta'] =  initial_pose['theta']
        
        x0 = self.simulator.x0.cat.full()

        self.mpc.x0 = x0
        self.estimator.x0 = x0

        self.mpc.set_initial_guess()
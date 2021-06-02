import numpy as np
from casadi import *
import itertools
import do_mpc
import baseline_integration as bi
from differential_drive_env_v1 import DifferentialDriveEnvV1

class DifferentialDriveExperiment:
    def __init__(self,axle_lengths_dict,wheel_radii_dict, tracking_trajectories=None,n_horizon=50):
        #tracking_trajectories is an array of hashes 
        #   like this  {'L':..,'r':,'path':[[x0,y0,theta0],[x1,y1,theta1],...],'actions':[[u_l0,u_r0],[u_l0,u_r0],...]}
        #   or like this  {'L':..,'r':,'policy_name': model_name}
        self.preprocess_axle_info(axle_lengths_dict)
        self.preprocess_wheel_info(wheel_radii_dict)
    
        #if, for at the least one of the possible parameters of the model, user has defined the probabilities 
        #only one of the following situation are supported:
        #the other parameter is a costant in the model so param1_probabilities=[some array] and param2_probabilities=None but param2 is constant
        #the other parameter has a set of values > 1 and user has provided a set of custom probabilities also for them param1_probabilities=[some array] param2_probabilities=[some array]
        #the other parameter is not a costant but has a single value and we can set its probability to one

        if self.axle_probabilities: 
            if not self.wheel_probabilities:
                if len(self.wheel_radii)>1:
                    raise Exception("Custom probabilities provided for a multihypothesis param: the axle, but not for the other, the wheel radius")   
                else:
                    if self.is_wheel_radius_param:
                        self.wheel_probabilities = [1] 
                    else:
                        print ("ok only axle prob assigned but the other param has a single value and it will be embedded")   
            else:
                print ("ok both weigths assigned")
        else:
            if self.wheel_probabilities:
                if len(self.axle_lengths)>1:
                    raise Exception("Custom probabilities provided for a multihypothesis param: the wheel radius, but not for the other, the axle length") 
                else:
                    if self.is_axle_length_param:
                        self.axle_probabilities = [1]
                    else:
                       print ("ok only wheel prob assigned but the other param has a single value and it will be embedded")
            else:
                print ("ok both weights are not assigned so the library will assign the same prob to each scenario")      

        self.params_combinations = len(self.axle_lengths)*len(self.wheel_radii)
        self.preprocess_trajectories(tracking_trajectories)
            

        self.delta_t = 0.01 # Euler sampling interval and controller interval    
        
        #self.min_wheel_ang_vel = -2
        #self.max_wheel_ang_vel = 2
        self.min_wheel_ang_vel = -3
        self.max_wheel_ang_vel = 3
        
        #self.n_horizon = 50 # prediction horizion
        self.n_horizon = n_horizon
        self.n_robust = 1 # robust horizon (if scenario based experiment)
        
        self._model = None
        self._mpc = None
        self._estimator = None
        self._simulator = None
        

    def preprocess_axle_info(self,axle_length_info):
        #Assumption: axle_length_info is a dict with the following structure 
        # {'values':[list of values also only one], #mandatory
        #  'probs':[list of the probabilities of the axle_values], #optional 
        #  'true_value': groundtruth length value} #optional
        assert 'values' in axle_length_info.keys(),'values is a mandatory key in axle_length_info'
        assert isinstance(axle_length_info['values'], list),'axle length values have to be provided as a list'
        assert len(axle_length_info['values'])>0 ,'at the least one value for the axle parameter has to be provided'
        assert len(set(axle_length_info['values'])) == len(axle_length_info['values']), 'Axle length values must be distinct'
        #in theory I had also to check that the length values are numbers
        self.axle_lengths = axle_length_info['values']

        self.is_axle_length_param = True
        self.true_axle_length = None
        self.axle_probabilities = None
        if len(axle_length_info['values'])==1: #a single value for the axle length has been provided
            if('probs'in axle_length_info.keys()):
                raise Exception("Probability of axle length are not meanigful if only one single value is provided")
            else:
                if 'true_value'in axle_length_info.keys():
                    self.true_axle_length = axle_length_info['true_value']
                else:
                    self.is_axle_length_param = False #if axle length has a single value and the simulator will use the same value no need to do it as a param
                    self.true_axle_length = self.axle_lengths[0] #only for graphics the true value in the simulator is embedded
        else: # more than one value has been provided for the axle length
            if('probs' in axle_length_info.keys()):
                assert isinstance(axle_length_info['probs'], list),'probability of the hypothesis can be provided only through a list'
                assert len(axle_length_info['values'])==len(axle_length_info['probs']), 'axle probabilities have to be in the same number of the axle lengths'
                self.axle_probabilities = axle_length_info['probs']
            if('true_value' in axle_length_info.keys()):
                self.true_axle_length = axle_length_info['true_value']
            else:
                self.true_axle_length = self.axle_lengths[0]
    
    def preprocess_wheel_info(self, wheel_radii_info):
        #Assumption: wheel_radii_info is a dict with following structure 
        # {'values':[list of possible wheel radii, at the least one], #mandatory
        #  'probs':[list of the probabilities of the wheel_radii_values], #optional 
        #  'true_value': groundtruth radii value} #optional
        assert 'values' in wheel_radii_info.keys(),'values is a mandatory key in the wheel radii dict'
        assert isinstance(wheel_radii_info['values'], list),'wheel radii values have to be provided as a list'
        assert len(wheel_radii_info['values'])>0 ,'at the least one value for the wheel radii parameter has to be provided'
        assert len(set(wheel_radii_info['values'])) == len(wheel_radii_info['values']), 'Axle length values must be distinct'
        #in theory I should also to check that the radii values provided are all numeric
        self.wheel_radii = wheel_radii_info['values']

        self.is_wheel_radius_param = True
        self.true_wheel_radius = None
        self.wheel_probabilities = None
        if len(wheel_radii_info['values'])==1: #it has been provided only a SINGLE value for the wheel radius
            if('probs'in wheel_radii_info.keys()):
                raise Exception("Probability of a wheel radii are not meanigful if only one single value is provided")
            else:
                if 'true_value'in wheel_radii_info.keys():
                    self.true_wheel_radius = wheel_radii_info['true_value']
                else:
                    self.is_wheel_radius_param = False #if wheel radius has a single value and the simulator will use the same value, no need for parametrize it
                    self.true_wheel_radius = self.wheel_radii[0]  #only for graphics the true value in the simulator is embedded
        else: # more than one value has been provided for the wheel radius
            if('probs' in wheel_radii_info.keys()):
                assert isinstance( wheel_radii_info['probs'], list),'probability of the hypothesis can be provided only through a list'
                assert len( wheel_radii_info['values'])==len(wheel_radii_info['probs']), 'wheel radii probabilities must have the same size of wheel radii values'
                self.wheel_probabilities = wheel_radii_info['probs']
            if('true_value' in wheel_radii_info.keys()):
                self.true_wheel_radius = wheel_radii_info['true_value']
            else:
                self.true_wheel_radius = self.wheel_radii[0]  

    def preprocess_trajectories(self,tracking_trajectories):   
        self.tracking_trajectories = tracking_trajectories 
        #check the structure of the trajectories dict
        if self.tracking_trajectory_mode:
            for t in self.tracking_trajectories:
                t_keys = list(t.keys())
                assert 'L' in t.keys() and 'r' in t.keys(), 'In tracking trajectory mode the L and r keys are mandatory in the trajectory dict'
                if self.online_trajectory_mode:
                    assert len(t_keys) == 3 or len(t_keys) == 4
                    assert 'policy_name' in t.keys(), 'In online tracking trajectory mode policy_name is a mandatory entry in the trajectory dict'
                    if len(t_keys) == 4:
                        assert 'env_class_name' in t.keys(), 'To specify the env of the policy for the tracking use the env_class_name key in the trajectory dict'
                        env_class_name = t['env_class_name']
                    else:
                        env_class_name = DifferentialDriveEnvV1
                    policy_env_dict = bi.setup_model_execution_on_env(t['policy_name'],t['L'],t['r'],init_pos=None,env_class=env_class_name)
                    t['policy'] = policy_env_dict['policy']
                    t['env'] = policy_env_dict['env']
                else:
                    assert len(t_keys) == 4
                    assert 'path' in t.keys() and 'actions' in t.keys(),'In offline trajectory mode path and actions keys are mandatory in the trajectory dict'
        if self.scenario_based_trajectory_tracking:
            assert len(tracking_trajectories)==self.params_combinations, "In trajectory mode the allowed number of trajectories is 1 or the number of possible combinations of parameters"
            # verify the structure of the tracking_trajectories dict:
            # verify that for each combination <L,r> exists a trajectory 
            # (if yes the previous assert guarantee that each trajectory corresponds to a scenario of the problem)
            # IMPORTANT:
            # reorder tracking trajectories with the same order of mpc.set_uncertainty_parameters (controller.py line 802)
            # Important verify that for each scenario there is or the policy or actions and path key

            scenarios = list(itertools.product(self.axle_lengths,self.wheel_radii))
            print("Scenarios {}".format(scenarios))
            ordered_trajectories = []
            for s in scenarios:
                traj = list(filter(lambda t:t['L']==s[0] and t['r']==s[1],tracking_trajectories))
                assert len(traj) == 1, "Scenario L={} and r={} must have one single reference trajectory".format(s[0],s[1])
                ordered_trajectories.append(traj[0])
            self.tracking_trajectories = ordered_trajectories

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
    def regulation_mode(self):
        return bool(self.tracking_trajectories == None or len(self.tracking_trajectories) ==0) 

    @property
    def tracking_trajectory_mode(self):
        return not self.regulation_mode
    
    @property
    def online_trajectory_mode(self):
        return self.tracking_trajectory_mode and  ('policy_name' in self.tracking_trajectories[0].keys())
    
    @property
    def scenario_based_trajectory_tracking(self):
        return self.tracking_trajectory_mode and (len(self.tracking_trajectories)>1)

 
    #True if axle length or wheel radius are parameters (also if a single value is provided for them)
    @property
    def is_a_parametrized_model(self):
        return self.is_axle_length_param or self.is_wheel_radius_param
    
    @property
    def is_fully_parametrized_model(self):
        return self.is_axle_length_param and self.is_wheel_radius_param
    
    #True if axle length or wheel radius are parameters and more than one possible value is provided at least for one of them
    @property
    def is_scenario_based(self):
        return self.is_a_parametrized_model and len(self.axle_lengths)>1 or len(self.wheel_radii)>1

    @property
    def is_custom_weighted_scenario_based(self):
        if self.is_fully_parametrized_model:
            return not self.axle_probabilities is None and not self.wheel_probabilities is None #the first check is enough
        #here if at least one in [axle_length, wheel_radius] is not a parameter
        return not self.axle_probabilities is None or not self.wheel_probabilities is None

    def _setup_differential_drive_model(self):
        model_type = 'discrete'
        model = do_mpc.model.Model(model_type)
        # Euler sampling interval
        delta_t = self.delta_t
        # Verify if L, distance between wheels, is an uncertain parameter
        if self.is_axle_length_param:
            L = model.set_variable('_p', 'L')
        else:
            # L, the intrawheels distance, is not a parameter and it is embedded in the model
            L = self.axle_lengths[0]
        # Verify if r, distance between wheels is an uncertain parameter
        if self.is_wheel_radius_param:
            r = model.set_variable('_p', 'r')
        else:
            # r is not a parameter and it is embedded in the model
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

        #time varying parameters for trajectory tracking
        if self.tracking_trajectory_mode:
            x_ref = model.set_variable(var_type='_tvp', var_name='x_ref')
            y_ref = model.set_variable(var_type='_tvp', var_name='y_ref')
            theta_ref = model.set_variable(var_type='_tvp', var_name='theta_ref')
            u_l_ref = model.set_variable(var_type='_tvp', var_name='u_l_ref')
            u_r_ref = model.set_variable(var_type='_tvp', var_name='u_r_ref')
        else:
            x_ref = 0
            y_ref = 0
            theta_ref = 0
            u_l_ref = 0
            u_r_ref = 0

        # right-hand side of the dynamics
        x_next = x + v * delta_t * cos(theta)
        y_next = y + v * delta_t * sin(theta)
        theta_next = theta + w * delta_t
        model.set_rhs('x', x_next)
        model.set_rhs('y', y_next)
        model.set_rhs('theta', theta_next)

        # auxiliary expressions for the quadratic distance to the origin
        model.set_expression('squared_distance_to_target', x**2+y**2)
        model.set_expression('zero',x-x)
        model.set_expression('position_norm', sqrt(x**2+y**2))
        
        #expressions for trajectory tracking
        if self.tracking_trajectory_mode:
            #model.set_expression('squared_trajectory_error',(x-x_ref)**2+(y-y_ref)**2)
            model.set_expression('squared_trajectory_error', (x-x_ref)**2+(y-y_ref)**2)
            model.set_expression('trajectory_error',sqrt((x-x_ref)**2+(y-y_ref)**2))
            model.set_expression('orientation_trajectory_difference',atan2(sin(theta-theta_ref),cos(theta-theta_ref)))
            model.set_expression('squared_orientation_trajectory_difference',(atan2(sin(theta-theta_ref),cos(theta-theta_ref)))**2)
            model.set_expression('squared_action_trajectory_error',(u_l-u_l_ref)**2+(u_r-u_r_ref)**2)
        model.setup()

        self._model = model
    
    @staticmethod
    def _fill_mpc_scenario_tvp_template_with_offline_trajectories(t_now,delta_t,s_tvp_template,track_trajectories):
        p_combinations = len(s_tvp_template['_tvp'])
        horizon_plus_1 = len(s_tvp_template['_tvp'][0])
        for c in range(p_combinations):
            curr_trj = track_trajectories[c]
            #print("CURR TRAJ PATH {}".format(curr_trj['path']))
            for k in range(horizon_plus_1):    
                base_index = int(t_now / delta_t)
                #print("t_now val {} type {} base_index value {}".format(t_now,type(t_now),base_index))
                if (base_index + k) < len(curr_trj['path']):
                    path_index = base_index + k
                else:
                    path_index = -1
                s_tvp_template['_tvp',c,k,'x_ref'] = curr_trj['path'][path_index][0]
                s_tvp_template['_tvp',c,k,'y_ref'] = curr_trj['path'][path_index][1]
                s_tvp_template['_tvp',c,k,'theta_ref'] = curr_trj['path'][path_index][2]
                if (base_index + k) < len(curr_trj['actions']):
                        act_index = base_index + k
                else:
                        act_index = -1 
                s_tvp_template['_tvp',c,k,'u_l_ref'] = curr_trj['actions'][act_index][0]
                s_tvp_template['_tvp',c,k,'u_r_ref'] = curr_trj['actions'][act_index][1]
        return s_tvp_template
    
    @staticmethod
    def _fill_mpc_scenario_tvp_template_with_online_trajectories(curr_state,s_tvp_template,track_trajectories):
        p_combinations = len(s_tvp_template['_tvp'])
        horizon_plus_1 = len(s_tvp_template['_tvp'][0])
        for c in range(p_combinations):
            curr_trj = track_trajectories[c]
            path, actions = bi.run_model(curr_trj['policy'],curr_trj['env'],horizon_plus_1,curr_state)
            for k in range(horizon_plus_1):
                path_index = k if k<len(path) else -1
                s_tvp_template['_tvp',c,k,'x_ref'] = path[path_index][0]
                s_tvp_template['_tvp',c,k,'y_ref'] = path[path_index][1]
                s_tvp_template['_tvp',c,k,'theta_ref'] = path[path_index][2]
                act_index = k if k<len(actions) else -1
                s_tvp_template['_tvp',c,k,'u_l_ref'] = actions[act_index][0]
                s_tvp_template['_tvp',c,k,'u_r_ref'] = actions[act_index][1]
        return s_tvp_template
    
    @staticmethod
    def _fill_mpc_tvp_template_with_offline_trajectory(t_now,delta_t,s_tvp_template,trajectory):
        horizon_plus_1 = len(s_tvp_template['_tvp'])
        for k in range(horizon_plus_1):
            #print("TRAJ PATH has {} point".format(len(trajectory['path'])))
            base_index = int(t_now / delta_t)
            #print("t_now val {} type {} base_index value {}".format(t_now,type(t_now),base_index))
            if (base_index + k) < len(trajectory['path']):
                path_index = base_index + k
            else:
                path_index = -1
            s_tvp_template['_tvp',k,'x_ref'] = trajectory['path'][path_index][0]
            s_tvp_template['_tvp',k,'y_ref'] = trajectory['path'][path_index][1]
            s_tvp_template['_tvp',k,'theta_ref'] = trajectory['path'][path_index][2]
            if (base_index + k) < len(trajectory['actions']):
                act_index = base_index + k
            else:
                act_index = -1 
            s_tvp_template['_tvp',k,'u_l_ref'] = trajectory['actions'][act_index][0]
            s_tvp_template['_tvp',k,'u_r_ref'] = trajectory['actions'][act_index][1]
        #print("AFTER ASSIGN s_tvp_template {}".format(s_tvp_template['_tvp']))
        return s_tvp_template
    
    @staticmethod
    def _fill_mpc_tvp_template_with_online_trajectory(curr_state,s_tvp_template,trajectory):
        #print("QUIIIIIIIIIIIIIIIIIII MPC from curr_state {}".format(curr_state))
        horizon_plus_1 = len(s_tvp_template['_tvp'])
        path, actions = bi.run_model(trajectory['policy'],trajectory['env'],horizon_plus_1,curr_state)
        for k in range(horizon_plus_1):
            path_index = k if k<len(path) else -1
            s_tvp_template['_tvp',k,'x_ref'] = path[path_index][0]
            s_tvp_template['_tvp',k,'y_ref'] = path[path_index][1]
            s_tvp_template['_tvp',k,'theta_ref'] = path [path_index][2]
            act_index = k if k<len(actions) else -1
            s_tvp_template['_tvp',k,'u_l_ref'] = actions[act_index][0]
            s_tvp_template['_tvp',k,'u_r_ref'] = actions[act_index][1]
        return s_tvp_template

    def _setup_differential_drive_model_mpc_controller(self):
        
        mpc = do_mpc.controller.MPC(self.model)
        n_robust = 0
        if self.is_scenario_based:
            n_robust = self.n_robust
        setup_mpc = {
        'n_horizon': self.n_horizon, # prediction horizion
        'n_robust': n_robust, # robust horizon
        'open_loop': 0, # if set to false, for each time step and scenario an individual control input it computed
        'state_discretization': 'discrete',
        't_step': self.delta_t,  # timestep of the mpc
        'store_full_solution': True, # choose whether to store the full solution of the optimization problem
        'scenario_tvp': self.scenario_based_trajectory_tracking
        # Use MA27 linear solver in ipopt for faster calculations:
        #'nlpsol_opts': {'ipopt.linear_solver': 'mumps'}
        }
        
        mpc.set_param(**setup_mpc)
        if self.regulation_mode:
            mterm = self.model.aux['squared_distance_to_target'] # "naive" terminal cost
            #lterm = self.model.aux['zero']
            #lterm = self.model.aux['squared_distance_to_target']
            lterm = self.model.aux['position_norm']
        else:
            mterm = self.model.aux['squared_trajectory_error'] +self.model.aux['squared_orientation_trajectory_difference']# "naive" terminal cost for trajectory tracking
            #lterm = self.model.aux['trajectory_error'] 
            lterm = self.model.aux['squared_trajectory_error'] + self.model.aux['squared_orientation_trajectory_difference']
            

        mpc.set_objective(mterm=mterm,lterm=lterm) # "naive" cost function
        #mpc.set_rterm(u_l=1e-2, u_r=1e-2) # smooth factor for the penalization for the module of the input
        mpc.set_rterm(u_l=0, u_r=0) # smooth factor for the penalization for the module of the input: currently no penalization are taking into account (to be defined)
        
        # set the constraints of the control problem
        mpc.bounds['lower','_u','u_l'] = self.min_wheel_ang_vel
        mpc.bounds['upper','_u','u_l'] = self.max_wheel_ang_vel
        mpc.bounds['lower','_u','u_r'] = self.min_wheel_ang_vel
        mpc.bounds['upper','_u','u_r'] = self.max_wheel_ang_vel

        if self.tracking_trajectory_mode:
            if self.scenario_based_trajectory_tracking:
                tvp_template_mpc = mpc.get_tvp_template(n_combinations=self.params_combinations)
                if self.online_trajectory_mode:
                    def tvp_fun_mpc(t_now,current_x=[0,0,0]):
                        return self._fill_mpc_scenario_tvp_template_with_online_trajectories(current_x,tvp_template_mpc,self.tracking_trajectories)
                else:
                    def tvp_fun_mpc(t_now):
                        return self. _fill_mpc_scenario_tvp_template_with_offline_trajectories(t_now,self.delta_t,tvp_template_mpc,self.tracking_trajectories)

                mpc.set_tvp_fun(tvp_fun_mpc,n_combinations=self.params_combinations)

            else:
                tvp_template_mpc = mpc.get_tvp_template()
                if self.online_trajectory_mode:
                    def tvp_fun_mpc(t_now, current_x=[0,0,0]):
                        return self._fill_mpc_tvp_template_with_online_trajectory(current_x,tvp_template_mpc,self.tracking_trajectories[0])
                else:
                    def tvp_fun_mpc(t_now):
                        return self._fill_mpc_tvp_template_with_offline_trajectory(t_now,self.delta_t,tvp_template_mpc,self.tracking_trajectories[0])
                
                mpc.set_tvp_fun(tvp_fun_mpc)
            
        if self.is_axle_length_param:
            if self.is_wheel_radius_param:
                mpc.set_uncertainty_values(L=self.axle_lengths, r=self.wheel_radii)
                if (self.axle_probabilities): #if true also wheel_probability are defined for the check in the constructor
                    mpc.set_uncertainty_weights(L=self.axle_probabilities, r=self.wheel_probabilities)
            else:
                mpc.set_uncertainty_values(L=self.axle_lengths)
                if (self.axle_probabilities): 
                    mpc.set_uncertainty_weights(L=self.axle_probabilities)
        else:
            if self.is_wheel_radius_param:
                mpc.set_uncertainty_values(r=self.wheel_radii)
                if (self.wheel_probabilities): 
                    mpc.set_uncertainty_weights(r=self.wheel_probabilities)
        mpc.setup()
        self._mpc = mpc

    def _setup_differential_drive_estimator(self):
        # We assume that all states can be directly measured
        self._estimator = do_mpc.estimator.StateFeedback(self.model)

    @staticmethod
    def _fill_simulator_tvp_template_with_offline_trajectory(t_now,delta_t,sim_tvp_template,trajectory):
        base_index = int(t_now /delta_t)
        if (base_index) < len(trajectory['path']):
            path_index = base_index
        else:
            path_index = -1
        sim_tvp_template['x_ref'] = trajectory['path'][path_index][0]
        sim_tvp_template['y_ref'] = trajectory['path'][path_index][1]
        sim_tvp_template['theta_ref'] = trajectory['path'][path_index][2]
        if (base_index) < len(trajectory['actions']):
            act_index = base_index
        else:
            act_index = -1
        sim_tvp_template['u_l_ref'] = trajectory['actions'][act_index][0]
        sim_tvp_template['u_r_ref'] = trajectory['actions'][act_index][1]

        return sim_tvp_template
    
    @staticmethod
    def _fill_simulator_tvp_template_with_online_trajectory(curr_state,sim_tvp_template,trajectory):
        path, actions = bi.run_model(trajectory['policy'],trajectory['env'],1,curr_state)
        sim_tvp_template['x_ref'] = path[0][0]
        sim_tvp_template['y_ref'] = path[0][1]
        sim_tvp_template['theta_ref'] = path [0][2]
        sim_tvp_template['u_l_ref'] = actions[0][0]
        sim_tvp_template['u_r_ref'] = actions[0][1]
        return sim_tvp_template

    def _setup_differential_drive_simulator(self):
        # Create simulator in order to run MPC in a closed-loop
        simulator = do_mpc.simulator.Simulator(self.model)
        simulator.set_param(t_step=self.delta_t)
        if self.is_a_parametrized_model:
            p_template = simulator.get_p_template()
            def p_fun(t_now):
                if 'L' in p_template.keys():
                    p_template['L'] = self.true_axle_length
                if 'r' in p_template.keys():
                    p_template['r'] = self.true_wheel_radius
                return p_template
            simulator.set_p_fun(p_fun)

        if self.tracking_trajectory_mode:
            # Get the template
            tvp_template = simulator.get_tvp_template()
            if self.online_trajectory_mode:
                def sim_tvp_fun(t_now,current_x=[0,0,0]):
                    return self._fill_simulator_tvp_template_with_online_trajectory(current_x,tvp_template,self.tracking_trajectories[0])
            else:
                def sim_tvp_fun(t_now):
                    return self._fill_simulator_tvp_template_with_offline_trajectory(t_now,self.delta_t,tvp_template,self.tracking_trajectories[0])
            # Set the tvp_fun:
            simulator.set_tvp_fun(sim_tvp_fun)

        simulator.setup()
        self._simulator = simulator

    def setup_experiment(self,initial_pose,initial_action_guess={'u_l':0,'u_r':0}):
        # Define the initial state of the system and set for all parts of the closed-loop configuration:
        #self.simulator.x0['x'] = initial_pose['x']
        #self.simulator.x0['y'] = initial_pose['y']
        #self.simulator.x0['theta'] =  initial_pose['theta']

        #x0 = self.simulator.x0.cat.full()
        print(initial_pose.values())
        x0_np = np.array([[initial_pose['x']],
                          [initial_pose['y']],
                          [initial_pose['theta']]
                        ])
        u0_np = np.array([[initial_action_guess['u_l']],
                          [initial_action_guess['u_r']]
                        ])
        print(x0_np)
        #self.mpc.x0 = x0

        self.mpc.x0 = x0_np
        self.simulator.x0 = x0_np
        #self.estimator.x0 = x0
        self.estimator.x0 = x0_np

        self.mpc.set_initial_guess()
    
    #def run_experiment

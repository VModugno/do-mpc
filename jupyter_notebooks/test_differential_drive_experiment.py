import numpy as np
import do_mpc
from DifferentialDriveExperiment import DifferentialDriveExperiment 
import baseline_integration as bi
from casadi import *
from casadi.tools import *

#https://docs.pytest.org/en/stable/getting-started.html
#pytest -q [-rP] [-rx] test_differential_drive_experiment.py

class TestDifferentialDriveExperiment:
    
    #No custom weight tests
    #No scenario based
    def test_one_no_param_in_model(self):
        experiment = DifferentialDriveExperiment(
                axle_lengths_dict={'values':[0.5]}, 
                wheel_radii_dict={'values':[0.1]})
        init_robot_pose = {'x': -0.5, 'y': 0.0, 'theta': np.pi/2}
        experiment.setup_experiment(init_robot_pose)
          
        assert not experiment.is_axle_length_param
        assert not experiment.is_wheel_radius_param
        assert not experiment.is_a_parametrized_model
        assert not experiment.is_fully_parametrized_model
        assert not experiment.is_scenario_based
        assert not experiment.is_custom_weighted_scenario_based
        assert experiment.axle_probabilities == None
        assert experiment.wheel_probabilities == None
        assert experiment.true_axle_length == 0.5
        assert experiment.true_wheel_radius == 0.1
        assert experiment.regulation_mode
        assert not experiment.tracking_trajectory_mode

    def test_two_axle_length_single_value_param(self):
        experiment = DifferentialDriveExperiment(
                axle_lengths_dict={'values':[0.5],'true_value':0.47}, 
                wheel_radii_dict={'values':[0.1]})
        init_robot_pose = {'x': -0.5, 'y': 0.0, 'theta': np.pi/2}
        experiment.setup_experiment(init_robot_pose)
        
        assert experiment.is_a_parametrized_model
        assert experiment.is_axle_length_param
        assert not experiment.is_wheel_radius_param
        assert not experiment.is_scenario_based
        assert not experiment.is_custom_weighted_scenario_based
        assert experiment.axle_probabilities == None
        assert experiment.wheel_probabilities == None
        assert experiment.true_axle_length == 0.47
        assert experiment.true_wheel_radius == 0.1
        assert experiment.regulation_mode
        assert not experiment.tracking_trajectory_mode
    
    def test_two_wheel_radius_single_value_param(self):
        experiment = DifferentialDriveExperiment(
                axle_lengths_dict={'values':[0.5]}, 
                wheel_radii_dict={'values':[0.1],'true_value':0.11})
        init_robot_pose = {'x': -0.5, 'y': 0.0, 'theta': np.pi/2}
        experiment.setup_experiment(init_robot_pose)

        assert experiment.is_a_parametrized_model
        assert not experiment.is_axle_length_param
        assert experiment.is_wheel_radius_param
        assert not experiment.is_scenario_based
        assert not experiment.is_custom_weighted_scenario_based
        assert experiment.axle_probabilities == None
        assert experiment.wheel_probabilities == None
        assert experiment.true_axle_length == 0.5
        assert experiment.true_wheel_radius == 0.11
        assert experiment.regulation_mode
        assert not experiment.tracking_trajectory_mode

    def test_three_axle_length_wheel_radius_single_value_param(self):
        experiment = DifferentialDriveExperiment(
                axle_lengths_dict={'values':[0.5],'true_value':0.48}, 
                wheel_radii_dict={'values':[0.1],'true_value':0.11})
        init_robot_pose = {'x': -0.5, 'y': 0.0, 'theta': np.pi/2}
        experiment.setup_experiment(init_robot_pose)

        assert experiment.is_a_parametrized_model
        assert experiment.is_axle_length_param
        assert experiment.is_wheel_radius_param
        assert not experiment.is_scenario_based
        assert not experiment.is_custom_weighted_scenario_based
        assert experiment.axle_probabilities == None
        assert experiment.wheel_probabilities == None
        assert experiment.true_axle_length == 0.48
        assert experiment.true_wheel_radius == 0.11
        assert experiment.regulation_mode
        assert not experiment.tracking_trajectory_mode
    
    #No custom weight tests
    #Scenario based : at least one in [axle_length, wheel radius] is a param and at least one param is a multivalue param
    def test_four_axle_length_multival_param_wheel_radius_embedded(self):
        experiment = DifferentialDriveExperiment(
                axle_lengths_dict={'values':[0.48,0.5,0.53]}, 
                wheel_radii_dict={'values':[0.1]})
        init_robot_pose = {'x': -0.5, 'y': 0.0, 'theta': np.pi/2}
        experiment.setup_experiment(init_robot_pose)

        assert experiment.is_a_parametrized_model
        assert experiment.is_axle_length_param
        assert not experiment.is_wheel_radius_param
        assert experiment.is_scenario_based
        assert not experiment.is_custom_weighted_scenario_based
        assert experiment.axle_probabilities == None
        assert experiment.wheel_probabilities == None
        assert experiment.true_axle_length == 0.48
        assert experiment.true_wheel_radius == 0.1
        assert experiment.regulation_mode
        assert not experiment.tracking_trajectory_mode
    
    def test_five_wheel_radius_multival_param_axle_length_embedded(self):
        experiment = DifferentialDriveExperiment(
                axle_lengths_dict={'values':[0.5]}, 
                wheel_radii_dict={'values':[0.1,0.15,0.20]})
        init_robot_pose = {'x': -0.5, 'y': 0.0, 'theta': np.pi/2}
        experiment.setup_experiment(init_robot_pose)

        assert experiment.is_a_parametrized_model
        assert not experiment.is_axle_length_param
        assert experiment.is_wheel_radius_param
        assert experiment.is_scenario_based
        assert not experiment.is_custom_weighted_scenario_based
        assert experiment.axle_probabilities == None
        assert experiment.wheel_probabilities == None
        assert experiment.true_axle_length == 0.5
        assert experiment.true_wheel_radius == 0.1
        assert experiment.regulation_mode
        assert not experiment.tracking_trajectory_mode

    def test_six_axle_length_multival_param_and_true_value_for_sim_wheel_radius_embedded(self):
        experiment = DifferentialDriveExperiment(
                axle_lengths_dict={'values':[0.48,0.5,0.53],'true_value':0.53}, 
                wheel_radii_dict={'values':[0.1]})
        init_robot_pose = {'x': -0.5, 'y': 0.0, 'theta': np.pi/2}
        experiment.setup_experiment(init_robot_pose)

        assert experiment.is_a_parametrized_model
        assert experiment.is_axle_length_param
        assert not experiment.is_wheel_radius_param
        assert experiment.is_scenario_based
        assert not experiment.is_custom_weighted_scenario_based
        assert experiment.axle_probabilities == None
        assert experiment.wheel_probabilities == None
        assert experiment.true_axle_length == 0.53
        assert experiment.true_wheel_radius == 0.1
        assert experiment.regulation_mode
        assert not experiment.tracking_trajectory_mode
    
    def test_seven_wheel_radius_multival_param_and_true_value_for_sim_axle_length_embedded(self):
        experiment = DifferentialDriveExperiment(
                axle_lengths_dict={'values':[0.5]}, 
                wheel_radii_dict={'values':[0.1,0.15,0.20],'true_value':0.20})
        init_robot_pose = {'x': -0.5, 'y': 0.0, 'theta': np.pi/2}
        experiment.setup_experiment(init_robot_pose)

        assert experiment.is_a_parametrized_model
        assert not experiment.is_axle_length_param
        assert experiment.is_wheel_radius_param
        assert not experiment.is_custom_weighted_scenario_based
        assert experiment.is_scenario_based
        assert experiment.axle_probabilities == None
        assert experiment.wheel_probabilities == None
        assert experiment.true_axle_length == 0.5
        assert experiment.true_wheel_radius == 0.20
        assert experiment.regulation_mode
        assert not experiment.tracking_trajectory_mode

    def test_eight_axle_length_multival_param_wheel_radius_single_value_param(self):
        experiment = DifferentialDriveExperiment(
                axle_lengths_dict={'values':[0.48,0.5,0.53],'true_value':0.53}, 
                wheel_radii_dict={'values':[0.1],'true_value':0.12})
        init_robot_pose = {'x': -0.5, 'y': 0.0, 'theta': np.pi/2}
        experiment.setup_experiment(init_robot_pose)

        assert experiment.is_a_parametrized_model
        assert experiment.is_axle_length_param
        assert experiment.is_wheel_radius_param
        assert experiment.is_scenario_based
        assert not experiment.is_custom_weighted_scenario_based
        assert experiment.axle_probabilities == None
        assert experiment.wheel_probabilities == None
        assert experiment.true_axle_length == 0.53
        assert experiment.true_wheel_radius == 0.12
        assert experiment.regulation_mode
        assert not experiment.tracking_trajectory_mode
    def test_nine_wheel_radius_multival_param_axle_length_single_value_param(self):
        experiment = DifferentialDriveExperiment(
                axle_lengths_dict={'values':[0.5],'true_value':0.48}, 
                wheel_radii_dict={'values':[0.1,0.15,0.20],'true_value':0.20})
        init_robot_pose = {'x': -0.5, 'y': 0.0, 'theta': np.pi/2}
        experiment.setup_experiment(init_robot_pose)

        assert experiment.is_a_parametrized_model
        assert experiment.is_axle_length_param
        assert experiment.is_wheel_radius_param
        assert experiment.is_scenario_based
        assert not experiment.is_custom_weighted_scenario_based
        assert experiment.axle_probabilities == None
        assert experiment.wheel_probabilities == None
        assert experiment.true_axle_length == 0.48
        assert experiment.true_wheel_radius == 0.20
        assert experiment.regulation_mode
        assert not experiment.tracking_trajectory_mode
    def test_ten_wheel_radius_and_axle_length_multival_param(self):
        experiment = DifferentialDriveExperiment(
                axle_lengths_dict={'values':[0.48,0.5,0.53]}, 
                wheel_radii_dict={'values':[0.1,0.15,0.20]})
        init_robot_pose = {'x': -0.5, 'y': 0.0, 'theta': np.pi/2}
        experiment.setup_experiment(init_robot_pose)

        assert experiment.is_a_parametrized_model
        assert experiment.is_axle_length_param
        assert experiment.is_wheel_radius_param
        assert experiment.is_scenario_based
        assert not experiment.is_custom_weighted_scenario_based
        assert experiment.axle_probabilities == None
        assert experiment.wheel_probabilities == None
        assert experiment.true_axle_length == 0.48
        assert experiment.true_wheel_radius == 0.1
        assert experiment.regulation_mode
        assert not experiment.tracking_trajectory_mode
    def test_eleven_wheel_radius_and_axle_length_multival_param_with_true_value_for_sim(self):
        experiment = DifferentialDriveExperiment(
                axle_lengths_dict={'values':[0.48,0.5,0.53],'true_value':0.5}, 
                wheel_radii_dict={'values':[0.1,0.15,0.20],'true_value':0.20})
        init_robot_pose = {'x': -0.5, 'y': 0.0, 'theta': np.pi/2}
        experiment.setup_experiment(init_robot_pose)

        assert experiment.is_a_parametrized_model
        assert experiment.is_axle_length_param
        assert experiment.is_wheel_radius_param
        assert experiment.is_scenario_based
        assert not experiment.is_custom_weighted_scenario_based
        assert experiment.axle_probabilities == None
        assert experiment.wheel_probabilities == None
        assert experiment.true_axle_length == 0.5
        assert experiment.true_wheel_radius == 0.20
        assert experiment.regulation_mode
        assert not experiment.tracking_trajectory_mode
    #Custom weight tests => only scenario based experiments

    def test_twelve_axle_length_multival_param_wheel_radius_embedded(self):
        experiment = DifferentialDriveExperiment(
                axle_lengths_dict={'values':[0.48,0.5,0.53], 'probs':[0.35,0.25,0.4]}, 
                wheel_radii_dict={'values':[0.1]})
        init_robot_pose = {'x': -0.5, 'y': 0.0, 'theta': np.pi/2}
        experiment.setup_experiment(init_robot_pose)

        assert experiment.is_a_parametrized_model
        assert experiment.is_axle_length_param
        assert not experiment.is_wheel_radius_param
        assert experiment.is_scenario_based
        assert experiment.is_custom_weighted_scenario_based
        assert not experiment.axle_probabilities == None
        assert experiment.wheel_probabilities == None
        assert experiment.true_axle_length == 0.48
        assert experiment.true_wheel_radius == 0.1
        assert experiment.regulation_mode
        assert not experiment.tracking_trajectory_mode
    
    def test_thirteen_wheel_radius_multival_param_axle_length_embedded(self):
        experiment = DifferentialDriveExperiment(
                axle_lengths_dict={'values':[0.5]}, 
                wheel_radii_dict={'values':[0.1, 0.15, 0.20],'probs':[0.25,0.4,0.25]})
        init_robot_pose = {'x': -0.5, 'y': 0.0, 'theta': np.pi/2}
        experiment.setup_experiment(init_robot_pose)

        assert experiment.is_a_parametrized_model
        assert not experiment.is_axle_length_param
        assert experiment.is_wheel_radius_param
        assert experiment.is_scenario_based
        assert experiment.is_custom_weighted_scenario_based
        assert experiment.axle_probabilities == None
        assert not experiment.wheel_probabilities == None
        assert experiment.true_axle_length == 0.5
        assert experiment.true_wheel_radius == 0.1
        assert experiment.regulation_mode
        assert not experiment.tracking_trajectory_mode
    
    def test_fourteen_axle_length_multival_param_wheel_radius_single_value_param(self):
        experiment = DifferentialDriveExperiment(
                axle_lengths_dict={'values':[0.48,0.5,0.53], 'probs':[0.35,0.25,0.4]}, 
                wheel_radii_dict={'values':[0.1],'true_value':0.15})
        init_robot_pose = {'x': -0.5, 'y': 0.0, 'theta': np.pi/2}
        experiment.setup_experiment(init_robot_pose)

        assert experiment.is_a_parametrized_model
        assert experiment.is_axle_length_param
        assert experiment.is_wheel_radius_param
        assert experiment.is_scenario_based
        assert experiment.is_fully_parametrized_model
        assert experiment.is_custom_weighted_scenario_based
        assert not experiment.axle_probabilities == None
        assert not experiment.wheel_probabilities == None
        assert experiment.true_axle_length == 0.48
        assert experiment.true_wheel_radius == 0.15
        assert experiment.regulation_mode
        assert not experiment.tracking_trajectory_mode
    
    def test_fifteen_wheel_radius_multival_param_axle_length_single_value_param(self):
        experiment = DifferentialDriveExperiment(
                axle_lengths_dict={'values':[0.5],'true_value':0.48}, 
                wheel_radii_dict={'values':[0.1, 0.15, 0.20],'probs':[0.25,0.4,0.25]})
        init_robot_pose = {'x': -0.5, 'y': 0.0, 'theta': np.pi/2}
        experiment.setup_experiment(init_robot_pose)

        assert experiment.is_a_parametrized_model
        assert experiment.is_axle_length_param
        assert experiment.is_wheel_radius_param
        assert experiment.is_scenario_based
        assert experiment.is_custom_weighted_scenario_based
        assert not experiment.axle_probabilities == None
        assert len(experiment.axle_probabilities) == 1
        assert experiment.axle_probabilities[0] == 1
        assert not experiment.wheel_probabilities == None
        assert len(experiment.wheel_probabilities) == len(experiment.wheel_radii)
        assert experiment.true_axle_length == 0.48
        assert experiment.true_wheel_radius == 0.1
        assert experiment.regulation_mode
        assert not experiment.tracking_trajectory_mode

    def test_sixteen_axle_lengths_wheel_radius_multival_params(self):
        experiment = DifferentialDriveExperiment(
                axle_lengths_dict={'values':[0.48,0.5,0.53], 'probs':[0.35,0.25,0.4]}, 
                wheel_radii_dict={'values':[0.1, 0.15, 0.20],'probs':[0.25,0.4,0.25]})
        init_robot_pose = {'x': -0.5, 'y': 0.0, 'theta': np.pi/2}
        experiment.setup_experiment(init_robot_pose)

        assert experiment.is_a_parametrized_model
        assert experiment.is_axle_length_param
        assert experiment.is_wheel_radius_param
        assert experiment.is_scenario_based
        assert experiment.is_custom_weighted_scenario_based
        assert not experiment.axle_probabilities == None
        assert len(experiment.axle_probabilities) == len(experiment.axle_lengths)
        assert not experiment.wheel_probabilities == None
        assert len(experiment.wheel_probabilities) == len(experiment.wheel_radii)
        assert experiment.true_axle_length == 0.48
        assert experiment.true_wheel_radius == 0.1
        assert experiment.regulation_mode
        assert not experiment.tracking_trajectory_mode
    #those tests are for the forward term integration

    def test_seventeen_trajectory_tracking_single_trajectory(self):

        experiment = DifferentialDriveExperiment(
                axle_lengths_dict={'values':[0.5]}, 
                wheel_radii_dict={'values':[0.15]},
                tracking_trajectories=[{'L':0.5,'r':0.15,'path':[[1,2,3],[4,5,6]],'actions':[[-1,-2],[-3,-4]]}])
        init_robot_pose = {'x': -0.5, 'y': 0.0, 'theta': np.pi/2}
        experiment.setup_experiment(init_robot_pose)

        assert not experiment.is_a_parametrized_model
        assert not experiment.is_axle_length_param
        assert not experiment.is_wheel_radius_param
        assert not experiment.is_scenario_based
        assert not experiment.is_custom_weighted_scenario_based
        assert experiment.axle_probabilities == None
        assert experiment.wheel_probabilities == None
        assert experiment.true_axle_length == 0.5
        assert experiment.true_wheel_radius == 0.15

        assert not experiment.regulation_mode
        assert experiment.tracking_trajectory_mode

    def test_eighteen_trajectory_tracking_single_trajectory(self):

        ppo2_model_name = "ppo2_meters_redesigned_1"
        init_robot_pose = {'x': 0.12, 'y': -0.25, 'theta': -np.pi/2}
        obss, actions = bi.load_and_run_model(ppo2_model_name,1000,list(init_robot_pose.values()))
        experiment = DifferentialDriveExperiment(
                axle_lengths_dict={'values':[0.5]}, 
                wheel_radii_dict={'values':[0.15]},
                tracking_trajectories=[{'L':0.5,'r':0.15,'path':obss,'actions':actions}])
        
        experiment.setup_experiment(init_robot_pose)

        assert not experiment.is_a_parametrized_model
        assert not experiment.is_axle_length_param
        assert not experiment.is_wheel_radius_param
        assert not experiment.is_scenario_based
        assert not experiment.is_custom_weighted_scenario_based
        assert experiment.axle_probabilities == None  
        assert experiment.wheel_probabilities == None
        assert experiment.true_axle_length == 0.5
        assert experiment.true_wheel_radius == 0.15

        assert not experiment.regulation_mode
        assert experiment.tracking_trajectory_mode

    def test_nineteen_trajectory_tracking_single_trajectory(self):

        ppo2_model_name = "ppo2_meters_redesigned_1"
        init_robot_pose = {'x': 0.12, 'y': -0.25, 'theta': -np.pi/2}
        obss, actions = bi.load_and_run_model(ppo2_model_name,1000,list(init_robot_pose.values()))
        experiment = DifferentialDriveExperiment(
                axle_lengths_dict={'values':[0.5]}, 
                wheel_radii_dict={'values':[0.15,0.14]},
                tracking_trajectories=[{'L':0.5,'r':0.15,'path':obss,'actions':actions},
                                    {'L':0.5,'r':0.14,'path':obss,'actions':actions}
                                ])
        
        experiment.setup_experiment(init_robot_pose)

        assert experiment.is_a_parametrized_model
        assert not experiment.is_axle_length_param
        assert experiment.is_wheel_radius_param
        assert experiment.is_scenario_based
        assert not experiment.is_custom_weighted_scenario_based
        assert experiment.axle_probabilities == None  
        assert experiment.wheel_probabilities == None
        assert experiment.true_axle_length == 0.5
        assert experiment.true_wheel_radius == 0.15

        assert not experiment.regulation_mode
        assert experiment.tracking_trajectory_mode

    def test_twenty_trajectory_tracking_single_trajectory(self):

        #ppo2_model_name = "ppo2_meters_redesigned_1"
        init_robot_pose = {'x': 2, 'y': 0, 'theta': 0}
        #obss, actions = bi.load_and_run_model(ppo2_model_name,1000,list(init_robot_pose.values()))
        abs_max_ul_ur = 3
        max_robot_vel_l1 = bi.from_commands_to_robot_velocity(3,3,0.50,0.15)[0]
        max_robot_vel_l2 = bi.from_commands_to_robot_velocity(3,3,0.49,0.15)[0]
        robot_commands_l1 = bi.from_robot_velocity_to_commands(max_robot_vel_l1,0.0,0.50,0.15)
        robot_commands_l2 = bi.from_robot_velocity_to_commands(max_robot_vel_l2,0.0,0.49,0.15)
        print("Max lin vel if ax 0.50 {} check required robot_commands {}".format(max_robot_vel_l1,robot_commands_l1))
        print("Max lin vel if ax 0.49 {} check required robot_commands {}".format(max_robot_vel_l2,robot_commands_l2))
        trajectories = bi.compute_trajectories_x_eq_y_x_eq_min_y(500,0.50,0.49,0.15,init_pos=[0,0],abs_max_ul=abs_max_ul_ur,abs_max_ur=abs_max_ul_ur,delta_t=0.01)
        obss1 = trajectories[0]['path']
        actions1 = trajectories[0]['actions']
        obss2 = trajectories[1]['path']
        actions2 = trajectories[1]['actions']
        experiment = DifferentialDriveExperiment(
                axle_lengths_dict={'values':[0.5,0.49]}, 
                wheel_radii_dict={'values':[0.15]},
                tracking_trajectories=[{'L':0.5,'r':0.15,'path':obss1,'actions':actions1},
                                    {'L':0.49,'r':0.15,'path':obss2,'actions':actions2}
                                ]) 
        experiment.setup_experiment(init_robot_pose)

        assert experiment.is_a_parametrized_model
        assert experiment.is_axle_length_param
        assert not experiment.is_wheel_radius_param
        assert experiment.is_scenario_based
        assert not experiment.is_custom_weighted_scenario_based
        assert experiment.axle_probabilities == None  
        assert experiment.wheel_probabilities == None
        assert experiment.true_axle_length == 0.5
        assert experiment.true_wheel_radius == 0.15

        assert not experiment.regulation_mode
        assert experiment.tracking_trajectory_mode
    
    def test_twenty_one_cost_function_comparison(self):
        init_robot_coords = {'x': 2, 'y': 2}
        abs_max_ul_ur = 3
        trajectories = bi.compute_trajectories_x_eq_y_x_eq_min_y(500,0.50,0.49,0.15,
                                                    init_pos=list(init_robot_coords.values()),
                                                    abs_max_ul=abs_max_ul_ur,abs_max_ur=abs_max_ul_ur,
                                                    delta_t=0.01)
        obss1 = trajectories[0]['path']
        actions1 = trajectories[0]['actions']
        obss2 = trajectories[1]['path']
        actions2 = trajectories[1]['actions']
        init_robot_pose = {'x': 0, 'y': 2,'theta': np.pi/2}
        horizon_steps = 2
        experiment1 = DifferentialDriveExperiment(axle_lengths_dict={'values':[0.5]}, 
                                                wheel_radii_dict={'values':[0.15]},
                                                tracking_trajectories=[{'L':0.5,'r':0.15,'path':obss1,'actions':actions1}],
                                                n_horizon=horizon_steps)
        
        experiment1.setup_experiment(init_robot_pose)
        assert not experiment1.is_a_parametrized_model
        assert not experiment1.is_axle_length_param
        assert not experiment1.is_wheel_radius_param

        cost_function1 = experiment1.mpc.obj_expression
        print("*************************Experiment 1: single scenario L=0.50 ***************************")
        cost1 = bi.compute_cost_of_tracking_along_the_horizon(cost_function1,experiment1.mpc,init_robot_pose,trajectories[0:1],[[3,3],[3,3]])        
        
        experiment2 = DifferentialDriveExperiment(axle_lengths_dict={'values':[0.49]}, 
                                                wheel_radii_dict={'values':[0.15]},
                                                tracking_trajectories=[{'L':0.49,'r':0.15,'path':obss2,'actions':actions2}],
                                                n_horizon=horizon_steps)
    
        experiment2.setup_experiment(init_robot_pose)
        assert not experiment2.is_a_parametrized_model
        assert not experiment2.is_axle_length_param
        assert not experiment2.is_wheel_radius_param
        cost_function2 = experiment2.mpc.obj_expression
        
        print("*************************Experiment 2: single scenario L=0.49 ***************************")
        cost2 = bi.compute_cost_of_tracking_along_the_horizon(cost_function2,experiment2.mpc,init_robot_pose,trajectories[1:2],[[3,3],[3,3]])
        
        experiment3 = DifferentialDriveExperiment(axle_lengths_dict={'values':[0.5,0.49]}, 
                                                wheel_radii_dict={'values':[0.15]},
                                                tracking_trajectories=[{'L':0.5,'r':0.15,'path':obss1,'actions':actions1},
                                                                    {'L':0.49,'r':0.15,'path':obss2,'actions':actions2}],
                                                n_horizon=horizon_steps)
        
        experiment3.setup_experiment(init_robot_pose)
        cost_function3 = experiment3.mpc.obj_expression
        print("*************************Experiment 3: combining the two scenarios ***************************")
        #print("Size of X {}, shape of X {}".format(experiment3.mpc.opt_x.size, experiment3.mpc.opt_x.cat.shape))
        #print("Num of rows of x and length of horizon + 1 {} ".format(len(experiment3.mpc.opt_x['_x'])))
        #print("Num of columns of x and num of scenarios {}".format(len(experiment3.mpc.opt_x['_x'][0])))
        #print("Size of OPT_P {}, shape of OPT_P {}".format(experiment3.mpc.opt_p.size, experiment3.mpc.opt_p.cat.shape))
        #print("Num of rows of TVP and num of scenarios {} ".format(len(experiment3.mpc.opt_p['_tvp'])))
        #print("Num of columns of TVP and length of horizon + 1 {}".format(len(experiment3.mpc.opt_p['_tvp'][0])))
        cost3 = bi.compute_cost_of_tracking_along_the_horizon(cost_function3,experiment3.mpc,init_robot_pose,trajectories,[[3,3],[3,3]])
        tolerance = 0.01
        average_cost1_2 = 0.5 *cost1 + 0.5 *cost2
        assert abs(float( cost3 - average_cost1_2)) < tolerance
        print("Cost in the two scenario mpc {} Average of the cost of the two single scenario mpc {}".format(cost3, average_cost1_2))
        
        

        


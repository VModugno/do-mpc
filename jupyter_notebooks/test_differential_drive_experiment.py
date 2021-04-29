import numpy as np
import do_mpc
from DifferentialDriveExperiment import DifferentialDriveExperiment 
import baseline_dompc_utils

#https://docs.pytest.org/en/stable/getting-started.html
#pytest -q test_differential_drive_experiment.py

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
                axle_lengths_dict={'values':[0.48,0.5,0.53], 'probs':[0.35,0.25,0.4]}, 
                wheel_radii_dict={'values':[0.1, 0.15, 0.20],'probs':[0.25,0.4,0.25]},
                tracking_trajectories=[{'L':0.5,'r':0.53,'path':[[1,2,3],[4,5,6]],'actions':[[-1,-2],[-3,-4]]}])
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

        #assert experiment.tracking_trajectory_mode

    
    

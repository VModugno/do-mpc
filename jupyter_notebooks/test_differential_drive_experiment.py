import numpy as np
import do_mpc
from DifferentialDriveExperiment import DifferentialDriveExperiment 

#https://docs.pytest.org/en/stable/getting-started.html
#pytest -q test_differential_drive_experiment.py

class TestDifferentialDriveExperiment:
    
    def test_one_no_param_in_model(self):
        experiment = DifferentialDriveExperiment(
                axle_lengths_dict={'values':[0.5]}, 
                wheel_radii_dict={'values':[0.1]})
        init_robot_pose = {'x': -0.5, 'y': 0.0, 'theta': np.pi/2}
        experiment.setup_experiment(init_robot_pose)
        
        assert not experiment.is_a_parametrized_model
        assert not experiment.is_axle_length_param
        assert not experiment.is_wheel_radius_param
        assert not experiment.is_scenario_based
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
        assert experiment.axle_probabilities == None
        assert experiment.wheel_probabilities == None
        assert experiment.true_axle_length == 0.5
        assert experiment.true_wheel_radius == 0.11
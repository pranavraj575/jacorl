################################################################################
###| ENV_RESET_TEST.PY - test if environment reset works properly           |###
################################################################################

import gym
import jaco_gym
import random
import numpy as np 
import rospy
import ros_numpy

# from stable_baselines.common.env_checker import check_env

# first launch Jaco in Gazebo with
# roslaunch kinova_gazebo robot_launch_noRender_noSphere.launch kinova_robotType:=j2n6s300
# roslaunch kinova_gazebo robot_launch_render.launch kinova_robotType:=j2n6s300


rospy.init_node("kinova_client", anonymous=True, log_level=rospy.INFO)

env = gym.make('JacoGazebo-v1')

env.reset()

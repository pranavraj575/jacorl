################################################################################
###| RECORD_POINTS.PY - script we used to record joint positions on the     |###
###|         pysical robot in a variety of poistions so we could run a      |###
###|         camera capture script on those points in the real world + sim  |###
###|       - Used to generate real and sim image training data for CycleGAN |###
################################################################################

import gym
import jaco_gym
import numpy as np 
import rospy
import os
# from stable_baselines.common.env_checker import check_env

# first launch Jaco in Gazebo with
# roslaunch kinova_gazebo robot_launch_noRender_noSphere.launch kinova_robotType:=j2n6s300
# roslaunch kinova_gazebo robot_launch_render.launch kinova_robotType:=j2n6s300


rospy.init_node("recording_client", log_level=rospy.INFO)

env = gym.make('BasicJacoEnv-v0')

arm_degs=[]

def get_rotation():
    pos,_,_=env.get_joint_state()
    env.get_tip_coord()

def create_rotation_matrix(x_angle, y_angle, z_angle):
    # Define rotation matrix for each of the 3 dimensions, and overall rotation matrix is the result of matrix multiplying the 3
    x_rotation_matrix = np.array([[1, 0, 0], [0, np.cos(x_angle), -np.sin(x_angle)], [0, np.sin(x_angle), np.cos(x_angle)]])
    y_rotation_matrix = np.array([[np.cos(y_angle), 0, np.sin(y_angle)], [0, 1, 0], [-np.sin(y_angle), 0, np.cos(y_angle)]])
    z_rotation_matrix = np.array([[np.cos(z_angle), -np.sin(z_angle), 0], [np.sin(z_angle), np.cos(z_angle), 0], [0, 0, 1]])
    rotation_matrix = np.matmul(np.matmul(z_rotation_matrix, y_rotation_matrix), x_rotation_matrix)

    return rotation_matrix

while not input('Press enter to record, type to exit:'):
    # Z axis points down (and is element 2 of basis[:, 2] of wrist joint, where 1 points up and -1 points down)
    # X axis points to the door and is element 0 of basis [:, 1] (1 points to door, and -1 points to baxter)
    # Y axis points to the emeregency exit, where 1 points to the door and -1 points to the drone cage
    pos,_,_=env.get_joint_state()
    print("\n")
    x_angle, y_angle, z_angle = env.get_camera_rotation_and_position()
    print(create_rotation_matrix(x_angle, y_angle, z_angle))
    print(np.rad2deg(x_angle), np.rad2deg(y_angle), np.rad2deg(z_angle))
    print(env.get_tip_coord())
    # print(env.get_cartesian_points()[6])
    arm_degs.append(np.degrees(pos[:6]))

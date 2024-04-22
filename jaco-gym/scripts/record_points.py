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

aim=os.path.join('sample_points',input('save file name: ')+'.npy')

arm_degs=[]

while not input('Press enter to record, type to exit:'):
    pos,_,_=env.get_joint_state()
    arm_degs.append(np.degrees(pos[:6]))
    print(np.degrees(pos[:6]))
    print(env.get_camera_rotation_and_position()[-1])
np.save(aim,np.array(arm_degs))
print(np.load(aim).shape)

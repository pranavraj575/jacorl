################################################################################
###| LIMP.PY - script to make the physical robot go "limp"                  |###
###|         - used for being able to manually move a robot to a specific   |###
###|           position w/o resistence from the joints                      |###
################################################################################

import gym
import jaco_gym
import random
import numpy as np 
import rospy
import ros_numpy
from PIL import Image as IMG

# from stable_baselines.common.env_checker import check_env

# first launch Jaco in Gazebo with
# roslaunch kinova_gazebo robot_launch_noRender_noSphere.launch kinova_robotType:=j2n6s300
# roslaunch kinova_gazebo robot_launch_render.launch kinova_robotType:=j2n6s300


rospy.init_node("limp_client", log_level=rospy.INFO)

env = gym.make('BasicJacoEnv-v0')
FACTOR=.1
env.reset()
while True:
    pos,vel,eff=env.get_joint_state()
    pos=np.array(pos)
    arm_deg=np.degrees(pos[:6])
    fingy=pos[6]
    print("ORIG:",[round(d,2) for d in arm_deg],round(fingy,2))
    print('effort:',[round(d,2) for d in eff])
    arm_goal=arm_deg-FACTOR*np.array(eff[:6]) #Minus since we are letting the arm give up a bit
    fingy_goal=pos[6]-eff[6]*FACTOR
    
    print('result:',[round(d,2) for d in arm_goal],round(fingy_goal,2))
    rospy.sleep(1)
    env.move_arm(arm_goal)
    env.move_fingy(fingy_goal/.8)# .8 SINCE FINGERS POSITIONS GO ON [0,.8]

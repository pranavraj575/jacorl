import gym
import jaco_gym
import numpy as np 
import rospy
import os
import time
# from stable_baselines.common.env_checker import check_env

# first launch Jaco in Gazebo with
# roslaunch kinova_gazebo robot_launch_noRender_noSphere.launch kinova_robotType:=j2n6s300
# roslaunch kinova_gazebo robot_launch_render.launch kinova_robotType:=j2n6s300


rospy.init_node("limp_client", log_level=rospy.INFO)

env = gym.make('BasicJacoEnv-v0')
env.reset()
aim=os.path.join('sample_points','points.npy')
save_dir=os.path.join('img_data','simulation')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

wait=.5
points=np.load(aim)

for p in points:
    env.move_arm(p)
    rospy.sleep(.5)
    t=str(time.time()).replace('.','_') # unique name
    env.save_image(os.path.join(save_dir,t+'.jpeg'))
    

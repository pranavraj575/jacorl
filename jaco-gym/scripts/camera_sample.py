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

wait=.3
points=np.load(aim)
finger_samples=3
noise_samples=10
noise_bounds=(  (-5,5), #random movement of each joint chosen from these bounds , put 0,0 for none
                (-5,5),
                (-5,5),
                (-5,5),
                (-5,5),
                (-180,180),) # this is the rotation, can do whatever

for p in points:
    for _ in range(noise_samples):
        loc=[]
        for i in range(len(p)):
            l,h=noise_bounds[i]
            noise=np.random.random()*(h-l)+l
            loc.append(p[i]+noise)
        env.move_arm(loc)
        
        for _ in range(finger_samples): # pinch/unpinch fingers a random amount
            fingy_pos=np.random.random()
            env.move_fingy(fingy_pos) 
            rospy.sleep(wait)
            t=str(time.time()).replace('.','_') # unique name
            env.save_image(os.path.join(save_dir,t+'.jpeg'))
    

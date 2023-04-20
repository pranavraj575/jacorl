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


rospy.init_node("camera_sampler", log_level=rospy.INFO)
filename=input('file name (without .npy): ')

env = gym.make(input('env name: '))#('JacoCupsGazebo-v0')
#JacoStackCupsGazebo()

aim=os.path.join('sample_points',filename+'.npy')
save_dir=os.path.join('img_data','simulation')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

wait=.3
points=np.load(aim)
finger_samples=1
noise_samples=1
noise_bounds=(  (-2,2), #random movement of each joint chosen from these bounds , put 0,0 for none
                (-.5,.5),
                (-.5,.5),
                (-2,2),
                (-1,1),
                (-180,180),) # this is the rotation, can do whatever
noise_bounds=[[0,0]]*6

base_angles=np.arange(-3,4)*5 # the angles to add to the base rotation, since all of our samples should be along a line, this multiplies data points by number of angles of base of arm

for a in base_angles:
    env.reset()
    for p in points:
        for _ in range(noise_samples):
            loc=[]
            for i in range(len(p)):
                l,h=noise_bounds[i]
                noise=np.random.random()*(h-l)+l
                loc.append(p[i]+noise+(a if i==0 else 0))
            env.move_arm(loc)
            
            for _ in range(finger_samples): # pinch/unpinch fingers a random amount
                fingy_pos=np.random.random()
                env.move_fingy(fingy_pos) 
                rospy.sleep(wait)
                t=str(time.time()).replace('.','_') # unique name
                env.save_image(os.path.join(save_dir,t+'.jpeg'))
        

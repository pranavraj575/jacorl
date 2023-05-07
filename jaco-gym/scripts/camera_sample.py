################################################################################
###| CAMERA_SAMPLE.PY - runs robot through captured joint coordinates       |###
###|      and records camera images at each loctation for CycleGAN training |###
################################################################################

import gym
import jaco_gym
import numpy as np 
import rospy
import os
import time
import rostopic
# from stable_baselines.common.env_checker import check_env

# first launch Jaco in Gazebo with
# roslaunch kinova_gazebo robot_launch_noRender_noSphere.launch kinova_robotType:=j2n6s300
# roslaunch kinova_gazebo robot_launch_render.launch kinova_robotType:=j2n6s300


rospy.init_node("camera_sampler", log_level=rospy.INFO)


try:
    stuff=rostopic.get_topic_list()
except: 
    raise Exception('ROS is not running, did u open either the gazebo simulator or the kortex driver (if real robot)? you could also just try waiting like 2 seconds before running this omg')
    
def recurse_search_str(thing,search):
    if thing is None: 
        return search is None
    if type(thing)==str:
        return search in thing
    return any([recurse_search_str(t,search) for t in thing])

if recurse_search_str(stuff,'gazebo'):
    env_id='JacoCupsGazebo-v0'
    print("SIMULATION DETECTED, using env",env_id)
    SIM=True
else:
    env_id='BasicJacoEnv-v0'
    print("SIMULATION NOT DETECTED, using env",env_id)
    SIM=False
          

env = gym.make(env_id)

HAVE_CUPS=False
print("IMPORTANT: HAVE_CUPS is",HAVE_CUPS,'this means','' if HAVE_CUPS else 'DONT' ,'PUT THE CUPS AROUND TABLE')
print('ALSO, SIM is',SIM,'make sure this is correct, or save names will be annoying')
filename=input('file name (without .npy, leave blank for all): ')
finger_samples=3
noise_samples=5
wait=.3
noise_bounds=(  (-5,5), #random movement of each joint chosen from these bounds , put 0,0 for none
                (-.5,0),
                (0,.5),
                (-1,1),
                (-1,1),
                (-180,180),) # this is the rotation, can do whatever
#noise_bounds=[[0,0]]*5+[[-180,180]]

base_angles=np.arange(0,5)*10 # the angles to add to the base rotation, since all of our samples should be along a line, this multiplies data points by number of angles of base of arm

env_string='JacoCupsGazebo-v0' if SIM else 'BasicJacoEnv-v0'
env = gym.make(env_string)
#JacoStackCupsGazebo()
if SIM:
    env.set_cup_ranges((0.0,1.),env.table_y_range)

if filename:
    files=[filename+'.npy']
else:
    files=os.listdir('sample_points')
for flnm in files:
    aim=os.path.join('sample_points',flnm)
    name=flnm[:flnm.index('.')]
    
    save_dir=os.path.join('Image-Data',
                            'simulation' if SIM else 'real',
                            'POSITIVE' if HAVE_CUPS else 'NEGATIVE',
                            name)
    if not os.path.exists(save_dir):
        raise Exception("DIRECTORY",save_dir,"DOES NOT EXIST, call \"git clone https://github.com/sophiazalewski1/Image-Data\"")
    
    points=np.load(aim)
    
    for a in base_angles:
        env.reset()
        first=True
        for p in points:
            for _ in range(noise_samples):
                if SIM:
                    if HAVE_CUPS:
                        env.reset_cups(.8,.18,.02) # .8 standing, .18 upside down, .02 fallen
                    else:
                        env.move_cups(((-10.,-1.,-.5),(-10.,1.,-.7),(-10.,0.,-.9)))
                loc=[]
                for i in range(len(p)):
                    l,h=noise_bounds[i]
                    noise=np.random.random()*(h-l)+l
                    loc.append(p[i]+noise+(a if i==0 else 0))
                if first:
                    loc_up=loc.copy()
                    loc_up[1]=0
                    env.move_arm(loc_up)
                    first=False
                env.move_arm(loc)
                
                for _ in range(finger_samples): # pinch/unpinch fingers a random amount
                    fingy_pos=np.random.random()
                    env.move_fingy(fingy_pos) 
                    rospy.sleep(wait)
                    t=str(time.time()).replace('.','_') # unique name
                    env.save_image(os.path.join(save_dir,'base_angle_'+str(a).replace('.','_')+'_time_'+t+'.jpeg'))
            

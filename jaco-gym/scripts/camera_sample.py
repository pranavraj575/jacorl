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
SIM=True
HAVE_CUPS=True
print("IMPORTANT: HAVE_CUPS is",HAVE_CUPS,'this means','' if HAVE_CUPS else 'DONT' ,'PUT THE CUPS AROUND TABLE')
print('ALSO, SIM is',SIM,'make sure this is correct, or save names will be annoying')
filename=input('file name (without .npy, leave blank for all): ')
finger_samples=3
noise_samples=1
wait=.3
noise_bounds=(  (-2,2), #random movement of each joint chosen from these bounds , put 0,0 for none
                (-.5,.5),
                (-.5,.5),
                (-2,2),
                (-1,1),
                (-180,180),) # this is the rotation, can do whatever
noise_bounds=[[0,0]]*5+[[-180,180]]

base_angles=np.arange(-2,4)*10 # the angles to add to the base rotation, since all of our samples should be along a line, this multiplies data points by number of angles of base of arm

env_string='JacoCupsGazebo-v0' if SIM else 'BasicJacoEnv-v0'
env = gym.make(env_string)
#JacoStackCupsGazebo()

if filename:
    files=[filename+'.npy']
else:
    files=os.listdir('sample_points')
for flnm in files:
    aim=os.path.join('sample_points',flnm)
    name=flnm[:flnm.index('.')]
    
    save_dir=os.path.join('img_data',
                            'simulation' if SIM else 'real',
                            'POSITIVE' if HAVE_CUPS else 'NEGATIVE',
                            name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    points=np.load(aim)
    
    for a in base_angles:
        env.reset()
        first=True
        for p in points:
            for _ in range(noise_samples):
                if SIM:
                    env.reset_cups()
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
            

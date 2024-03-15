import gym
import jaco_gym
import random
import numpy as np 
import rospy
import ros_numpy
from PIL import Image as IMG
import rostopic

rospy.init_node("test_client", log_level=rospy.INFO)

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
    env_id='JacoJengaGazebo-v0'
    print("SIMULATION DETECTED, using env",env_id)
else:
    env_id='BasicJacoEnv-v0'
    print("SIMULATION NOT DETECTED, using env",env_id)
          

env = gym.make(env_id)


    


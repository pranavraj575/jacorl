import gym
import jaco_gym
import random
import numpy as np 
import rospy
import ros_numpy
from PIL import Image as IMG
import rostopic
import pickle
import os

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
    env_id='JacoCupsGazebo-v0'
    print("SIMULATION DETECTED, using env",env_id)
else:
    env_id='BasicJacoEnv-v0'
    print("SIMULATION NOT DETECTED, using env",env_id)
          

env = gym.make(env_id)

## It will check your custom environment and output additional warnings if needed
# print("starting check")
# check_env(env, warn=True)
# print("check done")

print('Action space:')
print(env.action_space)
print(env.action_space.high)
print(env.action_space.low)

print('State space:')
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)

# obs = env.reset()
# action = env.action_space.sample()
# print('random action:', action)
# obs, reward, done, info = env.step(action)

render_flag=True

old_raw = None

for episode in range(3):
    #run rostopic list to get all topics being published (while robot running)
    obs = env.reset()
    # env.save_image(str(episode)+".jpg",mode='depth')
    # env.save_image(str(episode)+"_color.jpg", mode='color')
    pick_pos = env.get_object_dict()['cup1']
    place_pos = env.get_object_dict()['cup2']
    
    color_img = env.get_image_numpy(mode='color')
    depth_img = env.get_image_numpy(mode='depth')
    sample = {	"pick_action" : pick_pos,
    		"place_action" : place_pos,
    		"color_img" : color_img,
    		"depth_img" : depth_img }
	
    with open(f"task{episode}.pkl", "wb") as f:
    	pickle.dump(sample, f)
env.close()


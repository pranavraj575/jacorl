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
    rewards = []
    #env.save_image(str(episode)+".jpg")
    #img_numpy = env.robot.get_image_numpy()
    #print(img_numpy.shape)
    #img = env.robot.get_image_PIL()
    
    #env.robot.save_image("myimg.jpeg")

    for t in range(5):

        action = env.action_space.sample()
        #action = [0,0,0,0,0,0,-1+2*t/4] # -1 to 1
        print("last joint angle is %f\n",-1+2*t/4)
        obs, reward, done, info = env.step(action)
        #print(env.robot_intersects_self())

        # print("timestep:", t)
        # print("action: ", action)
        # print("observation: ", obs)
        # print("reward: ", reward)
        # print("done: ", done)
        # print("info: ", info)

        if done:
            rewards.append(reward)
            break

    print("Episode: {}, Cumulated reward: {}".format(episode, sum(rewards)))
    print("******************")

env.close()


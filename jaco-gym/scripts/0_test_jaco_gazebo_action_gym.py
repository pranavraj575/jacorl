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


rospy.init_node("kinova_client", anonymous=True, log_level=rospy.INFO)

env = gym.make('JacoGazebo-v1')

## It will check your custom environment and output additional warnings if needed
# print("starting check")
# check_env(env, warn=True)
# print("check done")

#test

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

for episode in range(3):
    #run rostopic list to get all topics being published (while robot running)
    obs = env.reset()
    rewards = []
    print('IMAgE:')
    camera_info,raw,compressed=env.robot.get_image()
    # camera_info
    # robot = 
    print('image recieved')
    print("camera info")
    #print("-------------")
    #print(camera_info)
    print("camera raw")
    img_numpy = ros_numpy.numpify(raw)
    print(img_numpy.shape)
    img = IMG.fromarray(img_numpy, "RGB")
    img.save("myimg.jpeg")
    #print(raw.shape())
    #print(cam)

    for t in range(5):

        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)

        print("timestep:", t)
        print("action: ", action)
        print("observation: ", obs)
        print("reward: ", reward)
        print("done: ", done)
        print("info: ", info)

        if done:
            rewards.append(reward)
            break

    print("Episode: {}, Cumulated reward: {}".format(episode, sum(rewards)))
    print("******************")

env.close()


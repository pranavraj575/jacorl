################################################################################
###| GET_IMAGE_DATA.PY - saves camera image data from robot                 |###
################################################################################

import gym
import jaco_gym
import random
import numpy as np 
import rospy
import ros_numpy
from PIL import Image
import os

rospy.init_node("kinova_client", anonymous=True, log_level=rospy.INFO)
env = gym.make('JacoGazebo-v1')
render_flag=True

# Create save image dir
log_dir = "../image_captures/"
os.makedirs(log_dir, exist_ok=True)


def capture_img(id):
    print("Capturing image " + id)
    camera_info,raw,compressed=env.robot.get_image()
    img_numpy = ros_numpy.numpify(raw)
    img = Image.fromarray(img_numpy, "RGB")
    img.save("../image_captures/"+id+".jpeg")

# for episode in range(5):
    
#     print("Running episode\n")
#     obs = env.reset()
#     rewards = []
    
#     for t in range(2):
#         print("heyo!")
#         action = env.action_space.sample()
#         obs, reward, done, info = env.step(action)
#         capture_img(str(t))

# env.close()

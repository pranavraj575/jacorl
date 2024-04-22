################################################################################
###| GET_IMAGE_DATA.PY - saves camera image data from robot                 |###
################################################################################

import gym
import jaco_gym
import random
import numpy as np 
import rospy
import ros_numpy
import cv2
from PIL import Image
from scipy import interpolate
import os

rospy.init_node("kinova_client", anonymous=True, log_level=rospy.INFO)
env = gym.make('BasicJacoEnv-v0')
render_flag=True

# Create save image dir
log_dir = "../image_captures/"
os.makedirs(log_dir, exist_ok=True)


def capture_img(id):
    print("Capturing image " + id)
    color=env.get_image_numpy(mode='color',cam_type="attached")
    depth=env.get_image_numpy(mode='depth',cam_type='attached')
    X = np.linspace(0, 1280, 1280)
    Y = np.linspace(0, 720, 720)
    x, y = np.meshgrid(X, Y)


    f_red = interpolate.interp2d(X, Y, color[:,:,0])
    f_green = interpolate.interp2d(X, Y, color[:,:,1])
    f_blue = interpolate.interp2d(X, Y, color[:,:,2])
    Xnew = np.linspace(0, 1280, 480)
    Ynew = np.linspace(0, 720, 270)
    
    new_red = f_red(Xnew,Ynew)
    new_green = f_green(Xnew,Ynew)
    new_blue = f_blue(Xnew, Ynew)

    color = np.dstack([new_blue, new_green, new_red])
    cv2.imshow("img", color / 255)
    cv2.waitKey(0)
    img = Image.fromarray(color / 255, "RGB")
    depth_img = Image.fromarray(depth)
    print(np.min(depth_img), np.max(depth_img))
    cv2.imwrite("../image_captures/"+id+".jpeg", color)
    depth_img.save("../image_captures/"+id+"_depth.png")
# for episode in range(5):
    
#     print("Running episode\n")
#     obs = env.reset()
#     rewards = []
    
#     for t in range(2):
#         print("heyo!")
#         action = env.action_space.sample()
#         obs, reward, done, info = env.step(action)
#         capture_img(str(t))
capture_img(str(0))
# env.close()

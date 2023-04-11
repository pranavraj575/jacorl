#! /usr/bin/env python

import actionlib
import rospy
import numpy as np
import random

from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal, JointTrajectoryControllerState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from gazebo_msgs.msg import LinkStates, ModelState
from std_srvs.srv import Empty
from sensor_msgs.msg import JointState, CameraInfo, Image, CompressedImage
from geometry_msgs.msg import Pose, Point


rospy.init_node("kinova_client")

#action_address = "/j2n6s300/effort_joint_trajectory_controller/follow_joint_trajectory"

action_address = "/j2n6s300/effort_finger_trajectory_controller/follow_joint_trajectory"
client = actionlib.SimpleActionClient(action_address, FollowJointTrajectoryAction)
client.wait_for_server()
goal = FollowJointTrajectoryGoal()
    


#trajectory_msg.joint_names = [
#            "j2n6s300_joint_1", 
#            "j2n6s300_joint_2", 
#            "j2n6s300_joint_3",
#            "j2n6s300_joint_4", 
#            "j2n6s300_joint_5", 
#            "j2n6s300_joint_6"
#            ]
#points_msg.positions = [ 0,3.14, 3.14, 0, 0, 0]


#points_msg.velocities = [0, 0, 0, 0, 0, 0]
#points_msg.accelerations = [0, 0, 0, 0, 0, 0]
#points_msg.effort = [0, 0, 0, 0, 0, 0]

grab=input('grabbyness (looks like this specific way is on [0,pi/2]): ')

while grab:
    trajectory_msg = JointTrajectory()
    
    trajectory_msg.joint_names = [
                "j2n6s300_joint_finger_1", 
                "j2n6s300_joint_finger_2", 
                "j2n6s300_joint_finger_3"]
                
    points_msg = JointTrajectoryPoint()
    points_msg.positions=[float(grab)]*3#[0,0,0]
    
    points_msg.velocities = [0, 0,0]
    points_msg.accelerations = [0, 0, 0]
    points_msg.effort = [0, 0, 0]



    points_msg.time_from_start = rospy.Duration(0.01)
    
    # fill in points message of the trajectory message
    trajectory_msg.points = [points_msg]
    
    # fill in trajectory message of the goal
    goal.trajectory = trajectory_msg
    
    # self.client.send_goal_and_wait(goal)
    client.send_goal(goal)
    client.wait_for_result()
    grab=input('grabbyness: ')

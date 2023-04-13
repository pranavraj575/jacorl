#! /usr/bin/env python

import actionlib
import rospy
import numpy as np
import random
import math
import ros_numpy
from PIL import Image as IMG

from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal, JointTrajectoryControllerState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from gazebo_msgs.msg import LinkStates, ModelState, ModelStates
from std_srvs.srv import Empty
from sensor_msgs.msg import JointState, CameraInfo, Image, CompressedImage
from geometry_msgs.msg import Pose, Point, Quaternion
from scipy.spatial.transform import Rotation


class JacoGazeboActionClient:

    def __init__(self):
        
        action_address = "/j2n6s200/effort_joint_trajectory_controller/follow_joint_trajectory"
        self.client = actionlib.SimpleActionClient(action_address, FollowJointTrajectoryAction)
        fingy_addy = "/j2n6s200/effort_finger_trajectory_controller/follow_joint_trajectory"
        self.finger_client = actionlib.SimpleActionClient(fingy_addy, FollowJointTrajectoryAction)
        self.pub_topic = '/gazebo/set_model_state'
        self.pub = rospy.Publisher(self.pub_topic, ModelState, queue_size=1)
        self.doota={}
        def _call_model_data(data):
            self.doota={}
            for i in range(len(data.name)):
                self.doota[data.name[i]]=data.pose[i]
        self.sub_topic="/gazebo/model_states"
        self.sub=rospy.Subscriber(self.sub_topic,ModelStates,_call_model_data)

        # Unpause the physics
        rospy.wait_for_service('/gazebo/unpause_physics')
        unpause_gazebo = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        unpause_gazebo()

    def sleepy(self,sec=2):
        rospy.sleep(sec)      # wait for 2s
    def move_finger(self,grippiness):
        # ADDED GRIPPY BOY, single number [0,pi/2] that grips
        
        # GRIPPY 
        
        
        finger_list= [grippiness]*2 # note: controlls all fingys at once, can change if necessary  
        
        self.finger_client.wait_for_server()

        fingy_goal = FollowJointTrajectoryGoal()
        
        fingy_trajectory_msg = JointTrajectory()
        fingy_trajectory_msg.joint_names = [
                "j2n6s200_joint_finger_1", 
                "j2n6s200_joint_finger_2", 
                #"j2n6s200_joint_finger_3"
                ]
        
        fingy_points_msg = JointTrajectoryPoint()
        
          
        fingy_points_msg.positions = finger_list  
        fingy_points_msg.velocities = [0, 0]#, 0]
        fingy_points_msg.accelerations = [0, 0]#, 0]
        fingy_points_msg.effort = [0, 0]#, 0]
        fingy_points_msg.time_from_start = rospy.Duration(0.01)
        
        fingy_trajectory_msg.points = [fingy_points_msg]
        
        fingy_goal.trajectory = fingy_trajectory_msg

        self.finger_client.send_goal(fingy_goal)
        self.finger_client.wait_for_result()

    def move_arm(self, points_list):
        #returns the difference, measure of the movement created
            
        # # Unpause the physics
        # rospy.wait_for_service('/gazebo/unpause_physics')
        # unpause_gazebo = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        # unpause_gazebo()

        self.client.wait_for_server()
        
        old_position=rospy.wait_for_message("/j2n6s200/joint_states", JointState).position[:6] # just the first 6

        goal = FollowJointTrajectoryGoal()    

        # We need to fill the goal message with its components
        #         
        # check msg structure with: rosmsg info FollowJointTrajectoryGoal
        # It is composed of 4 sub-messages:
        # "trajectory" of type trajectory_msgs/JointTrajectory 
        # "path_tolerance" of type control_msgs/JointTolerance
        # "goal_tolerance" of type control_msgs/JointTolerance
        # "goal_time_tolerance" of type duration

        trajectory_msg = JointTrajectory()
        # check msg structure with: rosmsg info JointTrajectory
        # It is composed of 3 sub-messages:
        # "header" of type std_msgs/Header 
        # "joint_names" of type string
        # "points" of type trajectory_msgs/JointTrajectoryPoint

        trajectory_msg.joint_names = [
            "j2n6s200_joint_1", 
            "j2n6s200_joint_2", 
            "j2n6s200_joint_3", 
            "j2n6s200_joint_4", 
            "j2n6s200_joint_5", 
            "j2n6s200_joint_6"
            ]

        points_msg = JointTrajectoryPoint()
        # check msg structure with: rosmsg info JointTrajectoryPoint
        # It is composed of 5 sub-messages:
        # "positions" of type float64
        # "velocities" of type float64
        # "accelerations" of type float64
        # "efforts" of type float64
        # "time_from_start" of type duration
        points_msg.positions = points_list
        points_msg.velocities = [0, 0, 0, 0, 0, 0]
        points_msg.accelerations = [0, 0, 0, 0, 0, 0]
        points_msg.effort = [0, 0, 0, 0, 0, 0]
        points_msg.time_from_start = rospy.Duration(0.01)

        # fill in points message of the trajectory message
        trajectory_msg.points = [points_msg]

        # fill in trajectory message of the goal
        goal.trajectory = trajectory_msg

        # self.client.send_goal_and_wait(goal)
        self.client.send_goal(goal)
        self.client.wait_for_result()
        #self.sleepy()
        
        diff1=(points_list-old_position)%(2*np.pi)# since angular, take mod
        diff2=(points_list-old_position)%(-2*np.pi)# other direction (python mod returns sign of divisor)
        
        diff=[min(diff1[i],diff2[i], key=abs) 
                for i in range(len(diff1))] #takes the minimum mod 2 pi since this is angular
        return diff

        # return self.client.get_state()
        
    def move_cups(self, positions,orientations=None):
        cup_names = ["cup1", "cup2", "cup3"]
        for zs in [[-1]*3,[p[2] for p in positions]]:
            for i in range(len(cup_names)):
                model_state_msg = ModelState()
                pose_msg = Pose()
                point_msg = Point()
                
                rot_msg=Quaternion()#default no rotation
                
                if orientations:
                  (roll,pitch,yaw)=orientations[i]
                  stuff=Rotation.from_euler('xyz',(roll,pitch,yaw)).as_quat()
                  (rot_msg.x,rot_msg.y,rot_msg.z,rot_msg.w)=stuff
                
                (x,y,_)=positions[i]
                point_msg.x = x
                point_msg.y = y
                point_msg.z = zs[i]
                pose_msg.position = point_msg
                
                pose_msg.orientation = rot_msg
                
                model_state_msg.model_name = cup_names[i]
                model_state_msg.pose = pose_msg
                model_state_msg.reference_frame = "world"
                self.pub.publish(model_state_msg)
                rospy.sleep(.01)


    def cancel_move(self):
        self.client.cancel_all_goals()
    
    def get_image(self):
        camera_info = rospy.wait_for_message("/wristcam/camera_info", CameraInfo)
        raw = rospy.wait_for_message("/wristcam/image_raw",Image)
        compressed = rospy.wait_for_message("/wristcam/image_raw/compressed",CompressedImage)
        return camera_info,raw,compressed
        
    def get_image_numpy(self):
        camera_info,raw,compressed=self.get_image()
        return ros_numpy.numpify(raw)
        
    def get_image_PIL(self):
        img_numpy=self.get_image_numpy()
        return IMG.fromarray(img_numpy, "RGB")
    
    def save_image(self,filee):
        img=self.get_image_PIL()
        img.save(filee)

    def read_state_old(self):
        self.status = rospy.wait_for_message("/j2n6s200/effort_joint_trajectory_controller/state", JointTrajectoryControllerState)
        
        # convert tuple to list and concatenate
        self.state = list(self.status.actual.positions) + list(self.status.actual.velocities)
        # also self.status.actual.accelerations, self.status.actual.effort

        return self.state
    
    def get_object_data(self):
        return self.doota
    def get_obs(self):
        return self.read_state_priviledged()
    def get_obs_dim(self):
        return 31
    def read_state(self):
        self.status = rospy.wait_for_message("/j2n6s200/joint_states", JointState)
        
        self.joint_names = self.status.name
        # print(self.joint_names)

        self.pos = self.status.position
        self.vel = self.status.velocity
        self.eff = self.status.effort

        # return self.status
        return np.asarray(self.pos + self.vel + self.eff)
    
    def read_state_priviledged(self):
        self.status = rospy.wait_for_message("/j2n6s200/joint_states", JointState)
        self.joint_names = self.status.name
        self.pos = self.status.position
        self.vel = self.status.velocity
        self.eff = self.status.effort[6:8] # Effort of joints 7 and 8 = fingertip efforts

        state_tuple = self.pos + self.vel + self.eff

        self.cup_positions = []
        obj_data = self.get_object_data()
        cups = ["cup1","cup2","cup3"]
        for cup in cups:
            pos = obj_data[cup].position
            state_tuple = state_tuple + (pos.x, pos.y, pos.z)

        # Array values 0-9 = position angles of each joint
        #              10-19 = velocity of each joint
        #              20-21 = effort of each fingertip 
        #              22-30 = (x,y,z) posititon of each of the 3 cups
        return np.asarray(state_tuple)


    def read_state_simple(self):
        """
        read state of the joints only (not the finglers) + removed the efforts
        """

        self.status = rospy.wait_for_message("/j2n6s200/joint_states", JointState)
        
        self.joint_names = self.status.name[:6]
        # print(self.joint_names)

        self.pos = self.status.position[:6]
        self.vel = self.status.velocity[:6]

        # return self.status
        return np.asarray(self.pos + self.vel)

    def get_finger_coords(self):
        self.status = rospy.wait_for_message("/gazebo/link_states", LinkStates)
        self.joint_names = self.status.name
        self.pos = self.status.pose
        return ([self.status.pose[7].position.x, self.status.pose[7].position.y, self.status.pose[7].position.z],
               [self.status.pose[8].position.x, self.status.pose[8].position.y, self.status.pose[8].position.z])


    def get_tip_coord(self):
        self.status = rospy.wait_for_message("/gazebo/link_states", LinkStates)
        # see also topic /tf

        self.joint_names = self.status.name
        self.pos = self.status.pose

        # BE CAREFUL: joint number changes if I add a sphere!
        # print(self.joint_names[8])
        # print(self.status.pose[8].position)


        # for i in range(14):
        #     print(i)
        #     print("joint:")
        #     print(self.joint_names[i])
        #     print("pose:")
        #     print(self.status.pose[i])

        return [self.status.pose[8].position.x, self.status.pose[8].position.y, self.status.pose[8].position.z]


# rospy.init_node("kinova_client")

# client = JacoGazeboActionClient()
# client.cancel_move()
# client.move_arm([3, 1.57, 3.14, 0, 0, 0])

# # client.move_sphere([1, 1, 1])

# print(client.read_state_simple())
# # print(client.read_state2())
# print(client.get_tip_coord())   # PB: reading coordinate doesn't wait until the arm has finished moving. SOLUTION: wait for 2s. To improve.

# client.read_state()

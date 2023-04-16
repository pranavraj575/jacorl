#!/usr/bin/env python
###
# KINOVA (R) KORTEX (TM)
#
# Copyright (c) 2019 Kinova inc. All rights reserved.
#
# This software may be modified and distributed 
# under the terms of the BSD 3-Clause license.
#
# Refer to the LICENSE file for details.
#
###
import gym
import sys
import rospy
import time
import numpy as np
from PIL import Image as IMG

from kortex_driver.srv import Base_ClearFaults, ReadAction, ExecuteAction, SetCartesianReferenceFrame,SendGripperCommand,OnNotificationActionTopic
from kortex_driver.srv import GetProductConfiguration, ValidateWaypointList,OnNotificationActionTopicRequest,ReadActionRequest,ExecuteActionRequest,SendGripperCommandRequest
from kortex_driver.msg import ActionNotification, ActionEvent,Finger,GripperMode

from sensor_msgs.msg import JointState, Image
import ros_numpy
# to be run either connected through real arm (make sure driver is run)
# or run with gazebo launch being run


class JacoEnv(gym.Env):
    def __init__(self,
                    ROBOT_NAME='my_gen3',
                    CAM_SPACE='camera', #call will look for /CAM_SPACE/color/image_raw and /CAM_SPACE/depth/image_raw
                    init_pos=(0,15,230,0,55,90), #HOME position
                    differences=(15,15,15,15,15,15), # angular movement allowed at each joint per action
                    bounds=# hard bounds for each joint
                      (
                        None, #UNBOUNDED, arm can rotate
                        (240,120),#this goes about (230, 130) IRL with 0 being straight up, about 130 degrees each side. in simulation, 180 is straight up
                        (220,140),# IRL (212,147) with  0 straight up, about 140 each side. in simulation, 180 is straight up
                        None, # UNBOUNDED
                        (235,115), # (239,120) with 0 straight up, about 115 each side. In simulation, 0 is still straight up
                        None, # UNBOUNDED
                      )
                    ):
    
        self.action_dim=7
        self.obs_dim=self.get_obs_dim()
    
        self.init_pos=init_pos
        
        self.diffs=differences
        
        self.BOUNDS=bounds
        #BOUNDS HERE FOR EACH JOINT in degrees
        
        
        high = np.ones([self.action_dim])
        self.action_space = gym.spaces.Box(-high, high)
        
        high = np.inf * np.ones([self.obs_dim])
        self.observation_space = gym.spaces.Box(-high, high)
    
        joint_namespace='/'+ROBOT_NAME+'/joint_states'
        
        def joint_callback(data):
            self.joint_data=data
        self.joint_sub=rospy.Subscriber(joint_namespace,JointState,joint_callback)
        
        def _camera_img_callback(data):
            self.camera_img = data
        self.image_sub=rospy.Subscriber("/"+CAM_SPACE+"/color/image_raw",Image,_camera_img_callback)
        
        #rospy.init_node('example_full_arm_movement_python')

        self.HOME_ACTION_IDENTIFIER = 2

        # Get node params
        self.robot_name = rospy.get_param('~robot_name', "my_gen3")
        self.degrees_of_freedom = rospy.get_param("/" + self.robot_name + "/degrees_of_freedom", 6)
        self.is_gripper_present = rospy.get_param("/" + self.robot_name + "/is_gripper_present", False)

        rospy.loginfo("Using robot_name " + self.robot_name + " , robot has " + str(self.degrees_of_freedom) + " degrees of freedom and is_gripper_present is " + str(self.is_gripper_present))

        # Init the action topic subscriber
        self.action_topic_sub = rospy.Subscriber("/" + self.robot_name + "/action_topic", ActionNotification, self.cb_action_topic)
        self.last_action_notif_type = None

        # Init the services
        clear_faults_full_name = '/' + self.robot_name + '/base/clear_faults'
        rospy.wait_for_service(clear_faults_full_name)
        self.clear_faults = rospy.ServiceProxy(clear_faults_full_name, Base_ClearFaults)

        read_action_full_name = '/' + self.robot_name + '/base/read_action'
        rospy.wait_for_service(read_action_full_name)
        self.read_action = rospy.ServiceProxy(read_action_full_name, ReadAction)

        execute_action_full_name = '/' + self.robot_name + '/base/execute_action'
        rospy.wait_for_service(execute_action_full_name)
        self.execute_action = rospy.ServiceProxy(execute_action_full_name, ExecuteAction)

        set_cartesian_reference_frame_full_name = '/' + self.robot_name + '/control_config/set_cartesian_reference_frame'
        rospy.wait_for_service(set_cartesian_reference_frame_full_name)
        self.set_cartesian_reference_frame = rospy.ServiceProxy(set_cartesian_reference_frame_full_name, SetCartesianReferenceFrame)

        send_gripper_command_full_name = '/' + self.robot_name + '/base/send_gripper_command'
        rospy.wait_for_service(send_gripper_command_full_name)
        self.send_gripper_command = rospy.ServiceProxy(send_gripper_command_full_name, SendGripperCommand)

        activate_publishing_of_action_notification_full_name = '/' + self.robot_name + '/base/activate_publishing_of_action_topic'
        rospy.wait_for_service(activate_publishing_of_action_notification_full_name)
        self.activate_publishing_of_action_notification = rospy.ServiceProxy(activate_publishing_of_action_notification_full_name, OnNotificationActionTopic)
    
        get_product_configuration_full_name = '/' + self.robot_name + '/base/get_product_configuration'
        rospy.wait_for_service(get_product_configuration_full_name)
        self.get_product_configuration = rospy.ServiceProxy(get_product_configuration_full_name, GetProductConfiguration)

        validate_waypoint_list_full_name = '/' + self.robot_name + '/base/validate_waypoint_list'
        rospy.wait_for_service(validate_waypoint_list_full_name)
        self.validate_waypoint_list = rospy.ServiceProxy(validate_waypoint_list_full_name, ValidateWaypointList)
        
        
        
        # is this necessary?
        # clear faults
        self._clear_faults()
        # Activate the action notifications
        self._notif_subscription()
    def in_bounds(self,angles):
        # todo later, push all angles in bounds
        return angles
    def step(self,action):
        old_pos=np.degrees(self.get_joint_state()[0][1:]) #FINGER is first one
        arm_diff=action[:6]*self.diffs
        arm_angles=old_pos+arm_diff
        arm_angles=self.in_bounds(arm_angles)
        
        self.move_arm(arm_angles)
        self.move_fingy((action[6]+1)/2) #fingy will always be between 0,1
        
        
        REWARD=-1
        DONE=False
        INFO={}
        obs=self.get_obs()
        print("IMPLEMENT THESE IN SUBCLASS")
        return obs,REWARD,DONE,INFO
    
    def reset(self):
        self.move_fingy(0)
        self.move_arm(self.init_pos)
        obs=self.get_obs()
        print("NEEDS TO RESET CUPS or whatever IN LOWER METHOD")
        return obs
        
    def get_obs(self):
        print("SPECIFY GET OBS IN SUBCLASS")
        pos,vel,eff= self.get_joint_state()
        return pos+vel+eff
        
        # should prob use self.get_joint_state as well as other stuff
        
    def get_obs_dim(self):
        print("SPECIFY  obs_dim IN SUBCLASS")
        return 21
    
    def cb_action_topic(self, notif):
        self.last_action_notif_type = notif.action_event
    

    def wait_for_action_end_or_abort(self):
        while not rospy.is_shutdown():
            if (self.last_action_notif_type == ActionEvent.ACTION_END):
                rospy.loginfo("Received ACTION_END notification")
                return True
            elif (self.last_action_notif_type == ActionEvent.ACTION_ABORT):
                rospy.loginfo("Received ACTION_ABORT notification")
                return False
            else:
                time.sleep(0.01)

    def _notif_subscription(self):
        # Activate the publishing of the ActionNotification
        req = OnNotificationActionTopicRequest()
        rospy.loginfo("Activating the action notifications...")
        try:
            self.activate_publishing_of_action_notification(req)
        except rospy.ServiceException:
            rospy.logerr("Failed to call OnNotificationActionTopic")
            return False
        else:
            rospy.loginfo("Successfully activated the Action Notifications!")

        #rospy.sleep(1.0)
        rospy.sleep(0.01)
        return True

    def _clear_faults(self):
        try:
            self.clear_faults()
        except rospy.ServiceException:
            rospy.logerr("Failed to call ClearFaults")
            return False
        else:
            rospy.loginfo("Cleared the faults successfully")
            #rospy.sleep(2.5)
            rospy.sleep(0.01)
            return True
            
    def get_image_numpy(self):
        return ros_numpy.numpify(self.camera_img)
    
    def get_image_PIL(self):
        img_numpy=self.get_image_numpy()
        return IMG.fromarray(img_numpy, "RGB")
    def save_image(self,filee):
        img=self.get_image_PIL()
        img.save(filee)
    def get_joint_state(self):
        #returns tuple with pos, velocity, effort
        #THIS IS IN RADIANS
        # NOTE: the order is finger, then the 6 joints
        curr=self.joint_data
        self.position=curr.position
        self.velocity=curr.velocity
        self.effort=curr.effort
        return curr.position,curr.velocity,curr.effort
    
    def move_arm(self,angles):
        # moves robot arm to the angles, requires a list of 6 (list of #dof)
        self.last_action_notif_type = None
        self.desired_angles=angles
        req = ReadActionRequest()
        req.input.identifier = self.HOME_ACTION_IDENTIFIER
        try:
            res = self.read_action(req)
        except rospy.ServiceException:
            rospy.logerr("Failed to call ReadAction")
            return False
        else:
            # What we just read is the input of the ExecuteAction service
            req = ExecuteActionRequest()
            req.input = res.output
            rospy.loginfo("action:"+str(angles))
            try:
                i=0
                for obj in (req.input.oneof_action_parameters.reach_joint_angles[0].joint_angles.joint_angles):
                    obj.value=angles[i]# now robot thinks "angles" is the home position 
                    # yes this is janky
                    i+=1
                self.execute_action(req)
            except rospy.ServiceException:
                rospy.logerr("Failed to call ExecuteAction")
                return False
            else:
                return self.wait_for_action_end_or_abort() #True

    def move_fingy(self, value):
        # Initialize the request
        # Close the gripper
        req = SendGripperCommandRequest()
        finger = Finger()
        finger.finger_identifier = 0
        finger.value = value
        req.input.gripper.finger.append(finger)
        req.input.mode = GripperMode.GRIPPER_POSITION

        rospy.loginfo("Sending the gripper command...")

        # Call the service 
        try:
            self.send_gripper_command(req)
        except rospy.ServiceException:
            rospy.logerr("Failed to call SendGripperCommand")
            return False
        else:
            #time.sleep(.5)
            rospy.sleep(0.5)
            return True
    def render(self, mode='human', close=False):
        pass
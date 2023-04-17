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
from gazebo_msgs.msg import LinkStates, ModelState, ModelStates
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
                    ):
        
        # Ranges for randomizing cups and determining goal
        self.table_y_range=(-0.29,0.29)
        self.cup_ranges=((-1.4,-0.31),self.table_y_range)
        self.cup_goal_x = -0.3 # or below

        self.action_dim=7
        self.obs_dim=self.get_obs_dim()
        self.init_pos=init_pos
        self.diffs=differences
        high = np.ones([self.action_dim])
        self.action_space = gym.spaces.Box(-high, high)
        high = np.inf * np.ones([self.obs_dim])
        self.observation_space = gym.spaces.Box(-high, high)
        joint_namespace='/'+ROBOT_NAME+'/joint_states'
        
        # Subscribe to joint data
        def joint_callback(data):
            self.joint_data=data
        self.joint_sub=rospy.Subscriber(joint_namespace,JointState,joint_callback)

        # Subscribe to object data
        self.object_data={}
        def _call_model_data(data):
            self.object_data={}
            for i in range(len(data.name)):
                self.object_data[data.name[i]]=data.pose[i]
        self.sub_topic="/gazebo/model_states"
        self.sub=rospy.Subscriber(self.sub_topic,ModelStates,_call_model_data)
        
        # Subscribe to camera image
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
    
    #============================ MAIN FUNCTIONS ==============================#

    def step(self,action):
        old_pos=np.degrees(self.get_joint_state()[0][1:]) #FINGER is first one
        arm_diff=action[:6]*self.diffs
        arm_angles=old_pos+arm_diff
        self.move_arm(arm_angles)
        self.move_fingy((action[6]+1)/2) #fingy will always be between 0,1
        
        REWARD=self.get_reward()
        DONE=False
        INFO={}
        obs=self.get_obs()
        print("IMPLEMENT THESE IN SUBCLASS")
        return obs,REWARD,DONE,INFO
    
    def reset(self):
        self.move_fingy(0)
        self.move_arm(self.init_pos)
        obs=self.get_obs()
        self.reset_cups()
        return obs
    
    def render(self, mode='human', close=False):
        pass

    #======================== OBSERVATION, REWARD =============================#
        
    def get_obs(self):
        print("SPECIFY GET OBS IN SUBCLASS")
        pos,vel,eff= self.get_joint_state()
        print("may need to fix this function")
        return np.array(pos+vel+eff+self.obj_data)
        
        # should prob use self.get_joint_state as well as other stuff
        
    def get_obs_dim(self):
        print("SPECIFY obs_dim IN SUBCLASS")
        return 31
    
    def get_reward(self):
        self.tip_coord = self.get_tip_coord() # This is not going to work yet
        self.reward = 100
        closest_dist = 100
        obj_data = self.get_object_data()
        cups = ["cup1","cup2","cup3"]
        for cup in cups:
            pos = obj_data[cup].position
            # print("\n--------------------")
            # self.robot.cup_in_hand(pos)
            # print("--------------------\n")
            # Negative reward for each cup that is off the table
            if (not self.cup_on_table(pos)):
                print(cup, " is off the table")
                self.reward -= 50
            else:  # Large positive reward for each cup in the goal zone
                if(self.cup_at_goal_loc(pos)):
                    print(cup, " is at the goal")
                    self.reward += 100
                else: # Reward incentivising cups to be close to goal
                    dist_to_goal = self.cup_goal_x - pos.x
                    self.reward -= dist_to_goal * 10
                    dist_to_cup = np.linalg.norm(self.tip_coord - np.array([pos.x,pos.y,pos.z]))
                    if(dist_to_cup < closest_dist):
                        closest_dist = dist_to_cup
        # Reward incentivising robot tip to be close to the nearest cup not 
        # already in the goal zone, as long as there are sitll cups not at goal
        if(closest_dist != 100):
            print(cup, " is dist ", closest_dist)
            self.reward -= closest_dist * 10
        print("Reward is ",self.reward)
    
    #========================= SAVING CAMERA IMAGE ============================#

    def get_image_numpy(self):
        return ros_numpy.numpify(self.camera_img)
    
    def get_image_PIL(self):
        img_numpy=self.get_image_numpy()
        return IMG.fromarray(img_numpy, "RGB")

    def save_image(self,filee):
        img=self.get_image_PIL()
        img.save(filee)
    
    #========================= CUP HELPER FUNCTIONS ===========================#

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
    
    def reset_cups(self):
        # generate random new cup positions
        cup_names = ["cup1", "cup2", "cup3"]
        cup_positions = []
        for i in range(len(cup_names)):
            x = random.uniform(self.cup_ranges[0][0],self.cup_ranges[0][1])
            y = random.uniform(self.cup_ranges[1][0],self.cup_ranges[1][1])
            while(self.cup_has_collision(x,y)):
                x = random.uniform(self.cup_ranges[0][0],self.cup_ranges[0][1])
                y = random.uniform(self.cup_ranges[1][0],self.cup_ranges[1][1])
            cup_positions.append((x,y,.065))
        self.move_cups(cup_positions)
    
    def is_upside_down(self,orientation,tol=.02):
            # orientation is a Quaternion object (example: orientation = self.doota['cup1'].orientation)
            # will convert to roll, pitch, yaw (rotation on x,y,z axis), ignore z axis and see if cup is exactly upside down
            (roll,pitch,yaw)=Rotation.from_quat((orientation.x,orientation.y,orientation.z,orientation.w)).as_euler('xyz')
            roll_inversion=bool(abs(np.pi-abs(roll))<=tol)
            pitch_inversion=bool(abs(np.pi-abs(pitch))<=tol)
            return roll_inversion^pitch_inversion #returns if exactly one of these are true (i.e if cup is flipped once)
        
    def cup_on_table(self,pos):
        return pos.z >= 0
    
    def cup_at_goal_loc(self,pos):
        return self.cup_on_table(pos) & (pos.x >= self.cup_goal_x)
    
    def cup_in_hand(self,pos):
        self.read_state()
        print(self.eff)
    
    #=========================== ROBOT MOVEMENT ===============================#

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
        goal=.8*value # position .8 for the joint position corresponds to a pinched finger
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
            moved=self.wait_fingy_done(goal)
            print('fingy moved:',moved)
            return True

    def wait_fingy_done(self,
                          goal, #goal finger position
                          ptol=.005, #closeness of the position
                          sleepy=.01, #sleep command for ros
                          vtol=lambda time:time # given time, acceptable velocity that makes a 'done' action, not measured at time 0
                          ):
        if abs(self.get_joint_state()[0][0]-goal)<=ptol:
            # already there
            return False
        rospy.sleep(sleepy)
        time=sleepy
        pos,vel,eff=self.get_joint_state()
        while abs(pos[0]-goal)>ptol and abs(vel[0])>vtol(time): # if either position is correct or finger stopped moving, we are done
            time+=sleepy
            rospy.sleep(sleepy)
            pos,vel,eff=self.get_joint_state()
        return True
        # returns if finger actually moved
    
    #===================== OTHER HELPER FUNCTIONS =============================#

    def get_tip_coord(self):
        print("IMPLEMENT THIS")
        return self.get_joint_state()[0][1:] 
    
    def get_joint_state(self):
        #returns tuple with pos, velocity, effort
        #THIS IS IN RADIANS
        # NOTE: the order is finger, then the 6 joints
        curr=self.joint_data
        self.position=curr.position
        self.velocity=curr.velocity
        self.effort=curr.effort
        return curr.position,curr.velocity,curr.effort
    
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
    
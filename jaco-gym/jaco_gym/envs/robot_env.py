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
import math
from PIL import Image as IMG
from kortex_driver.srv import Base_ClearFaults, ReadAction, ExecuteAction, SetCartesianReferenceFrame,SendGripperCommand,OnNotificationActionTopic
from kortex_driver.srv import GetProductConfiguration, ValidateWaypointList,OnNotificationActionTopicRequest,ReadActionRequest,ExecuteActionRequest,SendGripperCommandRequest
from kortex_driver.msg import ActionNotification, ActionEvent,Finger,GripperMode
from sensor_msgs.msg import JointState, Image
import ros_numpy
import cv2

class JacoEnv(gym.Env):
    def __init__(self,
                    ROBOT_NAME='my_gen3',
                    ATTACHED_CAM_SPACE='attached_camera', #call will look for /ATTACHED_CAM_SPACE/color/image_raw and /ATTACHED_CAM_SPACE/depth/image_raw
                    HEAD_CAM_SPACE='head_camera', #call will look for /HEAD_CAM_SPACE/color/image_raw and /HEAD_CAM_SPACE/depth/image_raw
                    init_pos=(0,15,230,0,55,90), #HOME position
                    differences=(15,15,15,15,15,15), # maximum angular movement allowed at each joint per action
                    image_dim=(128,128,3), # image vector, will resize input images to this
                    depth_max_head=10, # what to return when depth out of range
                    depth_max_att=10,# what to return when depth out of range
                    ):
    
        self.image_dim=image_dim
        
        self.action_dim=7
        self.obs_dim=self.get_obs_dim()
        self.init_pos=init_pos
        self.diffs=differences
        
        self.depth_max_head=depth_max_head
        self.depth_max_att=depth_max_att
        
        # Dimensions of each of the joints obtained from the Kinova Gen3 user
        # guide for 6 dof https://www.kinovarobotics.com/uploads/User-Guide-Gen3-R07.pdf
        # used for calculating cartesian positions of each joint
        self.LENGTHS=(.1564, # base to rotation joint
                      .1284, # rotation joint to shoulder
                      .410, # shoulder to elbow 
                      .2085, # elbow to rotation joint
                      .1059, # rotation joint to wrist tilt joint
                      .1059, # wrist tilt to wrist rotation
                      .0615, # wrist rotation joint to base of gripper
                      .088, # base of gripper to 'palm' of hand
                      .0613, # 'palm' of hand to open fingertip
                      .0135, # open fingertip length to closed fingertip length
                      )
        high = np.ones([self.action_dim])
        self.action_space = gym.spaces.Box(-high, high)
        high = np.inf * np.ones([self.obs_dim])
        self.observation_space = gym.spaces.Box(-high, high)
        joint_namespace='/'+ROBOT_NAME+'/joint_states'
        
        # Subscribe to joint data 
        def joint_callback(data):
            self.joint_data=data
        self.joint_sub=rospy.Subscriber(joint_namespace,JointState,joint_callback)
        
        # Subscribe to camera data
        self.attached_camera_img,self.head_camera_img,self.attached_camera_depth_img,self.head_camera_depth_img=None,None,None,None
        
        def _attached_camera_img_callback(data):
            self.attached_camera_img = data
        self.attached_image_sub=rospy.Subscriber("/"+ATTACHED_CAM_SPACE+"/color/image_raw",Image,_attached_camera_img_callback)
        
        def _head_camera_img_callback(data):
            self.head_camera_img = data
        self.head_image_sub=rospy.Subscriber("/"+HEAD_CAM_SPACE+"/color/image_raw",Image,_head_camera_img_callback)
        
        # depth as well        
        def _attached_camera_depth_img_callback(data):
            self.attached_camera_depth_img = data
        self.attached_depth_image_sub=rospy.Subscriber("/"+ATTACHED_CAM_SPACE+"/depth/image_raw",Image,_attached_camera_depth_img_callback)
        
        def _head_camera_depth_img_callback(data):
            self.head_camera_depth_img = data
        self.head_depth_image_sub=rospy.Subscriber("/"+HEAD_CAM_SPACE+"/depth/image_raw",Image,_head_camera_depth_img_callback)
        
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
        
        self._clear_faults()
        self._notif_subscription() # Activate the action notifications

    #========================== REQUIRED FUNCTIONS ============================#

    def step(self,action):
        old_pos=np.degrees(self.get_joint_state()[0][:6]) #First 6 elements are the joints
        arm_diff=action[:6]*self.diffs
        arm_angles=old_pos+arm_diff
        self.move_arm(arm_angles)
        self.move_fingy((action[6]+1)/2) #finger will always be between 0,1
        REWARD=-1 # IMPLEMENT THESE METHODS IN SUBCLASS
        DONE=False
        INFO={}
        obs=self.get_obs()
        return obs,REWARD,DONE,INFO
    
    def reset(self): # OVERWRITE THIS METHOD IN SUBCLASS
        self.move_fingy(0)
        self.move_arm(self.init_pos)
        obs=np.array([])
        return obs
        
    #========================== OBSERVATION FUNCTIONS ============================#
    
    def get_obs(self): # OVERWRITE THIS METHOD IN SUBCLASS
        pos,vel,eff= self.get_joint_state()
        return np.concatenate((pos%(2*np.pi),vel,eff)) #Mod position by 2pi since it is an angle
    
    def get_full_obs(self): # appends camera vector to observation, prob no need to mess with this in subclass, since it calls get_obs
        return np.concatenate((self.get_image_obs_vector(),self.get_obs()))
    
    def get_obs_dim(self): # OVERWRITE THIS METHOD IN SUBCLASS
        return 0
    
    
    #========================== GETTING ROBOT INFO ============================#

    # Returns information about robot state that can be attained in both
    # the simulation and real environment, including the joint positions (radians),
    # joint velocity, joint effort, and also the (x,y,z) cartesian coordinates
    # of each joint calculated using trig

    def get_joint_state(self):
        # Returns a tuple with pos (in radians), velocity, effort for each joint
        # NOTE: For simulation, the return order is just the 6 joints follwed by 
        # the finger, but for the physical robot the namesapce is
        # ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6', 'finger_joint',
        #  'left_inner_knuckle_joint', 'left_inner_finger_joint', 'right_outer_knuckle_joint', 
        #  'right_inner_knuckle_joint', 'right_inner_finger_joint']
        curr=self.joint_data
        fields=( 'joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6','finger_joint')
        indices=np.array([curr.name.index(f) for f in fields])
        self.position=np.array(curr.position)[indices]
        self.velocity=np.array(curr.velocity)[indices]
        self.effort=np.array(curr.effort)[indices]
        return self.position,self.velocity,self.effort
    
    def get_cartesian_points(self):
        # returns points of each joint, calculated with trig
        joint_angles,_,_=self.get_joint_state()
        pos=np.array([0.,0.,0.]) # keeps track of position
        bottom=pos.copy() # included for completion
        basis=np.identity(3) # keeps track of rotation, column vectors are the basis
        pos+=self.LENGTHS[0]*basis[:,2] # adding the z basis, which should be straight up
        base_rot_joint=pos.copy()
        basis=self.rotate_about(basis,2,-joint_angles[0]) # rotation is defined counterclockwise, the robot has ccw negative on the base joint
        pos+=self.LENGTHS[1]*basis[:,2] # adding z basis (straight up) again
        shoulder_joint=pos.copy()
        basis=self.rotate_about(basis,1,joint_angles[1]) # correct rotation along the y basis
        pos+=self.LENGTHS[2]*basis[:,2] # along z again
        elbow_joint=pos.copy()
        basis=self.rotate_about(basis,1,-joint_angles[2]) # rotation about y, but direction is opposite
        pos+=self.LENGTHS[3]*basis[:,2] # along z
        rot_joint=pos.copy()
        basis=self.rotate_about(basis,2,-joint_angles[3]) # about z, opposite again
        pos+=self.LENGTHS[4]*basis[:,2]
        wrist_flip_joint=pos.copy()
        basis=self.rotate_about(basis,1,-joint_angles[4]) #about y, negative 
        pos+=self.LENGTHS[5]*basis[:,2]
        wrist_joint=pos.copy()
        basis=self.rotate_about(basis,2,-joint_angles[5]) # about z, negative again 
        #NOTE: camera is positioned on the -y direction of the final joint
        # camera_pos=pos+ camera_dist * basis[:,2] + camera_height * (-basis[:,1])
        pos+=self.LENGTHS[6]*basis[:,2]
        gripper_base=pos.copy()
        gripper_positions=[]
        for leng in self.LENGTHS[7:]:
            pos+=leng*basis[:,2]
            gripper_positions.append(pos.copy())
        return bottom, base_rot_joint,shoulder_joint,elbow_joint,rot_joint,wrist_flip_joint,wrist_joint,gripper_base,gripper_positions
        # Gripper positions has palm of hand, and other data

    def get_tip_coord(self):
        return self.get_cartesian_points()[-1][-1]
    #========================= SAVING CAMERA IMAGES ===========================#

    # Saves the camera image as a numpy array or to a png to be viewed,
    # works on the simulated camera and real camera attached to the Kinova Arm
            
    def get_image_numpy(self,cam_type="head",mode='color'):
        if mode=='color':
            if cam_type=='head':
                return ros_numpy.numpify(self.head_camera_img)
            else:
                return ros_numpy.numpify(self.attached_camera_img)
        else:
            if cam_type=='head':
                mat=ros_numpy.numpify(self.head_camera_depth_img)
                mat[np.where(np.isnan(mat))]=self.depth_max_head
            else:
                mat=ros_numpy.numpify(self.attached_camera_depth_img)
                mat[np.where(np.isnan(mat))]=self.depth_max_att
            return mat
        #return ros_numpy.numpify(self.head_camera_img if cam_type=="head" else self.attached_camera_img)
    
    def get_image_obs_array(self,cam_type="head",mode='color'):
        nump=self.get_image_numpy(cam_type=cam_type,mode=mode).astype(np.uint8)
        resized=cv2.resize(nump, self.image_dim[:2], interpolation = cv2.INTER_AREA)
        resized=resized.transpose((2,1,0))# this changes order to [channels, height, width], which is used in CNNs
        return resized.astype(np.float64)
        
    def get_image_obs_vector(self,cam_type="head",mode='color'):
        return self.get_image_obs_array(cam_type=cam_type,mode=mode).flatten()
        
    def recover_image_from_obs_vector(self,vector,cam_type="head",mode='color'): # recovers image array (width, height, channels) from obs vector or image vector
        # assumes first part of vector is flattened image in [channels, height, width] order
        thingy=vector[:self.get_image_obs_vector_dim(cam_type=cam_type,mode=mode)].reshape(self.image_dim[::-1]) # in reverse, i.e. reshape(3,256,256)
        thingy=thingy.transpose((2,1,0)) # flips to (width, height, channels)
        return thingy
        
    def recover_images_from_obs_vectors(self,vectors,cam_type="head",mode='color'): 
        # recovers image arrays (M x width, height, channels) from obs vector or image vector (M X K)
        M=len(vectors)
        thingy=vector[:,:self.get_image_obs_vector_dim(cam_type=cam_type,mode=mode)].reshape((M,)+self.image_dim[::-1]) # in reverse, i.e. reshape(M,3,256,256)
        thingy=thingy.transpose((0,3,2,1))# flips to (M, width, height, channels)
        return thingy
    
    def get_image_obs_vector_dim(self,cam_type="head",mode='color'): 
        return np.prod(self.image_dim)
    
    def get_image_PIL(self,cam_type="head",mode='color'):
        img_numpy=self.get_image_numpy(cam_type=cam_type,mode=mode)
        return IMG.fromarray(img_numpy, "RGB")

    def save_image(self,filee,cam_type="head",mode='color'):
        img=self.get_image_PIL(cam_type=cam_type,mode=mode)
        img.save(filee)
    
    #============================= INTERSECTION ===============================#

    # Tests to see if a robot "intersects" itself or if its joints get too close
    # to eachother, doesn't use any ROS methods (just pure trig and vector
    # operations), so this can be called both in the simulation and real robot
    # simply by having access to the robot's joint angles
    
    def segment_dist_min(self,a0,a1,b0,b1,tol=.001):
        ''' Given two lines defined by numpy.array pairs (a0,a1,b0,b1)
            Return the closest points on each segment and their distance
        '''    
        # vectors, cross product
        A = a1 - a0
        B = b1 - b0
        magA = np.linalg.norm(A)
        magB = np.linalg.norm(B)
        
        _A = A / magA
        _B = B / magB
        
        cross = np.cross(_A, _B);
        denom = np.linalg.norm(cross)**2
        
        
        # If lines are parallel (denom=0) test if lines overlap.
        # If they don't overlap then there is a closest point solution.
        # If they do overlap, there are infinite closest positions, but there is a closest distance
        if denom<=tol:
            #crossing vectors
            w00=b0-a0
            w01=b1-a0
            w10=b0-a1
            w11=b1-a1
            
            #finding projection of a0->b0 onto line a:
            proj=np.dot(_A,w00)
            proj_v=_A*proj
            
            # proj_v+error_v = a0->b0 where proj_v is along line a and error v is perp to it
            error_v=w00-proj_v 
            
            if np.dot(w00,_A)*np.dot(w01,_A)<=0: # point a0 lies in between projections of point b0 and point b1 (i.e. w00 is 'backwards' and w01 is 'forwards', or inverse)
                return np.linalg.norm(error_v),(a0,a0+error_v) # note a0+error_v should be on line b
                
            elif np.dot(w10,_A)*np.dot(w11,_A)<=0: # point a1 is in between proj of points b0 and b1
                return np.linalg.norm(error_v),(a1,a1+error_v)
            
            elif np.dot(w00,_B)*np.dot(w10,_B)<=0: # point b0 in betweeen proj of pts a0 and a1
                return np.linalg.norm(error_v),(b0-error_v,b0) # minus since error_v is from line a to line b
                
            elif np.dot(w01,_B)*np.dot(w11,_B)<=0: # point b1 in betweeen proj of pts a0 and a1
                return np.linalg.norm(error_v),(b1-error_v,b1) 
            # we must now find min of the endpoints, since the centers do not overlap
            opt=(None,None)
            for ai in (a0,a1):
                for bi in (b0,b1):
                    dist=np.linalg.norm(ai-bi)
                    if opt[0] is None or dist<opt[0]:
                        opt=dist,(ai,bi)
            return opt           
        
        # Lines criss-cross: Calculate the projected closest points
        t = (b0 - a0);
        detA = np.linalg.det([t, _B, cross])
        detB = np.linalg.det([t, _A, cross])
    
        t0 = detA/denom;
        t1 = detB/denom;
    
        pA = a0 + (_A * t0) # Projected closest point on segment A
        pB = b0 + (_B * t1) # Projected closest point on segment B
    
        # Clamp projections
        if t0 < 0:
            pA = a0
        elif t0 > magA:
            pA = a1
        
        if t1 < 0:
            pB = b0
        elif t1 > magB:
            pB = b1
            
        # Clamp projection A
        if (t0 < 0) or (t0 > magA):
            dot = np.dot(_B,(pA-b0))
            if dot < 0:
                dot = 0
            elif dot > magB:
                dot = magB
            pB = b0 + (_B * dot)
    
        # Clamp projection B
        if (t1 < 0) or (t1 > magB):
            dot = np.dot(_A,(pB-a0))
            if dot < 0:
                dot = 0
            elif dot > magA:
                dot = magA
            pA = a0 + (_A * dot)
    
        return np.linalg.norm(pA-pB),(pA,pB)
    
    def robot_intersects_self(self,tol = 0.1,
                        check=( # which segments to check 
                                ((0,2),(3,5)),  # i.e. this means check segment on joints 0->2 and segment on joints 3->4 (base to shoulder segment and elbow to wrist segment)
                                ((0,2),(5,'f')), # 'f' represents finger, might be different based on finger grippy position
                                ((2,3),(5,'f')) # ALWAYS put 'f' at the end, this makes the method less annoying
                                )):
        jointy_pointy=self.get_cartesian_points()
        finger=jointy_pointy[-1]
        for (i0,i1),(j0,j1) in check: # checks line segments between joint points at these indices  
            a0,a1= jointy_pointy[i0], jointy_pointy[i1]
            b0=jointy_pointy[j0]
            if j1=='f':
                b1=finger[-1] # safe for now, just using furthest tip point
            else:
                b1=jointy_pointy[j1]
            dist,points=self.segment_dist_min(a0,a1,b0,b1)
            if dist<=tol: 
                return True
        return False

    #================================ MOVEMENT ================================#

    # Functions to execute actual movement of robot (don't mess with these),
    # these functions will be called automatically in the step function when
    # a given action is passed in, so to make the robot move just call 
    # env.step(action) instead of using these
    def wrist_tilt_bounds(self,gamma=np.radians(35)):
        # finds min and max possible distances that the wrist tilt joint is from the "shoulder" joint
        # gamma is lowest possible elbow angle
        a=self.LENGTHS[2]
        b=self.LENGTHS[3]+self.LENGTHS[4]
        return (np.sqrt(a**2+b**2-2*a*b*np.cos(gamma)),a+b)
        
    
    def find_chord(self,R,c,d):
        # given a triangle with base R, sides c and d, finds intersection angle of the d side
        a=(d**2-c**2+R**2)/(2*R)
        return np.arccos(a/d)
        
    def look_at_point(self,x,y,sight_dist=.3): 
        # returns x,y,h,deg to look at point on table
        # sight_dist is how far away to look at point
        temp=np.array((x,y))
        r=np.linalg.norm(temp)
        new_r=max(r-.04,.04)
        x,y=temp*new_r/r
        
        
        vec=np.array((x,y,-self.LENGTHS[0]-self.LENGTHS[1])) # vector to point
        R=np.linalg.norm(vec)
        r=np.linalg.norm(vec[:2])
        theta=np.arctan2(vec[2],r) # angle from arm shoulder to point
        
        
        d=sight_dist+self.LENGTHS[5]+self.LENGTHS[6] # this is how far the wrist tilt joint should be
        c_low,c_high = self.wrist_tilt_bounds()
        
        #print('wrist bounds:',c_low,c_high)
        #print('d:',d)
        #print('R:',R)
        
        if R>=c_high+d:
            #print('returning since',R,">=",c_high,'+',d)
            xp,yp,dh=vec*c_high/R
            h=self.LENGTHS[0]+self.LENGTHS[1]+dh
            return xp,yp,h,theta
        if R<=c_low:
            #print('returning since',R,"<=",c_low)
            return x,y,d,-90
        
        bounds=[np.radians(10),np.radians(90)] # bounds of angle from the point
        dead_zone=[-theta,-theta] # cannot choose here
        if c_high<R: 
            ang=self.find_chord(R,c_high,d)
            bounds[1]=min((-theta)+ang,bounds[1])
            bounds[0]=max(bounds[0],(-theta)-ang)
            #print('bounds:',np.degrees(bounds))
        if R<=c_low+d:
            ang=self.find_chord(R,c_low,d)
            dead_zone[0]=max((-theta)-ang,bounds[0])
            dead_zone[1]=min((-theta)+ang,bounds[1])
            #print('dead zone:',np.degrees(dead_zone))
        
        
        dead_area=dead_zone[1]-dead_zone[0]
        rand=np.random.uniform(bounds[0],bounds[1]-dead_area)
        if rand<=dead_zone[0]:
            angle=rand
        else:
            angle=rand+dead_area
            
        h=np.sin(angle)*d
        eaten=np.cos(angle)*d
        xp,yp=vec[:2]*(r-eaten)/r
        
        #print(r-eaten,h,-np.degrees(angle))
        
        
        return xp,yp,h,-np.degrees(angle)
        
    def cartesian_pick(self,x,y,h,sight_ang=0):
        r=np.linalg.norm([x,y])
        psi=np.arctan2(y,x)
        h_0=self.LENGTHS[0]+self.LENGTHS[1]
        h-=h_0
        
        a=self.LENGTHS[2]
        b=self.LENGTHS[3]+self.LENGTHS[4]        
        c_low,c_high=self.wrist_tilt_bounds()
        
        c=np.sqrt(h**2+r**2)
        if c>c_high: 
            print('reach out of bounds, going close')
            h=h*(c_high)/c
            r=r*(c_high)/c
            c=c_high
        if c==0:
            print('error, tried going to (x,y,h-h0)=(0,0,0)')
            h=.01
            r=.01
            c=np.sqrt(h**2+r**2)
        if c<c_low:
            print('reach out of bounds, going close')
            h=h*c_low/c
            r=r*c_low/c
            c=c_low
        theta=np.arccos((a**2+b**2-c**2)/(2*a*b))
        
        y=np.arctan2(h,r)
        x=np.arccos((a**2+c**2-b**2)/(2*a*c))
        phi=np.pi/2-x-y
        
        
        self.move_arm((-np.degrees(psi),np.degrees(phi),180+np.degrees(theta),0,np.degrees(phi-theta)+90+sight_ang,90))
    
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
    
    def rotate_about(self,basis,col,angle):
        #rotates a basis around the 'col'th vector by 'angle'
        # always goes counterclockwise, i.e. 
        #  rotation about x sends y to z, 
        #  rotation about y sends z to x, 
        #  rotation about z sends x to y, 
        new_basis=basis.copy()
        cols=[0,1,2]
        cols.remove(col) # these are the ones that change
        if col==1:
            cols.reverse() # now rotation is in direction col[0] -> col[1]
            
        new_basis[:,cols[0]]=np.cos(angle)*basis[:,cols[0]] + np.sin(angle) * basis[:,cols[1]] # cols[0] is rotated towards cols[1]
        new_basis[:,cols[1]]=np.cos(angle)*basis[:,cols[1]] - np.sin(angle) * basis[:,cols[0]] # cols[1] is rotated away from cols[0]
        return new_basis

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
            #print('fingy moved:',moved)
            return True

    def wait_fingy_done(self,
                          goal, #goal finger position
                          ptol=.005, #closeness of the position
                          sleepy=.01, #sleep command for ros
                          vtol=lambda time:time # given time, acceptable velocity that makes a 'done' action, not measured at time 0
                          ):
        if abs(self.get_joint_state()[0][6]-goal)<=ptol:
            # already there
            return False
        rospy.sleep(sleepy)
        time=sleepy
        pos,vel,eff=self.get_joint_state()
        while abs(pos[6]-goal)>ptol and abs(vel[6])>vtol(time): # if either position is correct or finger stopped moving, we are done
            time+=sleepy
            rospy.sleep(sleepy)
            pos,vel,eff=self.get_joint_state()
        return True # returns if finger actually moved
    
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
        
    def render(self, mode='human', close=False):
        pass

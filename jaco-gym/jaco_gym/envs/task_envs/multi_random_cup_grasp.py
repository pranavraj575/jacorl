from gazebo_msgs.msg import LinkStates, ModelState, ModelStates
from geometry_msgs.msg import Pose, Point, Quaternion
from gazebo_msgs.srv import DeleteModel, SpawnModel
from jaco_gym.envs.gazebo_env import JacoGazeboEnv
import numpy as np
import rospy
import math
from scipy.spatial.transform import Rotation
import os

class JacoMultiGrabCupGazebo(JacoGazeboEnv):
    def __init__(self,
                    ROBOT_NAME='my_gen3',
                    ATTACHED_CAM_SPACE='attached_camera', #call will look for /ATTACHED_CAM_SPACE/color/image_raw
                    HEAD_CAM_SPACE='head_camera', #call will look for /HEAD_CAM_SPACE/color/image_raw
                    init_pos=(0,15,230,0,55,90), #HOME position
                    differences=(15,15,15,15,15,15), # angular movement allowed at each joint per action
                    image_dim=(128,128,3), # image vector, will resize input images to this
                    ):
                    
        self.number=2
        super().__init__(ROBOT_NAME=ROBOT_NAME,
                        ATTACHED_CAM_SPACE=ATTACHED_CAM_SPACE,
                        HEAD_CAM_SPACE=HEAD_CAM_SPACE,
                        init_pos=init_pos,
                        differences=differences,
                        image_dim=image_dim)

        # task specific stuff
        
        # Ranges for randomizing cups and determining goal
        self.table_y_range=(-0.49,0.09)
        self.cup_ranges=((0.3,0.7),self.table_y_range)
        self.cup_goal_x = 0.3 # or above
        self.max_cup_x = self.cup_ranges[0][1]
        self.model_file_names=["solo_cup",
                                "solo_cup_blue",
                                "solo_cup_blue_large",
                                "solo_cup_blue_small",
                                "solo_cup_green",
                                "solo_cup_green_large",
                                "solo_cup_green_small",
                                "solo_cup_large",
                                "solo_cup_small",
                                "solo_cup_yellow",
                                "solo_cup_yellow_large",
                                "solo_cup_yellow_small",
                              ]
        self.ordered_names=[]
        

    def step(self,action):
        joint_obs,_,_,_=super().step(action)
        REWARD,DONE=self.get_reward_done()
        INFO={}
        obs=self.get_obs()
        #print("good")
        return obs,REWARD,DONE,INFO
    
    def main_cup_name(self):
        return 'cup0'
    
    def reset(self):
        self.ordered_names=[]
        joint_obs=super().reset()
        self.reset_cup()
        #print('RESETTING CUPS')
        cup_pos=self.get_pose_eulerian(self.main_cup_name())[:3]
        x,y,h,gamma=self.look_at_point(cup_pos[0],cup_pos[1])
        self.cartesian_pick(x,y,h,gamma)
        obs=self.get_obs()
        return obs
    
    #========================= OBSERVATION, REWARD ============================#
        
    def get_obs(self):
        # Observation is a concatination of our joint positions, joint velocities,
        # joint efforts, and the x,y,z coordinates of each of our 3 cups
        
        #print("good")
        pos,vel,eff= self.get_joint_state()
        pos=pos%(2*np.pi) # MOD POSITION since it is an angle
        
        return np.concatenate([pos,vel,eff] +
                                self.get_cup_positions()
                                )
    def get_cup_positions(self):
        return [self.get_pose_eulerian(nm) for (nm,_) in self.ordered_names]
    
    def get_obs_dim(self):
        #print("Here")
        return 21+self.number*6

    def robot_holding_cup_position(self,min_grab_pos=0.209, min_grab_eff=1.05e-1): 
        joint_positions,_,joint_efforts = self.get_joint_state()
        finger_pos = joint_positions[6]
        finger_eff = joint_efforts[6]
        return finger_pos >= min_grab_pos and finger_eff >= min_grab_eff
        
    def get_reward_done(self):
        debug=True
        if debug:
            print("\n--------------------")
        tip_coord = self.get_cartesian_points()[-1][-1] # should be max extension of fingers
        grabby_coord=self.get_cartesian_points()[-1][-2] # should be about where 'inside hand' is
        cup_p=self.get_pose_eulerian("targetObject")
        dist_to_cup=np.linalg.norm(cup_p[:3]-tip_coord)
        obj_dict=self.get_object_dict()
        grabbed_cup=True
        if not self.robot_holding_cup_position():
            grabbed_cup=False
        if dist_to_cup>.05:
            grabbed_cup=False
        
        
        if debug and grabbed_cup:
            print("cup grasp detected")
        
        return 0, False
    
    #========================= RESETTING ENVIRONMENT ==========================#
        
        
        
    def set_cup_ranges(self,x_range,y_range):
        self.cup_ranges=x_range,y_range
        
    def cup_has_collision(self,x,y,cup_positions,tol=.08):
        return any([np.linalg.norm((pos[0]-x,pos[1]-y)) <= tol for pos in cup_positions]) # any are within tolerance
        
    def reset_cup(self,prob_stand=1,prob_flip=0,prob_other=0): # input the probabilities that the cup is spawned normal, flipped, or fallen
        #print("RESETTing")
        # generate random new cup positions
        self.despawn_all()
        cup_positions = []
        #cup_rotations=[]
        for i in range(self.number):
            arr=np.random.random()
            yikes=False
            if arr<prob_stand:
                rot=(0.,0.,0.)
            elif arr<prob_stand+prob_flip:
                rot=(0.,np.pi,0.)
            else:
                yikes=True
                rot=tuple(np.random.random(2)*2*np.pi)+(0.,) # random numbers from 0 to 2pi for pitch and roll, prob gonna fall
            x = np.random.uniform(self.cup_ranges[0][0],self.cup_ranges[0][1])
            y = np.random.uniform(self.cup_ranges[1][0],self.cup_ranges[1][1])
            
            while self.cup_has_collision(x,y,cup_positions+[[0.,0.,0.]],tol=.08 if not yikes else .165): # 0,0,0 for the robot arm
                x = np.random.uniform(self.cup_ranges[0][0],self.cup_ranges[0][1])
                y = np.random.uniform(self.cup_ranges[1][0],self.cup_ranges[1][1])
    
            cup_positions.append((x,y,.065 if not yikes else .1))
            
            
            model_name=np.random.choice(self.model_file_names)
            
            obj_name='cup'+str(i)
            self.ordered_names.append((obj_name,model_name))
            
            self.spawn_model_from_name(model_name,obj_name,(x,y,.065 if not yikes else .1),rot)
        
    #========================= CUP INFORMATION ==========================#
        
    
    def robot_holding_cup_position(self,min_grab_pos=0.209, min_grab_eff=1.05e-1): 
        joint_positions,_,joint_efforts = self.get_joint_state()
        finger_pos = joint_positions[6]
        finger_eff = joint_efforts[6]
        return finger_pos >= min_grab_pos and finger_eff >= min_grab_eff

    

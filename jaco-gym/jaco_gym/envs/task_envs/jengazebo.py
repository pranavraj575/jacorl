from gazebo_msgs.msg import LinkStates, ModelState, ModelStates
from geometry_msgs.msg import Pose, Point, Quaternion
from gazebo_msgs.srv import DeleteModel, SpawnModel
from jaco_gym.envs.gazebo_env import JacoGazeboEnv
import numpy as np
import rospy
import math
from scipy.spatial.transform import Rotation
import os

class JacoJengaZebo(JacoGazeboEnv):
    def __init__(self,
                    ROBOT_NAME='my_gen3',
                    ATTACHED_CAM_SPACE='attached_camera', #call will look for /ATTACHED_CAM_SPACE/color/image_raw
                    HEAD_CAM_SPACE='head_camera', #call will look for /HEAD_CAM_SPACE/color/image_raw
                    init_pos=(0,15,230,0,55,90), #HOME position
                    differences=(15,15,15,15,15,15), # angular movement allowed at each joint per action
                    image_dim=(128,128,3), # image vector, will resize input images to this
                    ):
                    
        super().__init__(ROBOT_NAME=ROBOT_NAME,
                        ATTACHED_CAM_SPACE=ATTACHED_CAM_SPACE,
                        HEAD_CAM_SPACE=HEAD_CAM_SPACE,
                        init_pos=init_pos,
                        differences=differences,
                        image_dim=image_dim)

        # task specific stuff
        
        self.spawn_point=np.array((.25,-.25,0.))
        self.block_model="jenga_block"
        self.block_dim=np.array((.075,.025,.015))
        self.spacing=.005
        self.variance=.002
        

    def step(self,action):
        joint_obs,_,_,_=super().step(action)
        REWARD,DONE=self.get_reward_done()
        INFO={}
        obs=self.get_obs()
        #print("good")
        return obs,REWARD,DONE,INFO
    
    def reset(self):
        self.reset_tower()
        #print('RESETTING CUPS')
        joint_obs=super().reset()
        obs=self.get_obs()
        return obs
    
    #========================= OBSERVATION, REWARD ============================#
        
    def get_obs(self):
        # Observation is a concatination of our joint positions, joint velocities,
        # joint efforts, and 
        
        #print("good")
        pos,vel,eff= self.get_joint_state()
        pos=pos%(2*np.pi) # MOD POSITION since it is an angle
        
        
        return np.concatenate([pos,vel,eff])
        
    def get_obs_dim(self):
        #print("Here")
        return 21

    def get_reward_done(self):
        debug=True
        if debug:
            print("\n--------------------")
        tip_coord = self.get_cartesian_points()[-1][-1] # should be max extension of fingers
        grabby_coord=self.get_cartesian_points()[-1][-2] # should be about where 'inside hand' is
        obj_dict=self.get_object_dict()
        total_reward = 0
        if(self.robot_intersects_self()):
            print("Ending episode because robot is intersecting itself")
            return -1, True
        return 1,False
    
    #========================= RESETTING ENVIRONMENT ==========================#
    def angle_wiggle(self):
        return np.array((0.,0.,np.random.normal(0,self.variance*2*np.pi)))
        
    def pos_wiggle(self):
        return np.random.normal(0,self.variance,3)
        
    def build_basic_tower(self,n=18,randomness=True):
        for h in range(n):
            for i in range(3):
                identifier='block_'+str(h)+'_'+str(i) # height, number of block
                rot=np.array((0.,0.,(h%2)*np.pi/2)) # rotate if odd level
                offset=np.zeros(3)
                offset+=(0,0,h*self.block_dim[2]) # height of level
                width=self.block_dim[1]
                if h%2:
                    offset+=((i-1)*(width+self.spacing),0,0)
                else:
                    offset+=(0,(i-1)*(width+self.spacing),0)
                pos=offset+self.spawn_point
                if randomness:
                    rot+=self.angle_wiggle()
                    pos+=self.pos_wiggle()
                self.spawn_model_from_name(self.block_model,identifier,pos,rot)
                
    def reset_tower(self):
        #print("RESETTing")
        # generate random new cup positions
        self.despawn_all()
        self.build_basic_tower()
    

    

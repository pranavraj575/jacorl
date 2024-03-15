from gazebo_msgs.msg import LinkStates, ModelState, ModelStates
from geometry_msgs.msg import Pose, Point, Quaternion
from gazebo_msgs.srv import DeleteModel, SpawnModel
from jaco_gym.envs.gazebo_env import JacoGazeboEnv
from jaco_gym.envs.task_envs.jenga.tower import Tower, JENGA_BLOCK_DIM
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
        
        self.spawn_point=np.array((.35,-.25,0.))
        self.block_model="jenga_block"
        
        

    def step(self,action):
        joint_obs,_,_,_=super().step(action)
        REWARD,DONE=self.get_reward_done()
        INFO={}
        obs=self.get_obs()
        #print("good")
        return obs,REWARD,DONE,INFO
    
    def reset(self):
        self.despawn_all()
        joint_obs=super().reset()
        self.build_basic_tower()
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

    
    def build_tower(self,tower:Tower):
        for h in range(tower.height):
            for i in range(3):
                if tower.blocks[h][i]:
                    dic=tower.block_info[h][i]
                    identifier='block_'+str(h)+'_'+str(i)
                    
                    self.spawn_model_from_name(self.block_model,identifier,dic['pos'],dic['rot'])
        
        return
        for h in range(tower.height):
            for i in range(3):
                if tower.blocks[h][i]:
                    identifier='block_'+str(h)+'_'+str(i) # height, number of block
                    rot=np.array((0.,0.,(h%2)*np.pi/2)) # rotate if odd level
                    offset=np.zeros(3)
                    offset+=(0,0,h*JENGA_BLOCK_DIM[2]) # height of level
                    width=JENGA_BLOCK_DIM[1]
                    if h%2:
                        offset+=((i-1)*(width+tower.spacing),0,0)
                    else:
                        offset+=(0,(i-1)*(width+tower.spacing),0)
                    pos=offset+self.spawn_point
                    if randomness:
                        rot+=tower.angle_wiggle()
                        pos+=tower.pos_wiggle()
                    self.spawn_model_from_name(self.block_model,identifier,pos,rot)

        
    def build_basic_tower(self,n=5,randomness=True):
        
        tower=Tower(  
                      blocks=[[True,True,True] for _ in range(n)],
                      spawn_point=self.spawn_point,
                      variance=.0015 if randomness else 0.
                      )
        self.build_tower(tower=tower)
    

    

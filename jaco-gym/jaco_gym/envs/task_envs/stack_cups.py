from jaco_gym.envs.robot_env import JacoEnv
class JacoStackCups(JacoEnv):
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
    
    
        super().__init__(ROBOT_NAME,CAM_SPACE,init_pos,differences,bounds)
    def step(self,action):
        
        
        joint_obs,_,_,_=super().step(action)
        
        
        REWARD=-1
        DONE=False
        INFO={}
        obs=self.get_obs()
        print("good")
        return obs,REWARD,DONE,INFO
    
    def reset(self):
        joint_obs=super().reset()
        print('here')
        return joint_obs
        
    def get_obs(self):
        print("good")
        pos,vel,eff= self.get_joint_state()
        return pos+vel+eff
        
        # should prob use self.get_joint_state as well as other stuff
        
    def get_obs_dim(self):
        print("Here")
        return 21
    

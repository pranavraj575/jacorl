from jaco_gym.envs.robot_env import JacoEnv
import numpy as np
class JacoStackCupsGazebo(JacoEnv):
    def __init__(self,
                    ROBOT_NAME='my_gen3',
                    CAM_SPACE='camera', #call will look for /CAM_SPACE/color/image_raw and /CAM_SPACE/depth/image_raw
                    init_pos=(0,15,230,0,55,90), #HOME position
                    differences=(15,15,15,15,15,15), # angular movement allowed at each joint per action
                    ):
    
    
        super().__init__(ROBOT_NAME,CAM_SPACE,init_pos,differences)
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
        obs=self.get_obs()
        return obs
        
    def get_obs(self):
        print("good")
        pos,vel,eff= self.get_joint_state()
        return np.array(pos+vel+eff)
        
        # should prob use self.get_joint_state as well as other stuff
        
    def get_obs_dim(self):
        print("Here")
        return 21
    

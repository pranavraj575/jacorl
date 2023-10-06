#from gazebo_msgs.msg import LinkStates, ModelState, ModelStates
#from geometry_msgs.msg import Pose, Point, Quaternion

from jaco_gym.envs.task_envs.stack_cups_gazebo import JacoStackCupsGazebo
import numpy as np
#import rospy
#import math
#from scipy.spatial.transform import Rotation

class JacoStackCupsGazeboImg(JacoStackCupsGazebo):
    def __init__(self,
                    ROBOT_NAME='my_gen3',
                    CAM_SPACE='camera', #call will look for /CAM_SPACE/color/image_raw
                    init_pos=(0,15,230,0,55,90), #HOME position
                    differences=(15,15,15,15,15,15), # angular movement allowed at each joint per action
                    image_dim=(128,128,3), # image vector, will resize input images to this
                    ):
                    
        super().__init__(ROBOT_NAME,CAM_SPACE,init_pos,differences,image_dim)
        # maybe change 

    def step(self,action):
        old_obs,REWARD,DONE,INFO=super().step(action)
        obs=self.get_full_obs()
        return obs,REWARD,DONE,INFO
    
    def reset(self):
        old_obs=super().reset()
        obs=self.get_full_obs()
        return obs
    
    def get_obs_dim(self):
        #print("Here")
        return np.prod(self.image_dim) + 21+3*6
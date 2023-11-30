from gazebo_msgs.msg import LinkStates, ModelState, ModelStates
from geometry_msgs.msg import Pose, Point, Quaternion
from gazebo_msgs.srv import DeleteModel, SpawnModel
from jaco_gym.envs.gazebo_env import JacoGazeboEnv
import numpy as np
import rospy
import math
from scipy.spatial.transform import Rotation
import os

class JacoGrabCupGazebo(JacoGazeboEnv):
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
        
        # Ranges for randomizing cups and determining goal
        self.table_y_range=(-0.49,0.09)
        self.cup_ranges=((0.3,0.7),self.table_y_range)
        self.cup_goal_x = 0.3 # or above
        self.max_cup_x = self.cup_ranges[0][1]
        self.cup_model="solo_cup"
        

    def step(self,action):
        joint_obs,_,_,_=super().step(action)
        REWARD,DONE=self.get_reward_done()
        INFO={}
        obs=self.get_obs()
        #print("good")
        return obs,REWARD,DONE,INFO
    
    def reset(self):
        joint_obs=super().reset()
        self.reset_cup()
        #print('RESETTING CUPS')
        cup_pos=self.get_pose_eulerian('cup')[:3]
        x,y,h,gamma=self.look_at_cup(cup_pos[0],cup_pos[1])
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
                                [self.get_pose_eulerian('cup')]
                                )
        
    def get_obs_dim(self):
        #print("Here")
        return 21+1*6

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
        cup_p=self.get_pose_eulerian("cup")
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
    def find_chord(self,R,c,d):
        # given a triangle with base R, sides c and d, finds intersection angle of the d side
        a=(d**2-c**2+R**2)/(2*R)
        return np.arccos(a/d)
        
    def look_at_cup(self,x,y,sight_dist=.3): 
        # returns x,y,h,deg to look at cup on table
        # sight_dist is how far away to look at cup
        temp=np.array((x,y))
        r=np.linalg.norm(temp)
        new_r=max(r-.04,.04)
        x,y=temp*new_r/r
        
        
        cup_vec=np.array((x,y,-self.LENGTHS[0]-self.LENGTHS[1])) # vector to cup
        R=np.linalg.norm(cup_vec)
        r=np.linalg.norm(cup_vec[:2])
        theta=np.arctan2(cup_vec[2],r) # angle from arm shoulder to cup
        
        
        d=sight_dist+self.LENGTHS[5]+self.LENGTHS[6] # this is how far the wrist tilt joint should be
        c_low,c_high = self.wrist_tilt_bounds()
        
        #print('wrist bounds:',c_low,c_high)
        #print('d:',d)
        #print('R:',R)
        
        if R>=c_high+d:
            #print('returning since',R,">=",c_high,'+',d)
            xp,yp,dh=cup_vec*c_high/R
            h=self.LENGTHS[0]+self.LENGTHS[1]+dh
            return xp,yp,h,theta
        if R<=c_low:
            #print('returning since',R,"<=",c_low)
            return x,y,d,-90
        
        bounds=[np.radians(10),np.radians(90)] # bounds of angle from the cup
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
        xp,yp=cup_vec[:2]*(r-eaten)/r
        
        #print(r-eaten,h,-np.degrees(angle))
        
        
        return xp,yp,h,-np.degrees(angle)
        
        
    def set_cup_ranges(self,x_range,y_range):
        self.cup_ranges=x_range,y_range
    
    def reset_cup(self,prob_stand=1,prob_flip=0,prob_other=0): # input the probabilities that the cup is spawned normal, flipped, or fallen
        #print("RESETTing")
        # generate random new cup positions
        self.despawn_all()
        
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
        
        self.spawn_model_from_name(self.cup_model,'cup',(x,y,.065 if not yikes else .1),rot)
    
    #========================= CUP INFORMATION ==========================#
    def _inversion_check(self,name,tol=.02): 
        # ignores rotation about z axis from starting position
        # if object is not standing either upside down or the same as starting position, return -1
        # if object is standing in the same position as start, return 0
        # if object is upside down, return 1
        (roll,pitch,yaw)=self.get_pose_eulerian(name)[3:]%(2*np.pi) # now only positives
        roll_normal=bool(min(roll,abs(roll-2*np.pi))<=tol) # either roll is 0 or 2pi to make this true
        roll_inversion=bool(abs(np.pi-roll)<=tol) # roll is pi to make this true
        if not (roll_normal or roll_inversion):
            return -1 # not standing up or inverted, fell over
        
        pitch_normal=bool(min(pitch,abs(pitch-2*np.pi))<=tol)
        pitch_inversion=bool(abs(np.pi-abs(pitch))<=tol)
        if not (pitch_normal or pitch_inversion):
            return -1 # not standing up or inverted, fell over
        return int(roll_inversion^pitch_inversion) #returns 1 if exactly one of these are true (i.e if cup is flipped once), 0 if inverted twice or nonce
        
    def is_standing_up(self,name,tol=.02):
        #returns if object is only rotated about z axis. (if the object is still standing in the same starting position)
        return self._inversion_check(name,tol)==0
        
    def _is_upside_down(self,name):
        #returns if object named name is flipped (ignoring rotation on z axis, if object is exactly upside down)
        return self._inversion_check(name)==1
    
    def _is_fallen_over(self,name):
        #returns if object named name is fallen over
        return self._inversion_check(name)==-1
    
    def robot_holding_cup_position(self,min_grab_pos=0.209, min_grab_eff=1.05e-1): 
        joint_positions,_,joint_efforts = self.get_joint_state()
        finger_pos = joint_positions[6]
        finger_eff = joint_efforts[6]
        return finger_pos >= min_grab_pos and finger_eff >= min_grab_eff

    def get_tip_coord(self):
        return self.get_cartesian_points()[-1][-1]
    

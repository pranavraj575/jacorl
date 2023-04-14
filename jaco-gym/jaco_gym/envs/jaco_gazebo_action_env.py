import gym
import numpy as np
import random
import math

from gym import error, spaces, utils
from gym.utils import seeding
from jaco_gym.envs.ros_scripts.jaco_gazebo_action_client import JacoGazeboActionClient

class JacoEnv(gym.Env):

    def __init__(self):
        self.robot = JacoGazeboActionClient()
        self.action_dim = 7 #6 ADDED GRIPPY BOY
        self.obs_dim = self.robot.get_obs_dim()
        #self.obs_dim = 12   # when using read_state_simple
        high = np.ones([self.action_dim])
        self.action_space = gym.spaces.Box(-high, high)
        high = np.inf * np.ones([self.obs_dim])
        self.observation_space = gym.spaces.Box(-high, high)
        self.table_y_range=(-0.29,0.29)
        self.cup_ranges=((-1.4,-0.31),self.table_y_range)
        self.cup_goal_x = -0.3 # or below
        
        
        
        self.BOUNDS=[
        (0,360), #UNBOUNDED, arm can rotate
        (240,120),#(60,300), #this goes about (230, 130) IRL with 0 being straight up, about 130 degrees each side. in simulation, 180 is straight up
        (220,140),#(25,335), # IRL (212,147) with  0 straight up, about 140 each side. in simulation, 180 is straight up
        (0,360), # UNBOUNDED
        (235,115), # (239,120) with 0 straight up, about 115 each side. In simulation, 0 is still straight up
        (0,360), # UNBOUNDED
        ]
        #BOUNDS HERE FOR EACH JOINT in degrees

    def convert_action_to_deg(self, a, OldMin, OldMax, NewMin, NewMax):
        OldRange = (OldMax - OldMin)  
        NewRange = (NewMax - NewMin)  
        return (((a - OldMin) * NewRange) / OldRange) + NewMin
    
    
    def action2deg(self, action): #OLD
        for i in range(len(action)):
            NewMin,NewMax=self.BOUNDS[i] 
            if NewMax<= NewMin:
                NewMax=NewMax+360
            action[i]=self.convert_action_to_deg(action[i],OldMin=-1, OldMax=1, NewMin=NewMin,NewMax=NewMax)%360
        return action
    
    def action2diff(self,action):
        action[0] = self.convert_action_to_deg(action[0], OldMin=-1, OldMax=1, NewMin=-15, NewMax=15) # base rotation
        action[1] = self.convert_action_to_deg(action[1], OldMin=-1, OldMax=1, NewMin=-15, NewMax=15) # second joint, arm lever
        action[2] = self.convert_action_to_deg(action[2], OldMin=-1, OldMax=1, NewMin=-15, NewMax=15) # third joint, elbow thingy
        action[3] = self.convert_action_to_deg(action[3], OldMin=-1, OldMax=1, NewMin=-15, NewMax=15) # fourth joint, arm twist
        action[4] = self.convert_action_to_deg(action[4], OldMin=-1, OldMax=1, NewMin=-15, NewMax=15) # fifth joint, arm twist 2
        action[5] = self.convert_action_to_deg(action[5], OldMin=-1, OldMax=1, NewMin=-15, NewMax=15) # sixth join, wrist rotation
        
        #DIFFERENT: gripper just given an activation
        action[6] = self.convert_action_to_deg(action[6], OldMin=-1, OldMax=1, NewMin=0, NewMax=90) # gripper, given full ROM
        return action
        
    def diff2deg(self,da,old_pos):
        #NOTE: old_pos is REAL angles, was converted in GET_OBS
        # WE WILL WORK IN REAL ANGLES
        at_bounds=[]
        for i in range(len(da)):
            if self.BOUNDS[i]==(0,360): #UNBOUNDED
                da[i]=(old_pos[i]+da[i])%360
            else:
                l_bound,h_bound=self.BOUNDS[i]
                da[i]=(old_pos[i]+da[i])%360
                
                if l_bound<h_bound:
                    
                    da[i]=np.clip(da[i],l_bound,h_bound)
                    at_bounds.append(da[i]==l_bound or da[i]==h_bound)
                    da[i]=da[i]%360
                else:
                    #extra work needed
                    mid=(l_bound+h_bound)/2
                    
                    if da[i]>=mid:
                        da[i]-=360
                        # puts this value on [-360+mid,mid]
                    l_bound-=360
                    ## now its -360 < -360+mid < l_bound < 0 < h_bound < h_bound+mid < 360
                    # hopefully da[i] is between lbound and h_bound
                    da[i]=np.clip(da[i],l_bound,h_bound)
                    at_bounds.append(da[i]==l_bound or da[i]==h_bound)
                    da[i]=da[i]%360

        return da,at_bounds
    
        

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
    
    def cup_in_hand(self,cup_pos,tol=0.05):
        # Distance between finger coordinates is "small"
        # Cup center is between finger coordinates
        # Fingertips have effort
        # f1, f2 = self.robot.get_finger_coords()
        # f1[0] 
        pass

    def step(self, action):
        #self.action = self.action2deg(action) # convert action from range [-1, 1] to [0, 360] 
        #self.action = np.radians(self.action) # convert to radians         
        
        old_pos=self.robot.get_obs()[:6] # loads the positions in REAL ANGLES
        
        old_pos=np.degrees(old_pos)
        self.action=self.action2diff(action)
        self.action[:6],at_bounds=self.diff2deg(self.action[:6],old_pos)
        print(at_bounds)
        
        self.action=np.radians(self.action)
        diff=self.robot.move_arm(np.array(self.action[:6])) # move arm 
        self.robot.move_finger(self.action[6]) # move fingy
        
        #print(diff, np.linalg.norm(diff)) # the difference between original position and the new position, 
                                          #in radians, measure of how much angle movement this made, could be useful
        self.robot.sleepy(0.2) # wait
                
        self.observation = self.robot.get_obs()#_simple()   # get state, only return 12 values instead of 36
        
        #===================== Calculate Reward ====================#

        self.tip_coord = self.robot.get_tip_coord()
        self.reward = 100
        closest_dist = 100
        obj_data = self.robot.get_object_data()
        cups = ["cup1","cup2","cup3"]
        for cup in cups:
            pos = obj_data[cup].position
            self.cup_in_hand(pos)
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
        g = 10
        d = 0.1
        if(closest_dist != 100):
            print("Closest cup is ",closest_dist)
            self.reward += g/max((closest_dist**2),d**2)
        print("Reward is ",self.reward)

        #===========================================================#
       
        # create info
        self.info = {"tip coordinates": self.tip_coord}#, "target coordinates": self.target_vect}
        
        # create done
        self.done = False
        
        return self.observation, self.reward, self.done, self.info

    def cup_has_collision(self,x,y,tol=.08):
        for pos in self.cup_positions:
            x2 = pos[0]
            y2 = pos[1]
            dist = math.sqrt((x2-x)*(x2-x) + (y2-y)*(y2-y))
            if(dist <= tol):
                return True
        return False

    def reset(self): 

        self.robot.cancel_move()
        self.robot.move_finger(0)
        #init_pos=[1,-.3,-.7,0,1,0,-1]
        
        #changes to specific pose
        
        #init_pos=[-.865,-.2,-.7,.2,-.8,-.22,-1]
        #init_pos=self.action2deg(init_pos)
        #init_pos=np.radians(init_pos)
        #print(init_pos)
        
        init_pos=np.array([16.,  0., 220., 36.,  36., 50., 0.])
        # in DEGREES the initial angles
        init_pos=np.radians(init_pos)
        # in radians the inital angle
        
        
        #pos = [0, 180, 180, 0, 0, 0]
        #pos = np.radians(pos)
        self.robot.move_arm(init_pos[:6])
        self.robot.move_finger(init_pos[6])
        print("Jaco reset to initial position")
        self.obs = self.robot.get_obs()#_simple() # get observation

        # generate random new cup positions
        cup_names = ["cup1", "cup2", "cup3"]
        self.cup_positions = []
        for i in range(len(cup_names)):
            x = random.uniform(self.cup_ranges[0][0],self.cup_ranges[0][1])
            y = random.uniform(self.cup_ranges[1][0],self.cup_ranges[1][1])
            while(self.cup_has_collision(x,y)):
                x = random.uniform(self.cup_ranges[0][0],self.cup_ranges[0][1])
                y = random.uniform(self.cup_ranges[1][0],self.cup_ranges[1][1])
            self.cup_positions.append((x,y,.065))
        self.robot.move_cups(self.cup_positions)

        return self.obs


    def render(self, mode='human', close=False):
        pass

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
        # self.obs_dim = 30
        self.obs_dim = 12   # when using read_state_simple
        high = np.ones([self.action_dim])
        self.action_space = gym.spaces.Box(-high, high)
        high = np.inf * np.ones([self.obs_dim])
        self.observation_space = gym.spaces.Box(-high, high)
        self.table_y_range=(-0.29,0.29)
        self.cup_ranges=((-1.4,-0.31),self.table_y_range)
        self.cup_goal_x = -0.3 # or below

    def convert_action_to_deg(self, a, OldMin, OldMax, NewMin, NewMax):
        OldRange = (OldMax - OldMin)  
        NewRange = (NewMax - NewMin)  
        return (((a - OldMin) * NewRange) / OldRange) + NewMin
    
    
    def action2deg(self, action):
        action[0] = self.convert_action_to_deg(action[0], OldMin=-1, OldMax=1, NewMin=0, NewMax=360)
        action[1] = self.convert_action_to_deg(action[1], OldMin=-1, OldMax=1, NewMin=90, NewMax=270)
        action[2] = self.convert_action_to_deg(action[2], OldMin=-1, OldMax=1, NewMin=0, NewMax=180)
        action[3] = self.convert_action_to_deg(action[3], OldMin=-1, OldMax=1, NewMin=0, NewMax=360)
        action[4] = self.convert_action_to_deg(action[4], OldMin=-1, OldMax=1, NewMin=0, NewMax=360)
        action[5] = self.convert_action_to_deg(action[5], OldMin=-1, OldMax=1, NewMin=0, NewMax=360)
        action[6] = self.convert_action_to_deg(action[6], OldMin=-1, OldMax=1, NewMin=0, NewMax=90)
        return action

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
        self.robot.read_state()
        print(self.robot.eff[6:])

    def step(self, action):
        self.action = self.action2deg(action) # convert action from range [-1, 1] to [0, 360] 
        self.action = np.radians(self.action) # convert to radians         
        diff=self.robot.move_arm(self.action[:6]) # move arm 
        self.robot.move_finger(self.action[6]) # move fingy
        
        #print(diff, np.linalg.norm(diff)) # the difference between original position and the new position, 
                                          #in radians, measure of how much angle movement this made, could be useful
        self.robot.sleepy(2) # wait
                
        self.observation = self.robot.read_state_simple()   # get state, only return 12 values instead of 36
        
        #===================== Calculate Reward ====================#

        self.tip_coord = self.robot.get_tip_coord()
        self.reward = 100
        closest_dist = 100
        obj_data = self.robot.get_object_data()
        cups = ["cup1","cup2","cup3"]
        for cup in cups:
            pos = obj_data[cup].position
            self.cup_in_hand(pos)
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
        init_pos=[-.865,-.2,-.7,.2,-.8,-.22,-1]
        init_pos=self.action2deg(init_pos)
        init_pos=np.radians(init_pos)
        #pos = [0, 180, 180, 0, 0, 0]
        #pos = np.radians(pos)
        self.robot.move_arm(init_pos[:6])
        self.robot.move_finger(init_pos[6])
        print("Jaco reset to initial position")
        self.obs = self.robot.read_state_simple() # get observation

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

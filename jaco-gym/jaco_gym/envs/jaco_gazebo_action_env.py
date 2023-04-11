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
        # self.obs_dim = 36
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

    def step(self, action):
        self.action = self.action2deg(action) # convert action from range [-1, 1] to [0, 360] 
        self.action = np.radians(self.action) # convert to radians    
        self.robot.move_arm(self.action[:6]) # move arm 
        self.robot.move_finger(self.action[6]) # move fingy
        self.robot.sleepy(2) # wait
        self.observation = self.robot.read_state_simple()   # get state, only return 12 values instead of 36
        
        #===================== Calculate Reward ====================#

        self.tip_coord = self.robot.get_tip_coord()
        self.dist_to_target = np.linalg.norm(self.tip_coord - self.target_vect)
        self.reward = - self.dist_to_target 

        closest_dist = 1000
        for (x,y) in self.cup_positions:
            # ADD THE REWARD FOR ROBOT TO CLOSEST CUP
            # Assign a negative reward for each cup that is off the table
            if (y > self.table_y_range[1] or y < self.table_y_range[0]):
                self.reward -= 50
            else:  # Large positive reward for each cup in the goal zone
                if(x >= self.cup_goal_x):
                    self.reward += 100
                else: # Negative reward for cups farther from goal
                    dist_to_goal = self.cup_goal_x - x
                    self.reward -= dist_to_goal * 10
            

        #===========================================================#
       
        # create info
        self.info = {"tip coordinates": self.tip_coord, "target coordinates": self.target_vect}
        
        # create done
        self.done = False

        # IF DEFINING DONE AS FOLLOWS, THE EPISODE ENDS EARLY AND A GOOD AGENT WILL RECEIVED A PENALTY FOR BEING GOOD
        # COOMENT THIS
        # if self.dist_to_target < 0.01:
            # self.done = True
            
        # print("tip position: ", self.tip_coord)
        # print("target vect: ", self.target_vect)
        # print("dist_to_target: ", self.dist_to_target)

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
        init_pos=[1,-.3,-.7,0,1,0,-1]
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
            self.cup_positions.append((x,y))
        self.robot.move_cups(self.cup_positions)

        # Target stuff we don't care about
        x_target = random.uniform(-0.335, 0.335)
        y_target = random.uniform(-0.337, 0.337)
        z_target = random.uniform(0.686, 1.021)
        
        self.target_vect = np.array([x_target, y_target, z_target])


        print("Random target coordinates generated")

        # if testing: graphically move the sphere target, if training, comment this line
        # self.robot.move_sphere(self.target_vect)
        return self.obs


    def render(self, mode='human', close=False):
        pass

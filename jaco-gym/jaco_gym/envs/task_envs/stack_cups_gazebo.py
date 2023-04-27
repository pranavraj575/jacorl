from gazebo_msgs.msg import LinkStates, ModelState, ModelStates
from geometry_msgs.msg import Pose, Point, Quaternion
from jaco_gym.envs.robot_env import JacoEnv
import numpy as np
import rospy
import random
import math
from scipy.spatial.transform import Rotation

class JacoStackCupsGazebo(JacoEnv):
    def __init__(self,
                    ROBOT_NAME='my_gen3',
                    CAM_SPACE='camera', #call will look for /CAM_SPACE/color/image_raw
                    init_pos=(0,15,230,0,55,90), #HOME position
                    differences=(15,15,15,15,15,15), # angular movement allowed at each joint per action
                    ):
                    
        super().__init__(ROBOT_NAME,CAM_SPACE,init_pos,differences)
        self.pub_topic = '/gazebo/set_model_state'
        self.pub = rospy.Publisher(self.pub_topic, ModelState, queue_size=1)

        # Ranges for randomizing cups and determining goal
        self.table_y_range=(-0.49,0.09)
        self.cup_ranges=((1.4,0.31),self.table_y_range)
        self.cup_goal_x = 0.3 # or above
        self.max_cup_x = self.cup_ranges[0][0]

        # Subscribe to object data to obtain cup locations
        self.object_data={}
        def _call_model_data(data):
            self.object_data={}
            for i in range(len(data.name)):
                self.object_data[data.name[i]] = data.pose[i]
        self.sub_topic="/gazebo/model_states"
        self.sub=rospy.Subscriber(self.sub_topic,ModelStates,_call_model_data)

    def step(self,action):
        joint_obs,_,_,_=super().step(action)
        REWARD,DONE=self.get_reward_done()
        INFO={}
        obs=self.get_obs()
        print("good")
        return obs,REWARD,DONE,INFO
    
    def reset(self):
        self.reset_cups()
        print('RESETTING CUPS')
        joint_obs=super().reset()
        obs=self.get_obs()
        return obs
    
    #========================= OBSERVATION, REWARD ============================#
    def get_pose_eulerian(self,name):
        # returns numpy array with pose of an object, includes x,y,z and eulerian rotation
        position=self.object_data[name].position
        x,y,z=position.x,position.y,position.z
        orientation=self.object_data[name].orientation
        (roll,pitch,yaw)=Rotation.from_quat((orientation.x,orientation.y,orientation.z,orientation.w)).as_euler('xyz')
        return np.array([x,y,z,roll,pitch,yaw])
        
    def get_obs(self):
        # Observation is a concatination of our joint positions, joint velocities,
        # joint efforts, and the x,y,z coordinates of each of our 3 cups
        
        #print("good")
        pos,vel,eff= self.get_joint_state()
        pos=pos%(2*np.pi) # MOD POSITION since it is an angle
        
        object_names=['cup1','cup2','cup3']
        
        return np.concatenate([pos,vel,eff] +
                                [self.get_pose_eulerian(name) for name in object_names]
                                )
        
    def get_obs_dim(self):
        print("Here")
        return 21+3*6

    def get_reward_done(self):
        print("\n--------------------")
        tip_coord = self.get_tip_coord() 
        total_reward = 0
        if(self.robot_intersects_self()):
            print("Ending episode because robot is intersecting itself")
            return -1, True

        # Reward factors
        dist_to_cup_r = 0     #1 = touching cup, 0 = fal from cup
        cups_near_goal_r = 0  #1 = all cups at goal, 0 = cups at max dist from goal
        cups_at_goal_r = 0    #1 = all cups at goal, 0 = no cups at goal --- is this repetitive?
        robot_near_table = 1  #1 = robot in general vacinity of table, 0 = robot way far from table
        robot_holding_cup_r = self.robot_holding_cup_r() 

        cups = ["cup1","cup2","cup3"]
        num_cups = len(cups)
        closest_dist = None
        max_tip_to_cup_dist = self.max_cup_x
        max_cup_to_goal_dist = self.max_cup_x - self.cup_goal_x
        for cup in cups:
            pos = self.object_data[cup].position
            # A cup fell off the table, end the episode
            if (not self.cup_on_table(pos)):
                print("Ending episode because a cup fell off the table")
                return -1, True
            else:
                # Is cup at the goal zone?
                if(self.cup_at_goal_loc(pos)):
                    cups_at_goal_r += 1/num_cups
                else: 
                    # Calculate the distance of this cup to the goal
                    my_dist_to_goal = abs(self.cup_goal_x - pos.x)
                    cups_near_goal_r += 1/num_cups * max((1-my_dist_to_goal),0)
                    dist_to_cup = np.linalg.norm(tip_coord - np.array([pos.x,pos.y,pos.z])) 
                    if(closest_dist == None or dist_to_cup < closest_dist):
                        closest_dist = dist_to_cup

        # Determine distance to closest cup
        if(closest_dist != None):
            print(cup, " is dist ", closest_dist)
            dist_to_cup_r = max(min(max_tip_to_cup_dist - closest_dist,1),0)
            #print(dist_to_cup_r)
        
        # Want robot to be within table range
        x,y,z = tip_coord
        max_z = 0.5
        if (y <= self.table_y_range[0] or y >= self.table_y_range[1] or z >= max_z):
            y_penalty = np.linalg.norm(y-self.table_y_range)
            z_penalty = np.linalg.norm(z-max_z)
            robot_near_table = 1 - min((y_penalty + z_penalty)*10,1)
        
        reward_factors = [dist_to_cup_r,
                          cups_near_goal_r,
                          cups_at_goal_r,
                          robot_near_table,
                          robot_holding_cup_r]
        reward_weights = [1,1,1,1,1]

        print(reward_factors)
        for i in range(len(reward_factors)):
            total_reward += reward_factors[i] * reward_weights[i]
        total_reward = total_reward / sum(reward_weights) # Normalize

        print("Reward is ",total_reward)
        print("--------------------\n")
        return total_reward, False
    
    #========================= RESETTING ENVIRONMENT ==========================#
    
    def reset_cups(self):
        print("RESETTing")
        # generate random new cup positions
        cup_names = ["cup1", "cup2", "cup3"]
        cup_positions = []
        for i in range(len(cup_names)):
            x = random.uniform(self.cup_ranges[0][0],self.cup_ranges[0][1])
            y = random.uniform(self.cup_ranges[1][0],self.cup_ranges[1][1])
            while(self.cup_has_collision(x,y,cup_positions)):
                x = random.uniform(self.cup_ranges[0][0],self.cup_ranges[0][1])
                y = random.uniform(self.cup_ranges[1][0],self.cup_ranges[1][1])
            cup_positions.append((x,y,.065))
        self.move_cups(cup_positions)
    
    # To make sure random cup locations do not intersect
    def cup_has_collision(self,x,y,cup_positions,tol=.08):
        for pos in cup_positions:
            x2 = pos[0]
            y2 = pos[1]
            dist = math.sqrt((x2-x)*(x2-x) + (y2-y)*(y2-y))
            if(dist <= tol):
                return True
        return False
    
    def move_cups(self, positions,orientations=None):
        print("moving cups")
        # move cups to the randomized positions
        cup_names = ["cup1", "cup2", "cup3"]
        for zs in [[-1.]*3,[p[2] for p in positions]]:
            for i in range(len(cup_names)):
                model_state_msg = ModelState()
                pose_msg = Pose()
                point_msg = Point()
                rot_msg=Quaternion()#default no rotation
                if orientations:
                  (roll,pitch,yaw)=orientations[i]
                  stuff=Rotation.from_euler('xyz',(roll,pitch,yaw)).as_quat()
                  (rot_msg.x,rot_msg.y,rot_msg.z,rot_msg.w)=stuff
                (x,y,_)=positions[i]
                point_msg.x = x
                point_msg.y = y
                point_msg.z = zs[i]
                pose_msg.position = point_msg
                pose_msg.orientation = rot_msg
                model_state_msg.model_name = cup_names[i]
                model_state_msg.pose = pose_msg
                model_state_msg.reference_frame = "world"
                self.pub.publish(model_state_msg)
                rospy.sleep(.01)
    
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
        return self._inversion_check(name,tol)==1
    
    def _is_fallen_over(self,name):
        #returns if object named name is fallen over
        return self._inversion_check(name,tol)==-1
        

    def cup_on_table(self,pos):
        return pos.z >= 0
    
    def cup_at_goal_loc(self,pos):
        return self.cup_on_table(pos) and (pos.x <= self.cup_goal_x)
    
    def robot_holding_cup_r(self,min_grab_pos=0.209, min_grab_eff=1.05e-1): 
        joint_positions,_,joint_efforts = self.get_joint_state()
        finger_pos = joint_positions[6]
        finger_eff = joint_efforts[6]
        return finger_pos >= min_grab_pos and finger_eff >= min_grab_eff

    def get_tip_coord(self):
        return self.get_cartesian_points()[-1][-1]
    

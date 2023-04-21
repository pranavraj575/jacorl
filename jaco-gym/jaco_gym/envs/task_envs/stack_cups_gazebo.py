from gazebo_msgs.msg import LinkStates, ModelState, ModelStates
from geometry_msgs.msg import Pose, Point, Quaternion
from jaco_gym.envs.robot_env import JacoEnv
import numpy as np
import rospy
import random
import math

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
        self.table_y_range=(-0.29,0.29)
        self.cup_ranges=((1.4,0.31),self.table_y_range)
        self.cup_goal_x = 0.3 # or above

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
        REWARD=self.get_reward()
        DONE=False
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
        
    def get_obs(self):
        # Observation is a concatination of our joint positions, joint velocities,
        # joint efforts, and the x,y,z coordinates of each of our 3 cups

        print("good")
        pos,vel,eff= self.get_joint_state()
        pos=pos%(2*np.pi) # MOD POSITION since it is an angle
        cup1 = self.object_data["cup1"].position
        cup2 = self.object_data["cup2"].position
        cup3 = self.object_data["cup3"].position
        cup_positions = np.array([cup1.x, cup1.y, cup1.z, cup2.x, cup2.y, cup2.z, cup3.z, cup3.y, cup3.z])
        return np.concatenate((pos,vel,eff,cup_positions))
        
    def get_obs_dim(self):
        print("Here")
        return 30

    def get_reward(self):
        self.tip_coord = self.get_tip_coord() # This is not going to work yet
        self.reward = 100
        closest_dist = 100
        cups = ["cup1","cup2","cup3"]
        for cup in cups:
            pos = self.object_data[cup].position
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
                    # FIX ME!!! 
                    #dist_to_cup = np.linalg.norm(self.tip_coord - np.array([pos.x,pos.y,pos.z])) 
                    # if(dist_to_cup < closest_dist):
                    #     closest_dist = dist_to_cup
        # Reward incentivising robot tip to be close to the nearest cup not 
        # already in the goal zone, as long as there are sitll cups not at goal
        if(closest_dist != 100):
            print(cup, " is dist ", closest_dist)
            self.reward -= closest_dist * 10
        print("Reward is ",self.reward)
    
    #========================= RESETTING ENVIRONMENT ==========================#
    
    def reset_cups(self):
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
        # move cups to the randomized positions
        cup_names = ["cup1", "cup2", "cup3"]
        for zs in [[-1]*3,[p[2] for p in positions]]:
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
    
    def is_upside_down(self,orientation,tol=.02):
            # Orientation is a Quaternion object (example: orientation = 
            # self.doota['cup1'].orientation), will convert to roll, pitch, 
            # yaw (rotation on x,y,z axis), ignore z axis and see if cup is 
            # exactly upside down
            (roll,pitch,yaw)=Rotation.from_quat((orientation.x,orientation.y,orientation.z,orientation.w)).as_euler('xyz')
            roll_inversion=bool(abs(np.pi-abs(roll))<=tol)
            pitch_inversion=bool(abs(np.pi-abs(pitch))<=tol)
            return roll_inversion^pitch_inversion #returns if exactly one of these are true (i.e if cup is flipped once)

    def cup_on_table(self,pos):
        return pos.z >= 0
    
    def cup_at_goal_loc(self,pos):
        return self.cup_on_table(pos) and (pos.x <= self.cup_goal_x)
    
    def robot_holding_cup(self,min_grab_pos=0.209, min_grab_eff=1.05e-1): 
        joint_positions,_,joint_efforts = self.get_joint_state()
        finger_pos = joint_positions[6]
        finger_eff = joint_efforts[6]
        return finger_pos >= min_grap_pos and finger_eff >= min_grab_eff

    def get_tip_coord(self):
        print("IMPLEMENT THIS")
        return self.get_joint_state()[0][:6] 
    

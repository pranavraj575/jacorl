from gazebo_msgs.msg import LinkStates, ModelState, ModelStates
from geometry_msgs.msg import Pose, Point, Quaternion
from gazebo_msgs.srv import DeleteModel, SpawnModel
from jaco_gym.envs.robot_env import JacoEnv
import numpy as np
import rospy
import math
from scipy.spatial.transform import Rotation
import os

class JacoStackCupsGazebo(JacoEnv):
    def __init__(self,
                    ROBOT_NAME='my_gen3',
                    CAM_SPACE='camera', #call will look for /CAM_SPACE/color/image_raw
                    init_pos=(0,15,230,0,55,90), #HOME position
                    differences=(15,15,15,15,15,15), # angular movement allowed at each joint per action
                    image_dim=(128,128,3), # image vector, will resize input images to this
                    ):
                    
        super().__init__(ROBOT_NAME,CAM_SPACE,init_pos,differences,image_dim)
        
        #basic gazebo thingies
        rospy.wait_for_service("gazebo/delete_model")
        rospy.wait_for_service("gazebo/spawn_sdf_model")
        self.delete_model = rospy.ServiceProxy("gazebo/delete_model", DeleteModel)
        self.spawn_model = rospy.ServiceProxy("gazebo/spawn_sdf_model", SpawnModel)
        
        
        self.pub_topic = '/gazebo/set_model_state'
        self.pub = rospy.Publisher(self.pub_topic, ModelState, queue_size=1)

        # task specific stuff
        
        # Ranges for randomizing cups and determining goal
        self.table_y_range=(-0.49,0.09)
        self.cup_ranges=((0.3,0.7),self.table_y_range)
        self.cup_goal_x = 0.3 # or above
        self.max_cup_x = self.cup_ranges[0][1]
        
        self.cup_xml=None
        for path in os.environ.get('GAZEBO_MODEL_PATH', 'Nonesuch').split(':'):
            model_path=os.path.join(path,'solo_cup','model.sdf')
            if os.path.exists(model_path):
                with open(model_path,'r') as f:
                    self.cup_xml=f.read()
        
        
        self.cup_names=[]

        # Subscribe to object data to obtain cup locations
        def _call_model_data(data):
            self.object_data=data
            
        self.sub_topic="/gazebo/model_states"
        self.sub=rospy.Subscriber(self.sub_topic,ModelStates,_call_model_data)
        

    def step(self,action):
        joint_obs,_,_,_=super().step(action)
        REWARD,DONE=self.get_reward_done()
        INFO={}
        obs=self.get_obs()
        #print("good")
        return obs,REWARD,DONE,INFO
    
    def reset(self):
        self.reset_cups()
        #print('RESETTING CUPS')
        joint_obs=super().reset()
        obs=self.get_obs()
        return obs
    
    #========================= OBSERVATION, REWARD ============================#
    def get_object_dict(self):
        return {self.object_data.name[i]:self.object_data.pose[i] for i in range(len(self.object_data.name))}
    def get_pose_eulerian(self,name):
        # returns numpy array with pose of an object, includes x,y,z and eulerian rotation
        obj_dict=self.get_object_dict()
        position=obj_dict[name].position
        x,y,z=position.x,position.y,position.z
        orientation=obj_dict[name].orientation
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
        #print("Here")
        return 21+3*6

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

        # Reward factors
        dist_to_cup_r = 0     #1 = touching cup, 0 = far from cup
        cups_near_goal_r = 0  #1 = all cups at goal, 0 = cups at max dist from goal
        cups_at_goal_r = 0    #1 = all cups at goal, 0 = no cups at goal --- is this repetitive?
        robot_near_table = 1  #1 = robot in general vacinity of table, 0 = robot way far from table
        grabbin_somethin = self.robot_holding_cup_position()
        robot_holding_cup_r=0

        cups = ["cup1","cup2","cup3"]
        num_cups = len(cups)
        closest_dist = None
        max_tip_to_cup_dist = self.max_cup_x
        max_cup_to_goal_dist = self.max_cup_x - self.cup_goal_x
        for cup in cups:
            if not cup in obj_dict:
                obs=self.reset()
                print("HOW DID WE GET HERE,",cup,'DOES NOT EXIST IN SIMULATION, RESTARTING')
                return 0,False
            pos = obj_dict[cup].position
            # A cup fell off the table, end the episode
            if pos.z<-.1:
                print("Ending episode because a cup fell off the table")
                return -1, True
            
            # Is cup at the goal zone?
            if pos.x <= self.cup_goal_x:
                cups_at_goal_r += 1/num_cups
                cups_near_goal_r += 1/num_cups # prevent annoying errors by just setting this to max if this cup is in goal
            else: 
                # Calculate the distance of this cup to the goal
                my_dist_to_goal = abs(self.cup_goal_x - pos.x)/max_cup_to_goal_dist
                cups_near_goal_r += max((1-my_dist_to_goal),0)/num_cups
                dist_to_cup = np.linalg.norm(tip_coord - np.array([pos.x,pos.y,pos.z])) 
                grabby_dist= np.linalg.norm(grabby_coord - np.array([pos.x,pos.y,pos.z])) 
                if grabby_dist<=.03 and grabbin_somethin: # if center of hand within this, check if cup is grabbin
                    robot_holding_cup_r=1
                if(closest_dist == None or dist_to_cup < closest_dist):
                    closest_dist = dist_to_cup

        # Determine distance to closest cup
        if(closest_dist != None):
            #print(cup, " is dist ", closest_dist)
            dist_to_cup_r = .2/(max(closest_dist,.04)**.5) # function looks like a downward swoop, is 1 for distance <= 4 cm away
            #print(dist_to_cup_r)
        
        # Want robot to be within table range
        x,y,z = tip_coord
        max_z = 0.5
        if z<=-.1:
            print("Ending episode why is the robot arm this low")
            return -1, True
        pos_tol=.2
        x_error=0 #unused
        y_error=0
        z_error=0
        if y < self.table_y_range[0]-pos_tol:
            y_error=self.table_y_range[0]-pos_tol-y
        if y > self.table_y_range[1]+pos_tol:
            y_error=y-self.table_y_range[1]-pos_tol
        if z >= max_z+pos_tol:
            z_error = z-max_z-pos_tol
        
        robot_near_table = 1 - min((x_error+y_error + z_error)*5,1) # if robot gets more than 20 cm out of bounds, this is just 0
        
        reward_factors = [dist_to_cup_r,
                          cups_near_goal_r,
                          cups_at_goal_r, 
                          robot_near_table,
                          robot_holding_cup_r #3 times this should be LESS than the cups at goal reward, since this is no longer achieved if cup is at goal
                          ]
        reward_weights = [1,4,4,1,1]

        for i in range(len(reward_factors)):
            total_reward += reward_factors[i] * reward_weights[i]
        total_reward = total_reward / sum(reward_weights) # Normalize
        if debug:
            print(reward_factors)
            print("Reward is ",total_reward)
            print("--------------------\n")
        return total_reward, False
    
    #========================= RESETTING ENVIRONMENT ==========================#
    def despawn_cups(self):
        for cup in self.cup_names:
            self.delete_model(cup)
            rospy.sleep(.01)
        self.cup_names=[]
    
    def spawn_cup(self,name,position,orientation=None):
        pose = Pose()
        (pose.position.x,pose.position.y,pose.position.z)=position
        if orientation:
            (roll,pitch,yaw)=orientation
            stuff=Rotation.from_euler('xyz',(roll,pitch,yaw)).as_quat()
            (pose.orientation.x,pose.orientation.y,pose.orientation.z,pose.orientation.w)=stuff
        self.spawn_model(name, self.cup_xml, "", pose, "world")
        rospy.sleep(.01)
        self.cup_names.append(name)
        
    def set_cup_ranges(self,x_range,y_range):
        self.cup_ranges=x_range,y_range
    
    def reset_cups(self,prob_stand=1,prob_flip=0,prob_other=0): # input the probabilities that the cups are spawned normal, flipped, or fallen
        #print("RESETTing")
        # generate random new cup positions
        self.despawn_cups()
        cup_names = ["cup1", "cup2", "cup3"]
        cup_positions = []
        cup_rotations=[]
        for i in range(len(cup_names)):
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
            while self.cup_has_collision(x,y,cup_positions+[[0.,0.,0.]],tol=.08 if not yikes else .165): # 0,0,0 for the robot arm
                x = np.random.uniform(self.cup_ranges[0][0],self.cup_ranges[0][1])
                y = np.random.uniform(self.cup_ranges[1][0],self.cup_ranges[1][1])
            cup_positions.append((x,y,.065 if not yikes else .1))
            cup_rotations.append(rot)
        self.move_cups(cup_positions,cup_rotations)
    
    # To make sure random cup locations do not intersect
    def cup_has_collision(self,x,y,cup_positions,tol=.08):
        return any([np.linalg.norm((pos[0]-x,pos[1]-y)) <= tol for pos in cup_positions]) # any are within tolerance
    
    def move_cups(self, positions,orientations=None):
        #print("moving cups")
        # move cups to the randomized positions
        cup_names = ["cup1", "cup2", "cup3"]
        for i in range(len(cup_names)):
            (x,y,z)=positions[i]
            self.spawn_cup(cup_names[i],(x,y,z),orientations[i] if orientations is not None else None)
        #for zs in [[-1.]*3,[p[2] for p in positions]]:
        #    for i in range(len(cup_names)):
        #        (x,y,z)=positions[i]
                
                #model_state_msg = ModelState()
                #pose_msg = Pose()
                #point_msg = Point()
                #rot_msg=Quaternion()#default no rotation
                #if orientations:
                #  (roll,pitch,yaw)=orientations[i]
                #  stuff=Rotation.from_euler('xyz',(roll,pitch,yaw)).as_quat()
                #  (rot_msg.x,rot_msg.y,rot_msg.z,rot_msg.w)=stuff
                #(x,y,_)=positions[i]
                #point_msg.x = x
                #point_msg.y = y
                #point_msg.z = zs[i]
                #pose_msg.position = point_msg
                #pose_msg.orientation = rot_msg
                #self.spawn_cup(cup_names[i],pose_msg)
                #model_state_msg.model_name = cup_names[i]
                
                #model_state_msg.pose = pose_msg
                
                #model_state_msg.reference_frame = "world"
                
                #self.pub.publish(model_state_msg)
                #rospy.sleep(.01)
    
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
    
    def robot_holding_cup_position(self,min_grab_pos=0.209, min_grab_eff=1.05e-1): 
        joint_positions,_,joint_efforts = self.get_joint_state()
        finger_pos = joint_positions[6]
        finger_eff = joint_efforts[6]
        return finger_pos >= min_grab_pos and finger_eff >= min_grab_eff

    def get_tip_coord(self):
        return self.get_cartesian_points()[-1][-1]
    

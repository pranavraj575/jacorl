from gazebo_msgs.msg import LinkStates, ModelState, ModelStates
from geometry_msgs.msg import Pose, Point, Quaternion
from gazebo_msgs.srv import DeleteModel, SpawnModel
from jaco_gym.envs.robot_env import JacoEnv
import numpy as np
import rospy
import math
from scipy.spatial.transform import Rotation
import os

class JacoGazeboEnv(JacoEnv):
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
        
        #basic gazebo thingies
        rospy.wait_for_service("gazebo/delete_model")
        rospy.wait_for_service("gazebo/spawn_sdf_model")
        self.delete_model = rospy.ServiceProxy("gazebo/delete_model", DeleteModel)
        self.spawn_model = rospy.ServiceProxy("gazebo/spawn_sdf_model", SpawnModel)

        self.models_spawned=set() # set of object names
        
        
        self.pub_topic = '/gazebo/set_model_state'
        self.pub = rospy.Publisher(self.pub_topic, ModelState, queue_size=1)

        # Subscribe to object data to obtain cup locations
        def _call_model_data(data):
            self.object_data=data
            
        self.sub_topic="/gazebo/model_states"
        self.sub=rospy.Subscriber(self.sub_topic,ModelStates,_call_model_data)

    #========================== GYM FUNCTIONS ============================#
    def close(self):
        super().close()
        self.despawn_all()
    
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
    #========================= OBJECT EDITING ==========================#
    def object_exists(self,name):
        return name in self.get_object_dict()
        
    def spawn_model_from_xml(self,xml_text,name,position,orientation=None):
        if not name in self.models_spawned:
            while not self.object_exists(name):
                pose=Pose()
                (pose.position.x,pose.position.y,pose.position.z)=position
                if orientation is not None:
                    (roll,pitch,yaw)=orientation
                    stuff=Rotation.from_euler('xyz',(roll,pitch,yaw)).as_quat()
                    (pose.orientation.x,pose.orientation.y,pose.orientation.z,pose.orientation.w)=stuff
                self.spawn_model(name, xml_text, "", pose, "world")
                rospy.sleep(.3)
            self.models_spawned.add(name)
        else:
            print("WARNING: attempt to spawn model with existing name: "+name)
    
    def spawn_model_from_name(self,model_name,name,position,orientation=None):
        xml=None
        print("spawning:"+model_name)
        for path in os.environ.get('GAZEBO_MODEL_PATH', 'Nonesuch').split(':'):
            model_path=os.path.join(path,model_name,'model.sdf')
            if os.path.exists(model_path):
                with open(model_path,'r') as f:
                    xml=f.read()
        if xml is None:
            print("ERROR: model "+model_name+" not found")
            print("looked in directories of $GAZEBO_MODEL_PATH="+os.environ.get('GAZEBO_MODEL_PATH', '(not defined)'))
        else:
            self.spawn_model_from_xml(xml,name,position,orientation)
    
    def despawn_model(self,model_name):
        if model_name in self.models_spawned:
            self.delete_model(model_name)
            self.models_spawned.remove(model_name)
            rospy.sleep(.3)
        else:
            print("WARNING: attempt to despawn non-existant model")
    
    def despawn_all(self):
        for nm in list(self.models_spawned):
            self.despawn_model(nm)

    def move_object(self,name,position,orientation=None):
        pose=Pose()
        pose.position.x,pose.position.y,pose.position.z=position
        rot=Quaternion()
        if orientation:
            r,p,y=orientation
            rot.x,rot.y,rot.z,rot.w = Rotation.from_euler('xyz', (r, p, y)).as_quat()
        pose.orientation=rot
        model_state_msg = ModelState()
        model_state_msg.model_name = name
        model_state_msg.pose = pose
        model_state_msg.reference_frame = "world"
        self.pub.publish(model_state_msg)
        rospy.sleep(.1)

    

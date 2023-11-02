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

        self.models_spawned=set() # set of object names
        
        
        self.pub_topic = '/gazebo/set_model_state'
        self.pub = rospy.Publisher(self.pub_topic, ModelState, queue_size=1)

        # Subscribe to object data to obtain cup locations
        def _call_model_data(data):
            self.object_data=data
            
        self.sub_topic="/gazebo/model_states"
        self.sub=rospy.Subscriber(self.sub_topic,ModelStates,_call_model_data)

    
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

    
    #========================= RESETTING ENVIRONMENT ==========================#
    def spawn_model_from_xml(self,xml_text,name,position,orientation=None):
        if not name in self.models_spawned:
            pose=Pose()
            (pose.position.x,pose.position.y,pose.position.z)=position
            if orientation:
                (roll,pitch,yaw)=orientation
                stuff=Rotation.from_euler('xyz',(roll,pitch,yaw)).as_quat()
                (pose.orientation.x,pose.orientation.y,pose.orientation.z,pose.orientation.w)=stuff
            self.spawn_model(name, xml_text, "", pose, "world")
            self.models_spawned.add(name)
            rospy.sleep(.1)
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
            rospy.sleep(.1)
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

    

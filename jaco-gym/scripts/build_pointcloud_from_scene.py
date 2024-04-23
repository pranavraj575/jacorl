import gym
import jaco_gym
import numpy as np
import rospy
import os
import cv2
import utilities
from kortex_api.autogen.client_stubs.VisionConfigClientRpc import VisionConfigClient
from kortex_api.autogen.client_stubs.DeviceManagerClientRpc import DeviceManagerClient
from kortex_api.autogen.messages import DeviceConfig_pb2, Session_pb2, DeviceManager_pb2, VisionConfig_pb2
# Open3d needs to be version 0.14.1 for dumb reasons
import matplotlib.pyplot as plt
from PIL import Image
from scipy import interpolate


rospy.init_node("kinova_client", log_level=rospy.INFO)
env = gym.make("BasicJacoEnv-v0")


point1 = [-57.7845459, 5.85499763, -147.46310425,
         64.75018311, -26.95962524, -25.09310913]
point2 = [62.59965897, 31.63041687, -126.87625122,
         135.9168396, 61.16304016, 72.33145142]
point3 = [22.35601997, 48.6255455, -71.41812134,
         158.41389465, 94.48800659, 42.17469406]
point4 = [-26.58825684, 17.45651817, -110.35652161, -0.81622314, -52.89157104, 95.18431854]
point5 = [-26.51452637, 11.92891693, -102.33920288,   -0.76641846,  -66.44482422, 94.48609924]

point6 = [-12.83605957, -1.46264648, -138.17466736, -1.64755249, -43.38787842, 168.66375732]
point7 = [-6.97528076, 26.91834831, -99.08062744, -1.44592285, -54.0843811, 174.120224]
point8 = [32.82913208, 35.78237915, -83.26422119, -1.61169434, -61.02319336, -146.10087585]

point9 = [-3.95230103, 66.38656616, -10.08413696, -178.85952759, 103.99039459, 174.94233704]

point10 = [-2.77941895, 74.09378815, -6.506073, 178.42019653, 112.80392456, -1.2583313 ]
point11 = [-8.7003479, 26.85749435, -101.63357544, -178.95671082, 35.63476181, -9.00485229]

point12 = [5.39791536, 50.7124939, -58.33151245, 159.54020691, 84.15299988, 9.78591061]
point13 = [5.55740452, 17.332407, -121.49882507, 140.7688446, 28.2960434, 0.17746267]


points = [point12, point13]


# Create save image dir
imgs_dir = "pointcloud_samples/"
os.makedirs(imgs_dir, exist_ok=True)


# Take a 3 channel image and create a new image interpolated to the new shape, intended
# for use when color and depth image do not match size
def resize_image(img, new_shape):
   X = np.linspace(0, img.shape[1], img.shape[1])
   Y = np.linspace(0, img.shape[0], img.shape[0])
   x, y = np.meshgrid(X, Y)

   # Define interpolate function for each of the 3 channels
   f_red = interpolate.interp2d(X, Y, img[:, :, 0])
   f_green = interpolate.interp2d(X, Y, img[:, :, 1])
   f_blue = interpolate.interp2d(X, Y, img[:, :, 2])

   Xnew = np.linspace(0, img.shape[1], new_shape[1])
   Ynew = np.linspace(0, img.shape[0], new_shape[0])

   # Use interpolation function to define sampled image, and stack channels to create
   # 3 channel image
   new_red = f_red(Xnew, Ynew)
   new_green = f_green(Xnew, Ynew)
   new_blue = f_blue(Xnew, Ynew)
   new_image = np.dstack([new_blue, new_green, new_red])

   return new_image


def save_images(id, color, depth):
   cv2.imwrite(imgs_dir + str(id) + ".jpeg", color)

   depth_img = Image.fromarray(depth)
   depth_img.save(imgs_dir + str(id) + "_depth.png")


def example_vision_get_device_id(device_manager):
   vision_device_id = 0


   # Getting all device routing information (from DeviceManagerClient service)
   all_devices_info = device_manager.ReadAllDevices()


   vision_handles = [
       hd for hd in all_devices_info.device_handle if hd.device_type == DeviceConfig_pb2.VISION]
   if len(vision_handles) == 0:
       print("Error: there is no vision device registered in the devices info")
   elif len(vision_handles) > 1:
       print("Error: there are more than one vision device registered in the devices info")
   else:
       handle = vision_handles[0]
       vision_device_id = handle.device_identifier
       print("Vision module found, device Id: {0}".format(vision_device_id))


   return vision_device_id

def create_rotation_matrix(x_angle, y_angle, z_angle):
    # Define rotation matrix for each of the 3 dimensions, and overall rotation matrix is the result of matrix multiplying the 3
    #print("Angles")
    #print(x_angle, y_angle, z_angle)
    x_rotation_matrix = np.array([[1, 0, 0], [0, np.cos(x_angle), -np.sin(x_angle)], [0, np.sin(x_angle), np.cos(x_angle)]])
    y_rotation_matrix = np.array([[np.cos(y_angle), 0, np.sin(y_angle)], [0, 1, 0], [-np.sin(y_angle), 0, np.cos(y_angle)]])
    z_rotation_matrix = np.array([[np.cos(z_angle), -np.sin(z_angle), 0], [np.sin(z_angle), np.cos(z_angle), 0], [0, 0, 1]])
    rotation_matrix = np.matmul(np.matmul(z_rotation_matrix, y_rotation_matrix), x_rotation_matrix)

    return rotation_matrix

def extract_homography(extrinsic):
    np_extrinsic = np.array([[extrinsic.rotation.row1.column1, extrinsic.rotation.row1.column2, extrinsic.rotation.row1.column3],
                             [extrinsic.rotation.row2.column1, extrinsic.rotation.row2.column2, extrinsic.rotation.row2.column3],
                             [extrinsic.rotation.row3.column1, extrinsic.rotation.row3.column2, extrinsic.rotation.row3.column3]])
    
    # depth to color sensor
    np_translation = np.array([extrinsic.translation.t_x, extrinsic.translation.t_y, 0]) * 1000

    # depth to color sensor
    homography = np.array([[np_extrinsic[0, 0], np_extrinsic[0, 1], np_extrinsic[0, 2], np_translation[0]],
                           [np_extrinsic[1, 0], np_extrinsic[1, 1], np_extrinsic[1, 2], np_translation[1]],
                           [np_extrinsic[2, 0], np_extrinsic[2, 1], np_extrinsic[2, 2], np_translation[2]]])
    
    return homography

# Create a Tcp connection to the kinova arm in order to grab intrinsic and extrinsic
# matrix, formated for o3d
def get_intrinsic(args):
   # Create connection to the device and get the router
   with utilities.DeviceConnection.createTcpConnection(args) as router:
       device_manager = DeviceManagerClient(router)
       vision_config = VisionConfigClient(router)
       vision_device_id = example_vision_get_device_id(device_manager)

       if vision_device_id != 0:
           sensor_id = VisionConfig_pb2.SensorIdentifier()
           sensor_id.sensor = VisionConfig_pb2.SENSOR_DEPTH
           intrinsics = vision_config.GetIntrinsicParameters(
               sensor_id, vision_device_id)

           # Parse intrinsics into format that o3d can deal with
           W, H = 480, 270
           Fx, Fy = intrinsics.focal_length_x, intrinsics.focal_length_y
           Cx, Cy = intrinsics.principal_point_x, intrinsics.principal_point_y
           
           np_intrinsic = np.array([[Fx, 0, Cx], [0, Fy, Cy], [0, 0, 1]])
           
           extrinsics = vision_config.GetExtrinsicParameters(vision_device_id)
           homography = extract_homography(extrinsics)


   return np_intrinsic, homography

# Use the forward kinematic algorithms to determine rotation and translation of camera and return 4 by 4 marix
# describing movement
def get_extrinsic():
   pos,_,_=env.get_joint_state()
   x_angle, y_angle, z_angle, rot_matrix_2, cam_pos = env.get_camera_rotation_and_position()
   translation_matrix = env.get_tip_coord() * 1000
   print("Translation Matrix: " )
   print(translation_matrix)
   print("\n")

   o3d_extrinsic = np.array([[rot_matrix_2[0, 0], rot_matrix_2[0, 1], rot_matrix_2[0, 2], translation_matrix[0]],
                             [rot_matrix_2[1, 0], rot_matrix_2[1, 1], rot_matrix_2[1, 2], translation_matrix[1]],
                             [rot_matrix_2[2, 0], rot_matrix_2[2, 1], rot_matrix_2[2, 2], translation_matrix[2]],
                             [0, 0, 0, 1]])

   return o3d_extrinsic, rot_matrix_2, translation_matrix


# Take a color and depth image and return and o3d PointCloud object containing the
# pointcloud given by the two images
def image_to_point_clouds(color, depth, args):
   np_intrinsic, homography = get_intrinsic(args)
   extrinsic_o3d, rot_matrix, trans_matrix = get_extrinsic()

   x_data, y_data, z_data = [], [], []
   for y in range(len(depth)):
       for x in range(len(depth[0])):
          if depth[y, x] == 0 or depth[y, x] > (338 * 4):
             continue
          
          x_3d = (x - np_intrinsic[0, 2]) * depth[y, x] / np_intrinsic[0, 0]
          y_3d = (y - np_intrinsic[1, 2]) * depth[y, x] / np_intrinsic[1, 1]
          z_3d = depth[y, x]

          new_coord = np.array([[x_3d, y_3d, z_3d]]).T
          world_coord = np.matmul(rot_matrix, new_coord) + np.array([trans_matrix]).T

          color_coord = np.matmul(homography, np.array([[x_3d, y_3d, z_3d, 1]]).T) / np.matmul(homography, np.array([[x_3d, y_3d, z_3d, 1]]).T)[-1]
          # print("Color coordinate", color_coord)

          x_data.append(world_coord[0])
          y_data.append(world_coord[1])
          z_data.append(world_coord[2])

          # #print(world_coord)

   print("Average y data")
   print(sum(y_data) / len(y_data))
   print("\n")

   #print(np.median(depth))
   #print(np.sum(depth > (338 * 4)))
   world_coords = [x_data, y_data, z_data]
   fig = plt.figure()
   ax = plt.axes(projection='3d')
   ax.scatter3D(x_data, y_data, z_data, cmap='Greens')

   ax.set_xlim3d(np.median(x_data) - 500, np.median(x_data) + 500)
   ax.set_ylim3d(np.median(y_data) - 500, np.median(y_data) + 500)
   ax.set_zlim3d(np.median(z_data) - 500, np.median(z_data) + 500)
   plt.show()
   
   return


# Take RGB and depth photo at arm position and return both photos, where color photo
# is downsampled to be of equal size to the depth image, also depth image is in millimeters
def take_photo():
   color = env.get_image_numpy(mode='color', cam_type="attached")
   depth = env.get_image_numpy(mode='depth', cam_type='attached')

   color_downsampled = resize_image(color, depth.shape)
   return color_downsampled, depth

"""
def click_event(event, x, y, flags, params):
   if event == cv2.EVENT_LBUTTONDOWN: 
      #print(f"(X, Y) coordinate: ({x}, {y})")
"""

# Sequence manager that moves arm, takes photos and merges pointcloud
def run_sequence():
   args = utilities.parseConnectionArguments()
   all_world_coords = []

   for id, point in enumerate(points):
       env.move_arm(point,degrees=True)
       rospy.sleep(4)

       color, depth = take_photo()
       # cv2.imshow("Color image", color / 255)
       # cv2.setMouseCallback('Color image', click_event) 
       # cv2.waitKey(0)
       save_images(id, color, depth)

       image_to_point_clouds(color, depth, args)
   
   fig = plt.figure()
   ax = plt.axes(projection='3d')

   x_data, y_data, z_data = None, None, None
   for all_world_coord in all_world_coords:
      if x_data is None:
         x_data = all_world_coord[0]
         y_data = all_world_coord[1]
         z_data = all_world_coord[2]

      else:
         x_data += all_world_coord[0]
         y_data += all_world_coord[1]
         z_data += all_world_coord[2]

   #print(np.array(x_data))
   step = 20
   ax.scatter3D(np.array(x_data[::step]), np.array(y_data[::step]), np.array(z_data[::step]), cmap='Greens')

   xdifflow, xdiffhigh = np.median(x_data) - np.min(x_data), np.max(x_data) - np.median(x_data)
   ydifflow, ydiffhigh = np.median(y_data) - np.min(y_data), np.max(y_data) - np.median(y_data)
   zdifflow, zdiffhigh = np.median(z_data) - np.min(z_data), np.max(z_data) - np.median(z_data)

   diff = max(max(max(xdifflow, xdiffhigh), max(ydifflow, ydiffhigh)), max(zdifflow, zdiffhigh))
   print("Diff", diff)
   ax.set_xlim3d(np.median(x_data) - diff, np.median(x_data) + diff)
   ax.set_ylim3d(np.median(y_data) - diff, np.median(y_data) + diff)
   ax.set_zlim3d(np.median(z_data) - diff, np.median(z_data) + diff)
   plt.show()


if __name__ == "__main__":
   run_sequence()

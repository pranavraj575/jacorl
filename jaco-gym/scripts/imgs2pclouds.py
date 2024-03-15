import open3d as o3d
import numpy as np
from matplotlib.image import imread
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--color_path', type=str,
                    help='Path to the color image', required=True)
parser.add_argument('--depth_path', type=str,
                    help='Path to the depth image', required=True)
args = parser.parse_args()

depth_npy = imread(args.color_path)
color_npy = imread(args.depth_path)
if color_npy is None:
    print("Color image path incorrect, no image found")
if depth_npy is None:
    print("Depth image path incorrect, no image found")
if color_npy is None or depth_npy is None:
    exit(0)

color = o3d.geometry.Image(color_npy.astype(np.uint8))
depth = o3d.geometry.Image(depth_npy.astype(np.float32))
intrinsic = o3d.camera.PinholeCameraIntrinsic(
    640, 480, 653.68229, 651.855994, 311.753415, 232.400954)

rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
    color, depth)
pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
    rgbd, intrinsic)

# Uncomment this to visualize point cloud
# o3d.visualization.draw_geometries([pcd])
o3d.io.write_point_cloud(args.color_path.split(".")[0] + ".pcd", pcd)


from remote.robot_controll_client import robot_controller
from ensenso import Ensenso
import numpy as np
from scipy.spatial.transform import Rotation as R
import cv2
import subprocess
import time

    
en = Ensenso()
rgb, pcs = en.get_cam_obs()
extrinsic = en.extrinsic
height, width = pcs.shape[0], pcs.shape[1]
pcs = np.concatenate([pcs, np.ones((height, width, 1))], axis=-1)
pcs = pcs.reshape(-1, 4)
pcs = pcs @ extrinsic.T
pcs = pcs.reshape(-1, 3)
selected = np.logical_not(np.isnan(pcs).any(-1))
rgb = rgb.reshape(-1, 3)
pcs = pcs[selected]
rgb = rgb[selected]
import open3d as o3d
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pcs)
pcd.colors = o3d.utility.Vector3dVector(rgb / 255.)
o3d.io.write_point_cloud('debug.ply', pcd)

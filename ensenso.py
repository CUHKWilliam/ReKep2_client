from nxlib import NxLibCommand, NxLibException, NxLibItem
from nxlib.constants import *
import nxlib.api as api
import argparse
import open3d as o3d

from nxlib import NxLib
from nxlib import Camera
import numpy as np
import time
from scipy.spatial.transform import Rotation as R
import json

extrinsic = np.array([[ 0.37943252,  0.29933713, -0.87545888,  0.46888271],
       [-0.29118286,  0.93677002,  0.19409912, -0.08058141],
       [ 0.87820471,  0.1812711 ,  0.44260285,  0.28193391]])

class Ensenso():
    def __init__(self):
        self.serial_rgb = "4103078743"
        self.serial_d = "161080"
        self.camera_pose = np.eye(4)
        self.camera_pose[:3, :3] = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
        self.camera_intrinsics = {
            'width': 1000,
            'height': 750,
            'fx': 1502,
            'fy': 1502,
            'cx': 521,
            'cy': 240
        }
        self.verbose = False    
        self.open_camera()
        self.update_camera_params()
    
    def update_camera_params(self):
        # param = json.dumps(json.load(open("data_collection/fast_scan2.json", "r")))
        # NxLibItem()[ITM_CAMERAS][ITM_PARAMETERS] << param
        # camera_node = NxLibItem()[ITM_CAMERAS][self.serial_rgb]
        # camera_node << param
        param = json.dumps(json.load(open("d.json", "r")))
        camera_node = NxLibItem()[ITM_CAMERAS][self.serial_d]
        camera_node << param
    
    def close_camera(self):
        cmd = NxLibCommand(CMD_CLOSE)
        cmd.parameters()[ITM_CAMERAS] = self.serial_rgb
        cmd.execute()
        cmd.parameters()[ITM_CAMERAS] = self.serial_d
        cmd.execute()

    def filter_nans_mask(self,point_map):
        mask = ~np.isnan(point_map).any(axis = 1)
        return mask

    def filter_nans(self,point_map):
        mask = ~np.isnan(point_map).any(axis = 1)
        return point_map[~np.isnan(point_map).any(axis = 1)]
    
    def reshape_point_map(self,point_map):
        return point_map.reshape(
            (point_map.shape[0] * point_map.shape[1]), point_map.shape[2])
    
    def convert_to_open3d_color_point_cloud(self, point_map, color_map):
        point_map = self.reshape_point_map(point_map)
        color_map = self.reshape_point_map(color_map)
        # not filitering
        mask = self.filter_nans_mask(point_map)
        point_map = point_map[mask]
        color_map = color_map[mask][:,:3]/255.0
        # color_map = color_map[:,:3]/255.0
        o3d_point_cloud = o3d.geometry.PointCloud()
        o3d_point_cloud.points = o3d.utility.Vector3dVector(point_map)
        o3d_point_cloud.colors = o3d.utility.Vector3dVector(color_map)
        return o3d_point_cloud

    def convert_to_open3d_point_cloud(self, point_map):
        point_map = self.reshape_point_map(point_map)
        point_map = self.filter_nans(point_map)
        o3d_point_cloud = o3d.geometry.PointCloud()
        o3d_point_cloud.points = o3d.utility.Vector3dVector(point_map)
        return o3d_point_cloud
    
    def open_camera(self):
        try:
            # Waits for the cameras to be initialized
            api.initialize()
        except NxLibException as e:
            print("An NxLibException occured: Error Text: {}".format(e.get_error_text()))
        except:
            print("Something bad happenend, that has been out of our control.")
        cmd = NxLibCommand(CMD_OPEN)
        cmd.parameters()[ITM_CAMERAS][0] = self.serial_d
        cmd.parameters()[ITM_CAMERAS][1] = self.serial_rgb
        cmd.execute()
        # pass 
    
    def get_rgb_image(self):
        capture = NxLibCommand(CMD_CAPTURE)
        capture.parameters()[ITM_CAMERAS] = self.serial_rgb
        capture.execute()

        img = NxLibItem()[ITM_CAMERAS][self.serial_rgb][ITM_IMAGES][ITM_RAW].get_binary_data()
        return img
    
    def get_depth_image(self):
        capture = NxLibCommand(CMD_CAPTURE)
        capture.parameters()[ITM_CAMERAS] = self.serial_d
        capture.execute()
        print(1)

    def get_colored_pointcloud(self):
        try:
            capture = NxLibCommand(CMD_CAPTURE)
            # capture.parameters()[ITM_TIMEOUT] = 10000
            capture.execute()
            NxLibCommand(CMD_COMPUTE_DISPARITY_MAP).execute()
            NxLibCommand(CMD_COMPUTE_POINT_MAP).execute()

            capture = NxLibCommand(CMD_RENDER_POINT_MAP)
            capture.parameters()[ITM_CAMERA] = self.serial_rgb
            capture.parameters()[ITM_NEAR] = 100
            capture.parameters()[ITM_FAR] = 5000
            capture.parameters()[ITM_USE_OPEN_GL] = False
            capture.execute()

            colors = NxLibItem()[ITM_IMAGES][ITM_RENDER_POINT_MAP_TEXTURE].get_binary_data()
            points = NxLibItem()[ITM_IMAGES][ITM_RENDER_POINT_MAP].get_binary_data()
            if self.verbose:
                point_cloud = self.convert_to_open3d_color_point_cloud(points,colors)
                # apply the transformation
                rotm = R.from_rotvec(2.378177992351423686*np.array([0.08509144296158927,-0.428776052793116702,0.899394542392752738])).as_matrix()
                trans = np.array([-13.776148796081542969,-37.15863037109375,1290.7335205078125])
                transformation = np.eye(4)
                transformation[:3,:3] = rotm
                transformation[:3, 3] = trans
                point_cloud.transform(transformation)
                mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1000, origin=[0, 0, 0])
                o3d.visualization.draw_geometries([point_cloud, mesh_frame])
            points = points / 1000.
            return colors, points, False
        except NxLibException as e:
            print("An NxLibException occured: Error Text: {}".format(e.get_error_text()))
            return np.zeros([1280,1024,3]), np.zeros([1280,1024,3]), True
        except:
            print("Something bad happenend, that has been out of our control.")
            return np.zeros([1280,1024,3]), np.zeros([1280,1024,3]), True

    
    def point_to_rgbd(self, pcd):
        '''
        This function converts the point cloud to rgbd image
        input: point cloud, camera pose, camera intrinsics
        output: rgbd image
        '''
        camera_intrinsics = self.camera_intrinsics
        pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
            camera_intrinsics['width'], camera_intrinsics['height'], camera_intrinsics['fx'],
            camera_intrinsics['fy'], camera_intrinsics['cx'], camera_intrinsics['cy']
        )
        # transfer the pointcloud to open3d.t.geometry.PointCloud
        device = o3d.core.Device("CPU:0")
        dtype = o3d.core.float32
        pcd_t = pcd
        pcd_t.point.positions = pcd.point.positions.to(dtype)/1000
        
        # Create a RGBD image by projecting the point cloud to the image plane
        width = camera_intrinsics['width']
        height = camera_intrinsics['height']
        pinhole_camera_intrinsic_t = o3d.core.Tensor(pinhole_camera_intrinsic.intrinsic_matrix, device=device, dtype=dtype)
        camera_pose = o3d.core.Tensor(self.camera_pose, device=device, dtype=dtype)
        rgbd = pcd_t.project_to_rgbd_image(width, height, pinhole_camera_intrinsic_t,camera_pose)
        print("done project point cloud to rgbd image")
        
        color_numpy = np.asarray(rgbd.color.to_legacy())
        depth_numpy = np.asarray(rgbd.depth.to_legacy())
        if self.verbose:
            mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
            o3d.visualization.draw_geometries([pcd_t.to_legacy(), mesh_frame])
            # visualize the rgbd image
            from PIL import Image
            import matplotlib.pyplot as plt
            plt.figure()
            plt.subplot(1, 2, 1)
            plt.imshow(color_numpy)
            plt.subplot(1, 2, 2)
            plt.imshow(depth_numpy)
            plt.show()
        return color_numpy, depth_numpy
    
    def convert_pcd_to_tensor(self, pcd):
        device = o3d.core.Device("CPU:0")
        dtype = o3d.core.float32
        pcd_t = o3d.t.geometry.PointCloud(device)
        pcd_t.point.positions = o3d.core.Tensor(np.asarray(pcd.points), dtype, device)
        pcd_t.point.colors = o3d.core.Tensor(np.asarray(pcd.colors), dtype, device)
        return pcd_t
    
    def get_rgbd_image(self):
        pcd = self.get_colored_pointcloud()
        pcd_t = self.convert_pcd_to_tensor(pcd)
        color_numpy, depth_numpy = self.point_to_rgbd(pcd_t)
        return color_numpy, depth_numpy

    def refine_recorded_data(self,colors,points):
        pcd = self.convert_to_open3d_color_point_cloud(points,colors)
        # apply the transformation
        rotm = R.from_rotvec(2.378177992351423686*np.array([0.08509144296158927,-0.428776052793116702,0.899394542392752738])).as_matrix()
        trans = np.array([-13.776148796081542969,-37.15863037109375,1290.7335205078125])
        transformation = np.eye(4)
        transformation[:3,:3] = rotm
        transformation[:3, 3] = trans
        pcd.transform(transformation)
        pcd_t = self.convert_pcd_to_tensor(pcd)
        color_numpy, depth_numpy = self.point_to_rgbd(pcd_t)
        return color_numpy, depth_numpy
    
    def get_cam_obs(self,):
        for _ in range(5):
            data = self.get_colored_pointcloud()
        rgb = self.get_rgb_image()[:, :, :3]
        # rgb = data[0][:, :, :3]
        pcs = data[1]
        return rgb, pcs

    def close(self,):
        self.close_camera()

def get_colored_pointcloud(pipeline, results):
    colors, points, failed = pipeline.get_colored_pointcloud()
    results.append(colors)
    results.append(points)
    results.append(failed)


if __name__ == "__main__":
    ensenso = Ensenso()
    for _ in range(5):
        init_time = time.time()
        pcd = ensenso.get_colored_pointcloud()
        print("pcd time: ", time.time()-init_time)
        # pcd_t = ensenso.convert_pcd_to_tensor(pcd)
        # rgbd = ensenso.point_to_rgbd(pcd_t)
    import open3d as o3d
    pc = o3d.geometry.PointCloud()
    rgb = pcd[0].reshape(-1, 4)[:, :3]
    pcd = pcd[1].reshape(-1, 3)
    mask = np.logical_not(np.isnan(pcd).any(1))
    pcd = pcd[mask]
    rgb = rgb[mask]
    idx = np.random.randint(0, rgb.shape[0], 100000)
    pcd = pcd[idx]
    rgb = rgb[idx]
    pc.points = o3d.utility.Vector3dVector(pcd)
    pc.colors = o3d.utility.Vector3dVector(rgb / 256.)
    o3d.io.write_point_cloud('debug.ply', pc)
    import ipdb;ipdb.set_trace()
    ensenso.close_camera()
    print("done")

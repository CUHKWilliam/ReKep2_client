import abc
import warnings

import glfw
from gym import error
from gym.utils import seeding
import numpy as np
from os import path
import gym
import cv2
import sys
sys.path.append("/data/wltang/omnigibson/datasets/ReKep2")

from segment import segment
from tqdm import tqdm

try:
    import mujoco_py
except ImportError as e:
    raise error.DependencyNotInstalled("{}. (HINT: you need to install mujoco_py, and also perform the setup instructions here: https://github.com/openai/mujoco-py/.)".format(e))


def _assert_task_is_set(func):
    def inner(*args, **kwargs):
        env = args[0]
        if not env._set_task_called:
            raise RuntimeError(
                'You must call env.set_task before using env.'
                + func.__name__
            )
        return func(*args, **kwargs)
    return inner


DEFAULT_SIZE = 500
import open3d as o3d

class MujocoEnv(gym.Env, abc.ABC):
    """
    This is a simplified version of the gym MujocoEnv class.

    Some differences are:
     - Do not automatically set the observation/action space.
    """

    # max_path_length = 500
    max_path_length = 500

    def __init__(self, model_path, frame_skip):
        if not path.exists(model_path):
            raise IOError("File %s does not exist" % model_path)

        self.frame_skip = frame_skip
        self.model = mujoco_py.load_model_from_path(model_path)
        self.sim = mujoco_py.MjSim(self.model)
        self.data = self.sim.data
        self.viewer = None
        self._viewers = {}

        self.metadata = {
            'render.modes': ['human'],
            'video.frames_per_second': int(np.round(1.0 / self.dt))
        }
        self.init_qpos = self.sim.data.qpos.ravel().copy()
        self.init_qvel = self.sim.data.qvel.ravel().copy()

        self._did_see_sim_exception = False

        self.np_random, _ = seeding.np_random(None)
        
        self.keypoint2obj_names = []
        self.keypoint2obj_pt_indices = []
        self.part_to_pts_dict_simulation = None
        self.pts_num = None
        self.part_to_pts_dict = []
        self.is_grasping = False
        self.moving_part_names = ["gripper"]
        self.reset_joint_pos =  self.get_env_state()[0].qpos
    
    def get_moving_part_names(self,):
        return self.moving_part_names
    
    def segment(self, obj_part, rekep_program_dir, camera, resolution):
        obs = self.get_obs(camera, resolution=resolution, body_invisible=True)
        rgb = obs['rgb']
        cv2.imwrite("rgb_obs.png", rgb[:, :, ::-1])
        mask = segment(image_path="rgb_obs.png", obj_description=obj_part, rekep_program_dir=rekep_program_dir)
        return mask

    def get_part_to_pts_dict(self, ):
        part_lists = self.part_lists
        part_to_pts_dict = self.part_to_pts_dict
        start = 0
        part_to_pts_dict_latest = {}
        for i, part in enumerate(part_lists):
            end =  start + self.pts_num[i]
            if "gripper" in part:
                part_to_pts_dict_latest[part] = self.get_endeff_pos()[None, :]
            else:
                objs = self.keypoint2obj_names[start: end]
                pt_indices = self.keypoint2obj_pt_indices[start: end]
                part_pts = []
                for i in range(len(objs)):
                    import ipdb;ipdb.set_trace()
                    obj, pt_index = objs[i], pt_indices[i]
                    obj_pts_barycentric = self.obj_2_pts_barycentric[obj]
                    mesh_faces = self.model.mesh_face
                    mesh_faceadrs = self.model.mesh_faceadr
                    mesh_faceadrs = np.append(mesh_faceadrs, -1)
                    mesh_vertices = self.model.mesh_vert
                    mesh_vertadrs = self.model.mesh_vertadr
                    mesh_vertadrs = np.append(mesh_vertadrs, -1)
                    obj_id = self.model.mesh_name2id(obj)
                    start, end = mesh_faceadrs[obj_id], mesh_faceadrs[obj_id + 1]
                    obj_mesh_faces = mesh_faces[start: end]
                    start, end = mesh_vertadrs[obj_id], mesh_vertadrs[obj_id + 1]
                    obj_mesh_vertices = mesh_vertices[start: end]
                    obj_tri_vertices = obj_mesh_vertices[obj_mesh_faces]
                    obj_pts = get_obj_pts_from_barycentric(obj_tri_vertices, obj_pts_barycentric)
                    part_pts = obj_pts[pt_index]
                    part_pts.append(part_pts)
                part_to_pts_dict_latest[part] = np.stack(part_pts, axis=0)
            start = end
        part_to_pts_dict.append(part_to_pts_dict_latest)
        self.part_to_pts_dict = part_to_pts_dict
        return part_to_pts_dict


    def seed(self, seed):
        assert seed is not None
        self.np_random, seed = seeding.np_random(seed)
        self.action_space.seed(seed)
        self.observation_space.seed(seed)
        self.goal_space.seed(seed)
        return [seed]

    @abc.abstractmethod
    def reset_model(self):
        """
        Reset the robot degrees of freedom (qpos and qvel).
        Implement this in each subclass.
        """
        pass

    def viewer_setup(self):
        """
        This method is called when the viewer is initialized and after every reset
        Optionally implement this method, if you need to tinker with camera position
        and so forth.
        """
        pass

    @_assert_task_is_set
    def reset(self):
        self._did_see_sim_exception = False
        self.sim.reset()
        ob = self.reset_model()
        if self.viewer is not None:
            self.viewer_setup()
        return ob

    def set_state(self, qpos, qvel):
        assert qpos.shape == (self.model.nq,) and qvel.shape == (self.model.nv,)
        old_state = self.sim.get_state()
        new_state = mujoco_py.MjSimState(old_state.time, qpos, qvel,
                                         old_state.act, old_state.udd_state)
        self.sim.set_state(new_state)
        self.sim.forward()

    @property
    def dt(self):
        return self.model.opt.timestep * self.frame_skip

    def do_simulation(self, ctrl, n_frames=None):
        self.max_path_length=500
        if getattr(self, 'curr_path_length', 0) > self.max_path_length:
            raise ValueError('Maximum path length allowed by the benchmark has been exceeded')
        if self._did_see_sim_exception:
            return

        if n_frames is None:
            n_frames = self.frame_skip
        self.sim.data.ctrl[:] = ctrl

        for _ in range(n_frames):
            try:
                self.sim.step()
            except mujoco_py.MujocoException as err:
                warnings.warn(str(err), category=RuntimeWarning)
                self._did_see_sim_exception = True

    def render2(self, offscreen=True, camera_name="corner3", resolution=(640, 480), depth=False, segmentation=False, body_invisible=False):
        assert_string = ("camera_name should be one of ",
                "corner3, corner, corner2, topview, gripperPOV, behindGripper")
        assert camera_name in {"corner3", "corner", "corner2", 
            "topview", "gripperPOV", "behindGripper"}, assert_string
        self.model.site_rgba[:, -1] = 0.
        if body_invisible:
            invisible_ids = []
            rgba = self.model.geom_rgba.copy() 
            for name in self.model.body_names:
                if  'screen' in name or 'pad' in name or 'claw' in name or 'hand' in name or 'right' in name or 'screen' in name or 'head' in name:
                    invisible_ids.append(self.model.body_name2id(name))
            for id in invisible_ids:
                self.model.geom_rgba[id] = np.array([0,0,0,0])
        if segmentation: 
            data = self.sim.render(
                *resolution,
                mode='offscreen',
                camera_name=camera_name,
                depth=False,
                segmentation=True,
            )
            if body_invisible:
                for id in invisible_ids:
                    self.model.geom_rgba[id] = rgba[id]
            return data
        elif not offscreen:
            self._get_viewer('human').render()
        else:
            results = [*self.sim.render(
                *resolution,
                mode='offscreen',
                camera_name=camera_name,
                depth=depth,
            )]
            if depth:
                d = results[1]
                # Get the distances to the near and far clipping planes.
                extent = self.model.stat.extent
                near = self.model.vis.map.znear * extent
                far = self.model.vis.map.zfar * extent
                # Convert from [0 1] to depth in meters, see links below:
                # http://stackoverflow.com/a/6657284/1461210
                # https://www.khronos.org/opengl/wiki/Depth_Buffer_Precision
                results[1] = -near / (1 - d * (1 - near / far))
            if body_invisible:
                for id in invisible_ids:
                    self.model.geom_rgba[id] = rgba[id]
            return results

    def sample_with_binear(self, fmap, kp):
        max_x, max_y = fmap.shape[1]-1, fmap.shape[0]-1
        x0, y0 = int(kp[0]), int(kp[1])
        x1, y1 = min(x0+1, max_x - 1), min(y0+1, max_y - 1)
        x, y = kp[0]-x0, kp[1]-y0
        fmap_x0y0 = fmap[y0, x0]
        fmap_x1y0 = fmap[y0, x1]
        fmap_x0y1 = fmap[y1, x0]
        fmap_x1y1 = fmap[y1, x1]
        fmap_y0 = fmap_x0y0 * (1-x) + fmap_x1y0 * x
        fmap_y1 = fmap_x0y1 * (1-x) + fmap_x1y1 * x
        feature = fmap_y0 * (1-y) + fmap_y1 * y
        return feature
    
    def to_3d(self, points, depths, cmat):
        points = points.reshape(-1, 2)
        depths = np.array([[self.sample_with_binear(depths, kp)] for kp in points])
        points = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1) * depths
        points = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
        cmat = np.concatenate([cmat, np.array([[0, 0, 0, 1]])], axis=0)
        points = np.dot(np.linalg.inv(cmat), points.T).T
        points = points[:, :3]
        return points

    def get_cmat(self, camera_name, resolution):
        id = self.sim.model.camera_name2id(camera_name)
        fov = self.sim.model.cam_fovy[id]
        pos = self.sim.data.cam_xpos[id]
        rot = self.sim.data.cam_xmat[id].reshape(3, 3).T
        width, height = resolution
        # Translation matrix (4x4).
        translation = np.eye(4)
        translation[0:3, 3] = -pos
        # Rotation matrix (4x4).
        rotation = np.eye(4)
        rotation[0:3, 0:3] = rot
        # Focal transformation matrix (3x4).
        focal_scaling = (1./np.tan(np.deg2rad(fov)/2)) * height / 2.0 # focal length
        focal = np.diag([-focal_scaling, -focal_scaling, 1.0, 0])[0:3, :]
        # Image matrix (3x3).
        image = np.eye(3)
        image[0, 2] = (width - 1) / 2.0
        image[1, 2] = (height - 1) / 2.0

        return image @ focal @ rotation @ translation


    def get_obs(self, camera_name, resolution, body_invisible=False):
        import ipdb;ipdb.set_trace()
        rgb, depth = self.render2(depth=True,offscreen=True,camera_name=camera_name, segmentation=False, resolution=resolution, body_invisible=body_invisible)
        segmentation = np.stack(self.render2(depth=False,offscreen=True,camera_name="corner", segmentation=True, resolution=resolution, body_invisible=body_invisible), axis=0)
        pts_2d = np.stack(np.meshgrid(np.arange(rgb.shape[1]), np.arange(rgb.shape[0])), axis=-1).reshape(-1, 2)
        points = self.to_3d(pts_2d, depth, self.get_cmat(camera_name, resolution))
        # import open3d as o3d
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(points)
        # o3d.io.write_point_cloud('debug.ply', pcd)
        # import ipdb;ipdb.set_trace()
        H, W = rgb.shape[0], rgb.shape[1]
        points = points.reshape((H, W, 3))
        obs = {
            "rgb": rgb, 
            "depth": depth, 
            "seg": segmentation,
            "points": points,
        }
        self.last_obs = obs
        return obs

    def close(self):
        if self.viewer is not None:
            glfw.destroy_window(self.viewer.window)
            self.viewer = None

    def _get_viewer(self, mode):
        self.viewer = self._viewers.get(mode)
        if self.viewer is None:
            if mode == 'human':
                self.viewer = mujoco_py.MjViewer(self.sim)
            self.viewer_setup()
            self._viewers[mode] = self.viewer
        self.viewer_setup()
        return self.viewer

    def get_body_com(self, body_name):
        return self.data.get_body_xpos(body_name)

    def interpolate_convex_areas_random(self, triangles, num_points=100):
        """
        Randomly interpolates a specified number of points over the convex areas of multiple triangles.
        
        Parameters:
        triangles : np.ndarray
            An array of shape (N, 3, 3), where N is the number of triangles, each with 3 vertices in 3D.
        num_points : int
            Number of points to sample randomly within each triangle.
        
        Returns:
        interpolated_points_3d : list of np.ndarray
            A list of arrays, where each array contains `num_points` interpolated 3D points within a triangle.
        interpolated_points_barycentric : list of np.ndarray
            A list of arrays, where each array contains `num_points` barycentric coordinates of points within a triangle.
        """
        interpolated_points_3d = []
        interpolated_points_barycentric = []

        # Iterate over each triangle
        for triangle in triangles:
            pt1, pt2, pt3 = triangle[0], triangle[1], triangle[2]
            
            # Lists to store 3D and barycentric coordinates for the current triangle
            points_3d_for_triangle = []
            points_barycentric_for_triangle = []
            
            # Generate random barycentric coordinates
            r1 = np.sqrt(np.random.rand(num_points, 1))  # To ensure uniform distribution within the triangle
            r2 = np.random.rand(num_points, 1)
            
            w1 = 1 - r1
            w2 = r1 * (1 - r2)
            w3 = r1 * r2
            
            # Calculate the interpolated points
            points_3d = w1 * pt1 + w2 * pt2 + w3 * pt3
            
            # Append results
            interpolated_points_3d.append(points_3d)
            interpolated_points_barycentric.append(np.hstack([w1, w2, w3]))
        
        return interpolated_points_3d, interpolated_points_barycentric
    def register_keypoints(self, part_lists, rekep_program_dir, camera, resolution):
        part_to_pts_dict_latest = {}
        MAX_PTS = 80
        keypoints = []
        self.part_lists = part_lists
        pts_num = []
        obs = self.get_obs(camera, resolution=resolution, body_invisible=True)
        all_pts = obs['points']
        for obj_part in part_lists:
            if "gripper" in obj_part:
                pts = self.get_endeff_pos()[None, :]
            else:
                mask = self.segment(obj_part, rekep_program_dir, camera, resolution)
                pts_2d = np.stack(np.where(mask > 0), axis=-1)[..., ::-1]
                if len(pts_2d) > MAX_PTS:
                    # import fpsample
                    # fps_samples_idx = fpsample.fps_sampling(pts_2d, MAX_PTS)
                    samples_idx = np.random.choice(np.arange(len(pts_2d)), MAX_PTS)
                    pts_2d = pts_2d[samples_idx]
                pts = all_pts[pts_2d[:, 1], pts_2d[:, 0], :]
            part_to_pts_dict_latest[obj_part] = pts
            keypoints.append(pts)
            pts_num.append(len(pts))
        self.pts_num = pts_num
        self.part_to_pts_dict.append(part_to_pts_dict_latest)
        keypoints = np.concatenate(keypoints, axis=0)
        self.keypoints = keypoints
        exclude_names = ['wall', 'floor', 'ceiling', 'fetch', 'robot']
        print("registering points:")
        obj_2_pts = {}
        obj_2_pts_barycentric = {}
        import ipdb;ipdb.set_trace()
        mesh_to_geom_idx_dict = {}
        mesh_faces = self.model.mesh_face
        mesh_faceadrs = self.model.mesh_faceadr
        mesh_faceadrs = np.append(mesh_faceadrs, -1)
        mesh_vertices = self.model.mesh_vert
        mesh_vertadrs = self.model.mesh_vertadr
        mesh_vertadrs = np.append(mesh_vertadrs, -1)
        for i, obj in enumerate(self.model.mesh_names):
            obj_id = self.model.mesh_name2id(obj)
            start, end = mesh_faceadrs[obj_id], mesh_faceadrs[obj_id + 1]
            obj_mesh_faces = mesh_faces[start: end]
            start, end = mesh_vertadrs[obj_id], mesh_vertadrs[obj_id + 1]
            obj_mesh_vertices = mesh_vertices[start: end]
         
            mesh_to_geom_idx_dict[obj] = dist_mesh_geoms.argmin(0)
            obj_tri_vertices = obj_mesh_vertices[obj_mesh_faces]
            pts, pts_barycentric = self.interpolate_convex_areas_random(obj_tri_vertices, 10)
            pts, pts_barycentric = np.concatenate(pts, axis=0), np.concatenate(pts_barycentric, axis=0)
            obj_2_pts[obj] = pts
            obj_2_pts_barycentric[obj] = pts_barycentric
            
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts)
            o3d.io.write_point_cloud('debug.ply', pcd)
        self.mesh_to_geom_idx_dict = mesh_to_geom_idx_dict
        self.obj_2_pts = obj_2_pts
        self.obj_2_pts_barycentric = obj_2_pts_barycentric
        for idx, keypoint in tqdm(enumerate(keypoints)):
            closest_distance = np.inf
            for i, obj in enumerate(self.model.mesh_names):
                points = self.obj_2_pts[obj]
                if any([name in obj.lower() for name in exclude_names]):
                    continue
                dists = np.linalg.norm(points - keypoint, axis=1)
                point = points[np.argmin(dists)]
                point_idx = np.argmin(dists)
                distance = np.linalg.norm(point - keypoint)
                if distance < closest_distance:
                    closest_distance = distance
                    closest_point_idx = point_idx
                    closest_obj = obj
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points)
            self.keypoint2obj_names.append(closest_obj)
            self.keypoint2obj_pt_indices.append(closest_point_idx)
            

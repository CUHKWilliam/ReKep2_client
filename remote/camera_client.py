import paramiko
from scp import SCPClient
import pickle
import numpy as np
from realsense import *
import threading
from threading import Lock
from communication import Client
from ensenso import Ensenso
import time
import torch
import marshal
import os
from cotracker.utils.visualizer import Visualizer
import copy

torch.hub.set_dir("./cached_model")
class RemoteCameraClient(Client):
    def __init__(self,
            server_ip,
            server_name,
            server_port,
            server_pw,
            lock_file_path,
            data_file_path,
            local_lock_file_path,
            local_data_file_path,
        ):
        super().__init__(
            server_ip,
            server_name,
            server_port,
            server_pw,
            lock_file_path,
            data_file_path,
            local_lock_file_path,
            local_data_file_path,
        )
        self.lock = Lock()
        self.camera = RealSense()
        self.video = []
        self.is_recording = False
        self.cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker3_online").cuda()
        self.keypoints_dict = {}    
        self.visibility_dict = {}
        self.updated_keypoints_dict = {}
        self.new_obj_part = False
        self.thread_track_exiting = 0
        self.video_frames = []
        self.is_first_step = True
        thread_track_keypoints = threading.Thread(target=self._thread_track_keypoints)
        self.thread_track_keypoints = thread_track_keypoints
        thread_track_keypoints.start()
        self.scale_factor = 4
        self.palette = np.random.randint(low=(0, 0, 0), high=(255, 255, 255), size=(10000, 3))
        self.cnt = 0
        
    
    def increase_brightness(self, img, value=70):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        lim = 255 - value
        v[v > lim] = 255
        v[v <= lim] += value

        final_hsv = cv2.merge((h, s, v))
        img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
        return img
    
    def get_extrinsic(self,):
        cam_extrinsic = self.camera.extrinsic
        self.send(cam_extrinsic)        
        
    def get_intrinsic(self,):
        self.send(self.camera.intrinsic)
    
    def get_resolution(self,):
        self.send((self.camera.width, self.camera.height))
        
    def get_depth_scale(self,):
        self.send(self.camera.depth_scale)
        
    def get_depth_max(self,):
        self.send(self.camera.depth_max)

    def get_cam_obs(self,):
        rgb, depth, pcs = self.camera.get_cam_obs(return_depth=True)
        rgb = rgb.astype(np.uint8)
        data = {
            "rgb": rgb,
            "points": pcs,
            "depth": depth
        }
        self.send(data)
        
    def get_cam_rgb(self,):
        rgb = self.camera.get_cam_rgb()
        rgb = rgb.astype(np.uint8)
        data = {
            "rgb": rgb,
        }
        self.send(data)
    
    
    def get_occlusion_func(self,):
        occlusion_func = self.camera.get_occlusion_func()
        data = marshal.dumps(occlusion_func.__code__)
        self.send(data)
        
    def record_video_thread(self, ):
        self.is_recording = True
        ind = 0
        while self.is_recording:
            rgb, _ = self.camera.capture(once=False)
            self.video.append(rgb)
            print("capture RGB {}".format(ind))
            ind += 1

    
    def end_record_video(self,):
        self.is_recording = False
        time.sleep(0.5)
    
    def pop_video_frames(self, ):
        return self.video
        self.video = []
        

    def handle_data(self, data):
        if isinstance(data, str):
            type = data.split(":")[1]
            if type == "extrinsic":
                self.get_extrinsic()
            elif type == "record_video":
                self.record_video()
            elif type == "end_record_video":
                self.end_record_video()
            elif type == "pop_video_frame":
                self.pop_video_frames()
            elif type == "get_occlusion_func":
                self.get_occlusion_func()
            elif type == "depth_scale":
                self.get_depth_scale()
            elif type == "depth_max":
                self.get_depth_max()
            elif type == "intrinsic":
                self.get_intrinsic()
            elif type == "resolution":
                self.get_resolution()
            elif type == "cam_obs":
                self.get_cam_obs()
            elif type == "cam_rgb":
                self.get_cam_rgb()
                
        elif isinstance(data, dict):
            request = data['request']
            if request == "get keypoints":
                self.get_keypoints()
            elif request == "register keypoints":
                obj_part = data['obj_part']
                mask = data['mask']
                self.register_keypoints(obj_part, mask)
            elif request == "register keypoints pose changed":
                obj_part = data['obj_part']
                self.register_keypoints_pose_changed(obj_part)

    def _track_step(self, window_frames, is_first_step, queries):
        video_chunk = (
            torch.tensor(
                np.stack(window_frames[-self.cotracker.step * 2 :])
            ).cuda()
            .float()
            .permute(0, 3, 1, 2)[None]
        )  # (1, T, 3, H, W)
        try:
            return self.cotracker(
            video_chunk,
            is_first_step=is_first_step,
            queries=queries
            )
        except:
            import ipdb;ipdb.set_trace()
            
    def _thread_track_keypoints(self, ):
        obj_name_to_color = {}
        rgb_previous = None
        cnt = 0
        previous_key_num = len(self.keypoints_dict.keys())
        while True:
            with self.lock:
                if len(self.updated_keypoints_dict.keys()) > 0:
                    self.video_frames = []
                    self.is_first_step = True
                    for key in self.updated_keypoints_dict.keys():
                        self.keypoints_dict[key] = self.updated_keypoints_dict[key]
                    self.updated_keypoints_dict = {}
                    continue
            keypoints_dict = self.keypoints_dict
            num_points = {}
            queries = []
            for key in keypoints_dict.keys():
                queries.append(keypoints_dict[key])
                num_points[key] = len(keypoints_dict[key])
            if len(queries) == 0:
                self.video_frames = []
                continue
            queries = np.concatenate(queries, axis=0)
            queries = np.concatenate([np.zeros((len(queries), 1)), queries], axis=-1)
            queries = torch.from_numpy(queries).float().cuda()
            
            if len(self.video_frames) % self.cotracker.step == 0 and len(self.video_frames) // self.cotracker.step > 1:
                pred_tracks, pred_visibility = self._track_step(
                    self.video_frames,
                    self.is_first_step,
                    queries=queries[None]
                )
                self.is_first_step = False
                if pred_tracks is None:
                    continue
                rgb = self.video_frames[-1].copy()
                track_rgb_debug = rgb.copy()
                keypoints_dict2 = {}
                cnt2 = 0
                for key in keypoints_dict.keys():
                    if key not in num_points.keys():
                        continue
                    num_pt = num_points[key]
                    keypoints_dict2[key] = pred_tracks[0, -1, cnt2: cnt2 + num_pt, :].detach().cpu().numpy()
                    self.visibility_dict[key] = pred_visibility[0, -1, cnt2: cnt2 + num_pt].detach().cpu().numpy()
                    cnt2 += num_pt
                    ## TODO: for debug
                    xys = pred_tracks[0, -1, :, :]
                    visibility = pred_visibility[0, -1, :]
                    xys = xys[visibility]
                    if key not in obj_name_to_color.keys():
                        color = self.palette[self.cnt]
                        self.cnt += 1
                        obj_name_to_color[key] = color
                    else:
                        color = obj_name_to_color[key]
                    for xy in xys:
                        track_rgb_debug = cv2.circle(track_rgb_debug.copy(), xy.detach().cpu().numpy().astype(np.int64), 1, color=color.tolist(), thickness=-1)
                cv2.imwrite("debug_track_point.png", track_rgb_debug)
                ## TODO: end for debug   
                self.keypoints_dict = keypoints_dict2
                if len(self.video_frames) == self.cotracker.step * 5:
                    self.video_frames = self.video_frames[self.cotracker.step * 4:]
            
            rgb = self.camera.fast_capture()
            rgb = cv2.resize(rgb.copy(), (rgb.shape[1] // self.scale_factor, rgb.shape[0] // self.scale_factor))
            if cnt < 10:
                cnt += 1
                continue
            if rgb_previous is None:
                rgb_previous = rgb.copy()
            else:
                if np.mean(np.abs(rgb_previous.astype(np.float32) - rgb.astype(np.float32))) > 1:
                    rgb_previous = rgb.copy()
                    self.video_frames.append(rgb)
                else:
                    continue
            if len(self.video_frames) < self.cotracker.step * 2:
                continue
            
            if self.thread_track_exiting == 1:
                break
            
    def get_keypoints(self,):
        keypoints_dict = self.keypoints_dict
        keypoints_dict_scaled = {}
        for key in keypoints_dict.keys():
            _, points = self.camera.get_cam_obs()
            pts_2d = keypoints_dict[key]
            pts_2d = pts_2d * self.scale_factor
            visibility = self.visibility_dict[key]
            pts_3d = points[pts_2d[:, 1].astype(np.int64), pts_2d[:, 0].astype(np.int64)]
            pts_3d = pts_3d[visibility]
            pts_3d = pts_3d[np.logical_not(np.isnan(pts_3d).any(-1))]
            keypoints_dict_scaled[key] = (pts_2d, pts_3d)
        self.send(
            {
                "keypoints_dict": keypoints_dict_scaled,
            }
        )
    
    def register_keypoints(self, obj_part, mask):
        keypoints = np.stack(np.where(mask > 0), axis=-1)[:, ::-1]
        keypoints = keypoints // self.scale_factor
        with self.lock:
            self.updated_keypoints_dict[obj_part] = keypoints
        self.send({"data": "OK"})
    
        
    def start_track_part_points(self, part_dict):
        pass
    def end_track_part_points(self,):
        pass
    def get_part_to_pts_dict(self,):
        pass
    

class RemoteCamerasClient(RemoteCameraClient):
    def __init__(self,
            server_ip,
            server_name,
            server_pw,
            lock_file_path,
            data_file_path,
            local_lock_file_path,
            local_data_file_path,
            cameras
        ):
        super().__init__(
            server_ip,
            server_name,
            server_pw,
            lock_file_path,
            data_file_path,
            local_lock_file_path,
            local_data_file_path,
        )
        self.cameras = cameras
        
    def get_cam_obs(self,):
        rgbs = []
        depths = []
        pcss = []
        for camera in self.cameras:
            rgb, depth, pcs = camera.get_cam_obs(return_depth=True)
            rgb = rgb.astype(np.uint8)
            rgbs.append(rgb)
            depths.append(depth)
            pcss.append(pcs)
        rgb, depth, pcs = self.camera_fusion(rgbs, depths, pcss)
        data = {
            "rgb": rgb,
            "points": pcs,
            "depth": depth
        }
        self.send(data)
    
    def camera_fusion(self, rgbs, depths, pcss):
        pass
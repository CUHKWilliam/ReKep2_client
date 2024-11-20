import paramiko
from scp import SCPClient
import pickle
import numpy as np
from realsense import *
import threading
from communication import Client
from ensenso import Ensenso
import time
import torch
import marshal

class RemoteCameraClient(Client):
    def __init__(self,
            server_ip,
            server_name,
            server_pw,
            lock_file_path,
            data_file_path,
            local_lock_file_path,
            local_data_file_path,
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
        self.camera = RealSense()
        self.video = []
        self.is_recording = False
        # self.camera = Ensenso()
        self.cotracker = None
    
    # def get_cam_obs(self,):
    #     image, depth = self.camera.capture(once=True)
    #     pcd = self.camera.get_pcd(image, depth / 1000.)
    #     pcs = np.asarray(pcd.points).reshape(image.shape[0], image.shape[1], -1)
    #     data = {
    #         "rgb":image,
    #         "depth": depth,
    #         "points": pcs,
    #     }
    #     self.send(data)
    
    
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
    
    def get_occlusion_func(self,):
        occlusion_func = self.camera.get_occlusion_func()
        data = marshal.dumps(occlusion_func.__code__)
        self.send(data)
        
    def record_video_thread(self, ):
        self.is_recording = True
        ind = 0
        self.camera.pipe.start(self.camera.cfg)
        while self.is_recording:
            rgb, _ = self.camera.capture(once=False)
            self.video.append(rgb)
            time.sleep(0.1)
            print("capture RGB {}".format(ind))
            ind += 1
        self.camera.pipe.stop()

    
    def end_record_video(self,):
        self.is_recording = False
        time.sleep(0.5)
    
    def pop_video_frames(self, ):
        return self.video
        self.video = []
        
    
    def record_video(self, ):
        threading.Thread(target=self.record_video_thread).start()
        
    def handle_data(self, data):
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
        else:
            self.get_cam_obs()

    def start_track_part_points(self, part_dict):
        pass
    def end_track_part_points(self,):
        pass
    def get_part_to_pts_dict(self,):
        pass
    
    def track_points(self, video, points):
        if self.cotracker is None:
            cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker3_online").cuda()

        cotracker(video_chunk=video, is_first_step=True, grid_size=grid_size)  

        # Process the video
        for ind in range(0, video.shape[1] - cotracker.step, cotracker.step):
            pred_tracks, pred_visibility = cotracker(
                video_chunk=video[:, ind : ind + cotracker.step * 2]
            )  # B T N 2,  B T N 1

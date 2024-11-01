import paramiko
from scp import SCPClient
import pickle
import numpy as np
from realsense import *
import threading
from communication import Client
from ensenso import Ensenso

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
        # self.camera = RealSense()
        self.camera = Ensenso()
    
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

    def get_cam_obs(self,):
        rgb, pcs = self.camera.get_cam_obs()
        rgb = rgb.astype(np.uint8)
        data = {
            "rgb": rgb,
            "points": pcs,
        }
        self.send(data)
        
    def handle_data(self, data):
        return self.get_cam_obs()

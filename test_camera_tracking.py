from remote.camera_client import RemoteCameraClient
import time
import numpy as np

camera_client = RemoteCameraClient(
    server_ip = "128.32.164.115",
    server_name = "vision",
    server_pw = "MSCViSion-2021!",
    lock_file_path = "/data/wltang/omnigibson/datasets/tmp/lock2.txt",
    data_file_path = "/data/wltang/omnigibson/datasets/tmp/communication2.pkl",
    local_data_file_path="tmp2.pkl",
    local_lock_file_path = "tmp2.txt",
)
camera_client.record_video()
time.sleep(5)
camera_client.end_record_video()
time.sleep(0.5)
video = camera_client.pop_video_frames()
video = np.stack(video, axis=0)
points = np.array([1, 1])
camera_client.track_points(video, points)
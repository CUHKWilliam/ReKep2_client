from remote.camera_client import *

camera_client = RemoteCameraClient(
    server_ip = "128.32.164.115",
    server_name = "vision",
    server_pw = "MSCViSion-2021!",
    lock_file_path = "/data/wltang/omnigibson/datasets/tmp/lock2.txt",
    data_file_path = "/data/wltang/omnigibson/datasets/tmp/communication2.pkl",
    local_data_file_path="tmp2.pkl",
    local_lock_file_path = "tmp2.txt",
)
camera_client.start()
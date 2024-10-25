from remote.robot_controll_client import *

robot_client = RemoteRobotClient(
    server_ip = "128.32.164.115",
    server_name = "vision",
    server_pw = "MSCViSion-2021!",
    lock_file_path = "/data/wltang/omnigibson/datasets/tmp/lock.txt",
    data_file_path = "/data/wltang/omnigibson/datasets/tmp/communication.pkl",
    local_data_file_path="tmp.pkl",
    local_lock_file_path = "tmp.txt",
)
robot_client.start()
    
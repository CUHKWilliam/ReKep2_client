import paramiko
from scp import SCPClient
import pickle
import numpy as np
from realsense import *
import threading


class Client:
    def __init__(self,
            server_ip,
            server_name,
            server_pw,
            lock_file_path,
            data_file_path,           
            local_lock_file_path,
            local_data_file_path,  
        ):
            self.server_ip = server_ip
            self.server_name = server_name
            self.server_pw = server_pw
            self.lock_file_path = lock_file_path
            self.data_file_path = data_file_path
            self.local_lock_file_path = local_lock_file_path
            self.local_data_file_path = local_data_file_path
            with open(self.local_lock_file_path, "w") as f:
                f.write("0")
            client = paramiko.SSHClient()
            client.load_system_host_keys()
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            client.connect(self.server_ip, 7347, self.server_name, self.server_pw)
            self.ssh = client
            self.scp = SCPClient(self.ssh.get_transport())
            
    def check_recv(self,):
        scp = self.scp
        while True:
            scp.get(self.lock_file_path, self.local_lock_file_path)
            time.sleep(0.1)
            with open(self.local_lock_file_path, 'r') as f:
                content = f.read().strip()
            if content == "1":
                break
        scp.get(self.data_file_path, self.local_data_file_path)
        with open(self.local_data_file_path, 'rb') as f:
            action = pickle.load(f)
        with open(self.local_lock_file_path, "w") as f:
            f.write("0")
        scp.put(self.local_lock_file_path, self.lock_file_path)
        print("receive:", action)
        return action

    def _thread(self,):
        while True:
            print("waiting for remote...")
            message = self.check_recv()
            with open(self.local_data_file_path, "rb") as f:
                recv = pickle.load(f)
            print("recv request:{}".format(recv))
            self.handle_data(recv)
    
    def send(self, data):
        with open(self.local_data_file_path, 'wb') as f:
            pickle.dump(data, f)
        self.scp.put(self.local_data_file_path, self.data_file_path)        
        with open(self.local_lock_file_path, "w") as f:
            f.write("1")
        self.scp.put(self.local_lock_file_path, self.lock_file_path)
        print("send:", data)
        self.check_read()
    
    def check_read(self,):
        while True:
            self.scp.get(self.lock_file_path, self.local_lock_file_path)
            time.sleep(0.1)
            with open(self.local_lock_file_path, "r") as f:
                lock = f.read().strip()
            if lock == "0":
                break
            
    def start(self,):
        t1 = threading.Thread(target=self._thread())
        t1.start()
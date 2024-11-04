import socket
import numpy as np
import struct
import time
from scipy.spatial.transform import Rotation as R
import cv2
from leap_hand import LeapNode
from scipy.spatial.transform import Rotation as R

DEBUG = False
DEBUG2 = False

class robot_controller:
    def __init__(self):
        self.UDP_IP_IN = (
            "192.168.1.200"  # Ubuntu IP, should be the same as Matlab shows
        )
        self.UDP_PORT_IN = (
            57831  # Ubuntu receive port, should be the same as Matlab shows
        )
        self.UDP_IP_OUT = (
            "192.168.1.100"  # Target PC IP, should be the same as Matlab shows
        )
        self.UDP_PORT_OUT = 3826  # Robot 1 receive Port
        self.gripper_port = 3828  # Robot 1 receive Port

        # self.s_in = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # self.s_in.bind((self.UDP_IP_IN, self.UDP_PORT_IN))
        # Receive TCP position (3*), TCP Rotation Matrix (9*), TCP Velcoity (6*), Force Torque (6*)
        self.unpacker = struct.Struct("12d 6d 6d")

        self.robot_pose, self.robot_vel, self.TCP_wrench = None, None, None

        ## TODO:
        # self.gripper_move()
        self.gripper_state = "open"
        if not DEBUG:
            self.gripper = LeapNode()
        
        ## TODO: set init pos
        if not DEBUG:
            robot_pos, robot_ori, _, _ = self.get_current_pose()
            robot_ori = R.from_matrix(robot_ori).as_euler("ZYX")
            robot_ori[0] = -np.pi / 2.
            self.move_to_point([robot_pos[0], robot_pos[1], robot_pos[2],  robot_ori[0], robot_ori[1], robot_ori[2]])
        
        
    def receive(self):
        self.s_in = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.s_in.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.s_in.bind((self.UDP_IP_IN, self.UDP_PORT_IN))
        data, _ = self.s_in.recvfrom(1024)
        unpacked_data = np.array(self.unpacker.unpack(data))
        self.robot_pose, self.robot_vel, self.TCP_wrench = (
            unpacked_data[0:12],
            unpacked_data[12:18],
            unpacked_data[18:24]
        )
        self.s_in.close()
        

    def send(self, udp_cmd):
        '''
        UDP command 1~6 TCP desired Position Rotation
        UDP desired vel 7~12 
        UDP Kp 13~18
        UDP Kd 19~24
        UDP Mass 25~27
        UDP Interial 28~30
        '''
        self.s_out = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        udp_cmd = udp_cmd.astype("d").tostring()
        self.s_out.sendto(udp_cmd, (self.UDP_IP_OUT, self.UDP_PORT_OUT))
        self.s_out.close()
    
    def get_current_pose(self):
        self.receive()
        robot_pos = self.robot_pose[0:3]
        robot_ori = self.robot_pose[3:12].reshape(3, 3).T
        robot_vel = self.robot_vel
        contact_force = self.TCP_wrench[0:6]
        return robot_pos, robot_ori, robot_vel, contact_force
    
    ## TODO: for gripper
    # def gripper_move(self):
    #     if DEBUG:
    #         return
    #     one = np.array(1)
    #     zero = np.array(0)
    #     self.s_out = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    #     self.s_out.sendto(one.astype("d").tobytes(), (self.UDP_IP_OUT, self.gripper_port))
    #     time.sleep(0.5)
    #     self.s_out.sendto(zero.astype("d").tobytes(), (self.UDP_IP_OUT, self.gripper_port))
    #     time.sleep(0.5)

    ## TODO: for leap hand
    def gripper_move(self):
        if DEBUG:
            return
        if self.gripper_state == "open":
            self.gripper.close_gripper()
            self.gripper_state = "close"
        else:
            self.gripper.open_gripper()
            self.gripper_state = "open"

    def move_to_point(self, waypoint, compliant=False, wait=10):
        if DEBUG:
            return
        if DEBUG2:
            wait = 0.2
        if compliant:
            Mass = np.array([1,1,1])   # to determine
            Inertia = 1*np.array([2, 2, 0.1])   # to determine
            Kp = np.array([0,0,0,0,0,0])
            Kd = np.array([70,70,70,20,20,10])
            TCP_d_pos = waypoint[:3]
            TCP_d_euler = waypoint[3:]
            TCP_d_vel = np.zeros(6)
        else:
            Mass = np.array([2,2,2])   # to determine
            Inertia = 1*np.array([2, 2, 2])   # to determine
            Kp = np.array([600,600,600,200,200,200])
            Kd = np.array([300,300,300,250,250,250])
            TCP_d_pos = waypoint[:3]
            TCP_d_euler = waypoint[3:]
            TCP_d_vel = np.zeros(6)
        # send the command to robot until the robot reaches the waypoint
        dis = 100
        dis_ori = 100
        init_time = time.time()
        desired_ori = R.from_euler("ZYX", TCP_d_euler).as_matrix()
        while ((dis > 0.005 or dis_ori > 1/180*np.pi) and time.time()-init_time<wait):
            self.receive()
            dis = np.linalg.norm(self.robot_pose[0:3]-waypoint[:3])
            robot_ori = self.robot_pose[3:12].reshape(3,3).T
            dis_ori = np.arccos((np.trace(robot_ori@desired_ori.T)-1)/2)
            UDP_cmd = np.hstack([TCP_d_pos, TCP_d_euler, TCP_d_vel, Kp, Kd, Mass, Inertia])
            self.send(UDP_cmd)
            # print(dis_ori)

    def move_to_point_step(self, waypoint, compliant=False, wait=10):
        if compliant:
            Mass = np.array([1,1,1])   # to determine
            Inertia = 1*np.array([2, 2, 0.1])   # to determine
            Kp = np.array([0,0,0,0,0,0])
            Kd = np.array([70,70,70,20,20,10])
            TCP_d_pos = waypoint[:3]
            TCP_d_euler = waypoint[3:]
            TCP_d_vel = np.zeros(6)
        else:
            Mass = np.array([2,2,2])   # to determine
            Inertia = 1*np.array([2, 2, 2])   # to determine
            Kp = np.array([600,600,600,200,200,200])
            Kd = np.array([300,300,300,250,250,250])
            TCP_d_pos = waypoint[:3]
            TCP_d_euler = waypoint[3:]
            TCP_d_vel = np.zeros(6)
        # send the command to robot until the robot reaches the waypoint
        dis = 100
        dis_ori = 100
        init_time = time.time()
        desired_ori = R.from_euler("ZYX", TCP_d_euler).as_matrix()
        # while ((dis > 0.005 or dis_ori > 1/180*np.pi) and time.time()-init_time<wait):
        if True:
            self.receive()
            dis = np.linalg.norm(self.robot_pose[0:3]-waypoint[:3])
            robot_ori = self.robot_pose[3:12].reshape(3,3).T
            dis_ori = np.arccos((np.trace(robot_ori@desired_ori.T)-1)/2)
            UDP_cmd = np.hstack([TCP_d_pos, TCP_d_euler, TCP_d_vel, Kp, Kd, Mass, Inertia])
            self.send(UDP_cmd)
            # print(dis_ori)
        if dis <= 0.005 and dis_ori <= 1/180*np.pi:
            reached = True
        else:
            reached = False
        return reached

    def manual_control(self,):
        while True:
            robot_pos, robot_ori, robot_vel, contact_force = self.get_current_pose()
            robot_ori = R.from_matrix(robot_ori).as_euler("ZYX") # + np.array([np.pi / 6., 0., 0])
            cv2.namedWindow('RealSense', cv2.WINDOW_NORMAL)
            cv2.imshow('RealSense', np.zeros((50, 50, 3)))
            ch = cv2.waitKey(25)

            if ch == 27:
                cv2.destroyAllWindows()
                flag_stop = True
                break
            elif ch & 0xFF == ord('a'):
                robot_pos += np.array([0., -0.02, 0])
            elif ch & 0xFF == ord('d'):
                robot_pos += np.array([0., 0.02, 0])
            elif ch & 0xFF == ord('w'):
                robot_pos += np.array([0.02, 0., 0])
            elif ch & 0xFF == ord('s'):
                robot_pos += np.array([-0.02, 0., 0])
            elif ch & 0xFF == ord('q'):
                robot_pos += np.array([0., 0, 0.02])
            elif ch & 0xFF == ord('e'):
                robot_pos += np.array([0., 0, -0.02])
            elif ch & 0xFF == ord('j'):
                robot_ori += np.array([-0.1, 0, 0.])
            elif ch & 0xFF == ord('l'):
                robot_ori += np.array([0.1, 0, 0.])

            elif ch & 0xFF == ord('i'):
                robot_ori += np.array([0., 0.1, 0.])
            elif ch & 0xFF == ord('k'):
                robot_ori += np.array([0., -0.1, 0.])

            elif ch & 0xFF == ord('c'):
                self.gripper_move()
            elif ch & 0xFF == ord('v'):
                self.gripper_init()
            elif ch & 0xFF == ord('p'):
                robot_pos2, robot_ori2, robot_vel2, contact_force2 = self.get_current_pose()
                print("pos:", robot_pos2)
                print("ori:", robot_ori2)


            self.move_to_point([robot_pos[0], robot_pos[1], robot_pos[2],  robot_ori[0], robot_ori[1], robot_ori[2]])

    def manual_control_collect(self,):
        while True:
            robot_pos, robot_ori, robot_vel, contact_force = self.get_current_pose()
            robot_ori = R.from_matrix(robot_ori).as_euler("ZYX") # + np.array([np.pi / 6., 0., 0])
            cv2.namedWindow('RealSense', cv2.WINDOW_NORMAL)
            cv2.imshow('RealSense', np.zeros((50, 50, 3)))
            ch = cv2.waitKey(25)

            if ch == 27:
                cv2.destroyAllWindows()
                flag_stop = True
                break
            elif ch & 0xFF == ord('a'):
                robot_pos += np.array([0., -0.02, 0])
            elif ch & 0xFF == ord('d'):
                robot_pos += np.array([0., 0.02, 0])
            elif ch & 0xFF == ord('w'):
                robot_pos += np.array([0.02, 0., 0])
            elif ch & 0xFF == ord('s'):
                robot_pos += np.array([-0.02, 0., 0])
            elif ch & 0xFF == ord('q'):
                robot_pos += np.array([0., 0, 0.02])
            elif ch & 0xFF == ord('e'):
                robot_pos += np.array([0., 0, -0.02])
            elif ch & 0xFF == ord('j'):
                robot_ori += np.array([-0.1, 0, 0.])
            elif ch & 0xFF == ord('l'):
                robot_ori += np.array([0.1, 0, 0.])

            elif ch & 0xFF == ord('i'):
                robot_ori += np.array([0., 0.1, 0.])
            elif ch & 0xFF == ord('k'):
                robot_ori += np.array([0., -0.1, 0.])

            elif ch & 0xFF == ord('c'):
                self.gripper_move()
            elif ch & 0xFF == ord('v'):
                self.gripper_init()
            
            self.move_to_point([robot_pos[0], robot_pos[1], robot_pos[2],  robot_ori[0], robot_ori[1], robot_ori[2]])


if __name__ == "__main__": 
    rc = robot_controller()
    robot_pos, robot_ori, robot_vel, contact_force = rc.get_current_pose()
    print("pos:", robot_pos)
    print("ori:", robot_ori)
    robot_ori = R.from_matrix(robot_ori).as_euler("ZYX")
    rc.move_to_point([robot_pos[0]-0.1, robot_pos[1], robot_pos[2],  robot_ori[0], robot_ori[1], robot_ori[2]])
    import ipdb;ipdb.set_trace()
    rc.move_to_point()
    rc.manual_control()
    

    # poses = [
    #     np.array([0.49662015, 0.05465096, 0.13408024]),
    #     np.array([ 0.57101431,  0.08677494, -0.07367619]),
    #     np.array([ 0.56192786,  0.08658801, -0.094215  ]),
    #     np.array([ 0.55282547,  0.02881749, -0.01467278]),
    #     np.array([ 0.55059586, -0.06365659,  0.02162342]),
    #     np.array([ 0.53992546, -0.12296812,  0.03076995]),
    #     np.array([ 0.52773134, -0.11548457, -0.05926101]),

    # ]
    # oris = [
    #         np.array([[ 9.98401034e-01, -9.88997628e-04,  5.65190001e-02],
    #         [-1.77801970e-03, -9.99901647e-01,  1.39117173e-02],
    #         [ 5.64996826e-02, -1.39899648e-02, -9.98304596e-01]]),
            
    #         np.array([[ 0.92929528, -0.36210852,  0.07271658],
    #         [-0.35999866, -0.93206269, -0.04074431],
    #         [ 0.08253027,  0.01168562, -0.99652005]]),

    #         np.array([[ 0.92933298, -0.36224152,  0.07156319],
    #         [-0.35998981, -0.93198107, -0.04264532],
    #         [ 0.08214344,  0.01386968, -0.996524  ]]),


    #         np.array([[ 0.92489905, -0.37318174,  0.07278142],
    #         [-0.37032438, -0.92755865, -0.04994804],
    #         [ 0.08614873,  0.01924417, -0.99609641]]),

    #         np.array([[ 0.9170897,  -0.3909405,   0.07817926],
    #         [-0.38795069, -0.92026952, -0.05097318],
    #         [ 0.09187347,  0.01641728, -0.99563534]]),

    #         np.array([[ 0.91354143, -0.39886735,  0.07966743],
    #         [-0.39546218, -0.91680944, -0.05540862],
    #         [ 0.09514054,  0.01911261, -0.99528035]]),

    #         np.array([[ 0.81582333, -0.53989084, -0.20724426],
    #         [-0.52960076, -0.8414405,   0.10724234],
    #         [-0.23228287,  0.02226591, -0.97239339]])

    # ]



    # poses = [
    #     np.array([0.54318629, 0.12508113, 0.08021187]),
    #     np.array([ 0.59682921,  0.18373348, -0.03659622]),
    #     np.array([ 0.59326692,  0.18274197, -0.09206459]),
    #     np.array([ 0.59441573,  0.18252358, -0.00963819]),
    #     np.array([ 0.61016674,  0.16667077, -0.00801435]),
    #     np.array([0.60819014, 0.1142489,  0.06973206]),
    #     np.array([0.59447413, 0.03241191, 0.12640597]),

    # ]
    # oris = [
    #         np.array([[ 0.99963101, -0.02640951 , 0.00635528],
    #             [-0.02647797, -0.99958953,  0.01094055],
    #             [ 0.00606374, -0.01110479, -0.99991995]]),
            
    #         np.array([[ 0.38215703, -0.92393529 , 0.01730832],
    #             [-0.92383851, -0.38242627, -0.01650902],
    #             [ 0.02187242, -0.00968106 ,-0.9997139 ]]),

    #         np.array([[ 0.38362348, -0.92311405,  0.02633374],
    #             [-0.92323015, -0.38403563, -0.01275629],
    #             [ 0.0218886 , -0.0194185 , -0.99957181]]),


    #         np.array([[ 0.3800034 , -0.92434067 , 0.0345217 ],
    #             [-0.92466911 ,-0.38058436, -0.0119403 ],
    #             [ 0.02417532 ,-0.0273838 , -0.99933262]]),

    #         np.array([[ 0.27098024 ,-0.96244957 ,-0.01614131],
    #             [-0.95073095, -0.27022909 , 0.15194372],
    #             [-0.15060002 ,-0.0258277,  -0.98825734]]),

    #         np.array([[ 0.1577751 , -0.98493761, -0.07074553],
    #             [-0.82157001 ,-0.17067681,  0.5439597 ],
    #             [-0.54784098 ,-0.02770089 ,-0.83612375]]),

    #         np.array([[ 0.06554768, -0.9843911,  -0.16333298],
    #             [-0.31880671, -0.17576562,  0.93138002],
    #             [-0.94555052 ,-0.00897815 ,-0.32535151]])

    # ]

    
    
    # for i in range(len(poses)):
    #     robot_pos = poses[i]
    #     ori = oris[i]
    #     robot_ori = R.from_matrix(ori).as_euler("ZYX") 
    #     rc.move_to_point([robot_pos[0], robot_pos[1], robot_pos[2],  robot_ori[0], robot_ori[1], robot_ori[2]])
    #     import ipdb;ipdb.set_trace()
    # rc.gripper_move()
    # robot_ori = R.from_matrix(robot_ori).as_euler("ZYX") # + np.array([np.pi / 6., 0., 0])
    # rc.move_to_point([robot_pos[0], robot_pos[1] + 0.2, robot_pos[2],  robot_ori[0], robot_ori[1], robot_ori[2]])


import socket
import threading
import pickle
import paramiko
from scp import SCPClient
from communication import Client

class RemoteRobotClient(Client):
    def __init__(self,
                 server_ip,
                 server_name,
                 server_pw,
                 lock_file_path,
                 data_file_path,
                 local_lock_file_path,
                 local_data_file_path,
                 ) -> None:
        super().__init__(
            server_ip,
            server_name,
            server_pw,
            lock_file_path,
            data_file_path,
            local_lock_file_path,
            local_data_file_path,
        )
        self.robot_controller = robot_controller()
        ## TODO:
        self.APPROACH0 =  np.array([0, 0, 1]).astype(np.float32)
        self.BINORMAL0 = np.array([-1, 0, 0]).astype(np.float32)
    
    def get_init_approach(self,):
        return self.APPROACH0
    
    def get_init_binormal(self,):
        return self.BINORMAL0
        
    def handle_data(self, data):
        try:
            action = np.fromstring(data, dtype=np.float32)
            action = np.concatenate([action[:3], R.from_quat(action[3:]).as_euler("ZYX")])
            self.robot_controller.move_to_point(action)
            return
        except:
            pass
        type = data.split(":")[0]
        content = ":".join(data.split(":")[1:])
        if content == "gripper open":
            self.robot_controller.gripper_move()
        elif content == "gripper close":
            self.robot_controller.gripper_move()
        else:
            if type == "action":
                if content == "close gripper" or content == "open gripper":
                    self.robot_controller.gripper_move()
            elif type == "query":
                if content == "ee_pose":
                    ee_pose = self.get_ee_pose()
                    ee_pose = ee_pose.astype(np.float32)
                    self.send(ee_pose.tostring())
                elif content == "approach0":
                    approach0 = self.get_init_approach()
                    approach0 = approach0.astype(np.float32)
                    self.send(approach0.tostring())
                elif content == "binormal0":
                    binormal0 = self.get_init_binormal()
                    binormal0 = binormal0.astype(np.float32)
                    self.send(binormal0.tostring())
               
                    
    
    def get_ee_pose(self,):
        if DEBUG:
            return np.array([0,0,0,0,0,0])
        robot_pos, robot_ori, _, _ = self.robot_controller.get_current_pose()
        robot_ori = R.from_matrix(robot_ori).as_euler("ZYX")
        robot_pose = np.concatenate([robot_pos, robot_ori], axis=0)
        return robot_pose
        
    def get_approach(self,):
        if DEBUG:
            return np.array([0, 1, 0])
        import ipdb;ipdb.set_trace()
        robot_ori = self.get_ee_pose()[3:]
        mat = R.from_euler("ZYX", robot_ori).as_matrix()
        approach = self.dot(self.APPROACH0, mat)
        return approach

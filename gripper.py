import numpy as np
import time
import socket

class Gripper:
    def __init__(self, udp_ip_in, udp_ip_out, udp_port_in, udp_port_out, gripper_port) -> None:
        self.udp_ip_in = udp_ip_in
        self.udp_ip_out = udp_ip_out
        self.udp_port_in = udp_port_in
        self.udp_port_out = udp_port_out
        self.gripper_port = gripper_port
        
    
    def gripper_move(self):
        one = np.array(1)
        zero = np.array(0)
        self.s_out = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.s_out.sendto(one.astype("d").tobytes(), (self.udp_ip_in, self.gripper_port))
        time.sleep(0.5)
        self.s_out.sendto(zero.astype("d").tobytes(), (self.udp_ip_out, self.gripper_port))
        time.sleep(0.5)
    
    
    def open_gripper(self,):
        self.gripper_move()
        
    def close_gripper(self,):
        self.gripper_move()

if __name__ == "__main__":
    gripper = Gripper(
        udp_ip_in=("192.168.1.200"),
        udp_ip_out=("192.168.1.100"),
        udp_port_in=(57831),
        udp_port_out=3826,
        gripper_port=3828,
    )
    gripper.gripper_move()
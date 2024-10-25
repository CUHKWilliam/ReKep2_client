from remote.robot_controll_client import robot_controller
from ensenso import Ensenso
import numpy as np
from scipy.spatial.transform import Rotation as R
import cv2
import subprocess

def get_red_mask(rgb):
    rgb = rgb.astype(np.float32)
    rgb = rgb / 255.
    rgb = rgb / np.linalg.norm(rgb, axis=-1)[:, :, None]
    red_vec = np.array([1., 0., 0.])
    dot = (rgb * red_vec).sum(-1)
    mask = dot > 0.9
    return mask
    
    
rc = robot_controller()
en = Ensenso()


# rc.gripper_move()
import ipdb;ipdb.set_trace()

robot_poses = [
    np.array([ 0.60237817, -0.00358309,  0.00848667]),
    np.array([ 0.58244207, -0.15353775,  0.00851401]),
    np.array([ 0.61241601, 0.0475107 , 0.21928453]),
    np.array([0.71244127, 0.04752105, 0.11300941]),
    np.array([0.71080442, 0.14749861, 0.10650563]),
    np.array([ 0.71123959, -0.00235846,  0.09946314]),
    np.array([ 0.7113076 , -0.00231346,  0.04273969]),
    np.array([0.71138603, 0.09772672, 0.13504161]),
]
cam_poses = []


for robot_pos in robot_poses:
    _, robot_ori, robot_vel, contact_force = rc.get_current_pose()
    robot_ori = R.from_matrix(robot_ori).as_euler("ZYX")
    rc.move_to_point([robot_pos[0], robot_pos[1], robot_pos[2],  robot_ori[0], robot_ori[1], robot_ori[2]])
    rgb, pcs = en.get_cam_obs()
    cv2.imwrite('debug.png', rgb[:, :, ::-1])
    red_mask = get_red_mask(rgb)
    pcs = pcs[red_mask]
    pcs = pcs[np.logical_not(np.isnan(pcs).any(1))]
    cam_pos = pcs.mean(0)
    cam_poses.append(cam_pos)

import ipdb;ipdb.set_trace()
cam_poses = np.stack(cam_poses, axis=0)
trans_mat = cv2.estimateAffine3D(np.asarray([
        cam_poses / 1000.
    ]), 
    np.asarray([
       robot_poses
    ]), force_rotation=True)[0]
en.close()
print(trans_mat)
from remote.robot_controll_client import robot_controller
from realsense import RealSense
import numpy as np
from scipy.spatial.transform import Rotation as R
import cv2
import subprocess

rc = robot_controller()
rs = RealSense()

rc.gripper_move()
import ipdb;ipdb.set_trace()

def get_mask_by_color(rgb):
    rgb = rgb.astype(np.float32)
    rgb = rgb / 255.
    rgb = rgb / np.linalg.norm(rgb, axis=-1)[:, :, None]
    red_vec = np.array([0., 0., 1.])
    dot = (rgb * red_vec).sum(-1)
    mask = dot > 0.8
    return mask



robot_poses = [
    np.array([ 0.60237817, -0.00358309,  0.00848667]),
    np.array([ 0.58244207, -0.15353775,  0.00851401]),
    np.array([ 0.39558841, -0.08404134, -0.06983644]),
    np.array([ 0.33405783, -0.07839411, -0.03746787]),
    np.array([ 0.34500841,  0.02095637, -0.00296116]),
    np.array([ 0.27361064,  0.01836667, -0.06037764]),
    np.array([ 0.26257569,  0.07910767, -0.06037579]),
    np.array([ 0.33593574,  0.03777073, -0.06802048]),
    np.array([ 0.41222993,  0.04748754, -0.07018064]),
]
cam_poses = []

for robot_pos in robot_poses:
    _, robot_ori, robot_vel, contact_force = rc.get_current_pose()
    robot_ori = R.from_matrix(robot_ori).as_euler("ZYX")
    rc.move_to_point([robot_pos[0], robot_pos[1], robot_pos[2],  robot_ori[0], robot_ori[1], robot_ori[2]])
    rgb, pcs = rs.get_cam_obs()
    cv2.imwrite('debug.png', rgb[:, :, ::-1])
    red_mask = get_mask_by_color(rgb)
    cv2.imwrite("debug2.png", (red_mask).astype(np.uint8) * 255)
    pcs = pcs[red_mask]
    pcs = pcs[np.logical_not(np.isnan(pcs).any(1))]
    cam_pos = pcs.mean(0)
    cam_poses.append(cam_pos)


import ipdb;ipdb.set_trace()
trans_mat = cv2.estimateAffine3D(np.asarray([
        cam_poses
    ]), 
    np.asarray([
       robot_poses
    ]))[1]

print(trans_mat)
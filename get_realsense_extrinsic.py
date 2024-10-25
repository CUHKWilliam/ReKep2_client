from robot_controller import robot_controller
from realsense import RealSense
import numpy as np
from scipy.spatial.transform import Rotation as R
import cv2
import subprocess

rc = robot_controller()
rs = RealSense()

rc.gripper_move()
import ipdb;ipdb.set_trace()


def get_grasp_in_cam(color, depth):
    pcd = rs.get_pcd(color, depth)
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)

    mask = np.linalg.norm(color.astype(np.float32) - np.array([91., 20., 6.]), axis=-1) < 30
    color2 = color.copy()
    color2[mask] = np.array([255, 255, 255]).astype(np.uint8)
    cv2.imwrite("debug.png",color2[:, :, ::-1])
    # import ipdb;ipdb.set_trace()

    mask = np.linalg.norm(colors.astype(np.float32) * 255 - np.array([91., 20., 6.]), axis=-1) < 30
    points = points[mask]
    pos_in_cam = points.mean(0)
    return pos_in_cam    



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
    color_image, depth_image = rs.capture(once=True)
    # try:
    cam_pos = get_grasp_in_cam(color_image, depth_image / 1000.)
    # except:
    #     print("error")
    #     continue
    cam_poses.append(cam_pos)



trans_mat = cv2.estimateAffine3D(np.asarray([
        cam_poses
    ]), 
    np.asarray([
       robot_poses
    ]), force_rotation=True)[0]

print(trans_mat)
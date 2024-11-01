import cv2                                # state of the art computer vision algorithms library
import numpy as np                        # fundamental package for scientific computing
import matplotlib.pyplot as plt           # 2D plotting library producing publication quality figures
import pyrealsense2 as rs                 # Intel RealSense cross-platform open-source API
import open3d as o3d
import time

## intrinsic:
intr_width = 640
intr_height = 480
intr_fx = 385.8968200683594
intr_fy = 385.8968200683594
intr_ppx = 325.1910400390625
intr_ppy = 237.3795928955078

# extric_mat = np.array([[-0.26164651,  0.11053399, -0.95881351, 0.99496819 + 0.04],
#  [ 0.9651535,   0.03454824, -0.25939381, 0.09863384 - 0.06],
#  [ 0.00445348, -0.99327169, -0.1157217,  -0.05693541 - 0.05]])

extric_mat = np.array(
    [[-0.55102369, -0.09769575,  0.82875113,  0.12123387],
 [ 0.83379086, -0.10508816,  0.54198642, -0.21633408],
 [ 0.03414216,  0.98965247,  0.13936389,  0.02611314],]
)

class RealSense():
    def __init__(self, ):
        # Setup:
        pipe = rs.pipeline()
        cfg = rs.config()
        # Tell config that we will use a recorded device from filem to be used by the pipeline through playback.

        serial = rs.context().devices[0].get_info(rs.camera_info.serial_number)

        rs.config.enable_device(cfg, str(serial))
        # Configure the pipeline to stream the depth stream
        cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        cfg.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)

        align = rs.align(rs.stream.color)

        ## TODO: get intrinsic
        # import ipdb;ipdb.set_trace()
        # config = pipe.start(cfg)
        # profile = config.get_stream(rs.stream.depth)  
        # intr = profile.as_video_stream_profile().get_intrinsics()


        self.cfg = cfg
        self.pipe = pipe
        self.serial = serial
        self.ctx = rs.context()
        self.align = align
        self.extric_mat = extric_mat

    def get_data(self,):
        pipe = self.pipe
        cfg = self.cfg
        pipe.start(cfg)
        flag_stop = False
        while True:
            color_image, depth_image = self.capture()
            cv2.namedWindow('RealSense', cv2.WINDOW_NORMAL)
            cv2.imshow('RealSense', color_image)
            ch = cv2.waitKey(25)
            if ch == 115:
                break
            elif ch & 0xFF == ord('q') or ch == 27:
                cv2.destroyAllWindows()
                flag_stop = True
                break
        pipe.stop()
        return color_image, depth_image, flag_stop

    def capture(self, once=False):
        # Skip 5 first frames to give the Auto-Exposure time to adjust
        pipe = self.pipe
        serial = self.serial
        cfg = self.cfg
        
        if once:
            config = pipe.start(cfg)

        time.sleep(0.5)
        try:
            pipe.wait_for_frames()
        except:
            devices = self.ctx.query_devices()
            for dev in devices:
                dev.hardware_reset()
        # Store next frameset for later processing:
        for _ in range(20):
            frameset = pipe.wait_for_frames()
            pipe.wait_for_frames()
            frameset = self.align.process(frameset)
            color_frame = frameset.get_color_frame()
            depth_frame = frameset.get_depth_frame()
            if not once:
                break
        # Cleanup:
        if once:
            pipe.stop()
        ## TODO: for debug
        # self.get_pcd(depth_frame, color_frame)
        # import ipdb;ipdb.set_trace()
        
        color = np.asanyarray(color_frame.get_data())

        color_image = np.asanyarray(color)
        cv2.imwrite("debug.png", color_image)
        depth_image = np.asanyarray(depth_frame.get_data())
        return color_image, depth_image

    def get_pcd(self, image, depth):
        depth[depth < 0.2] = 0.2
        depth[depth > 1.] = 1.
        o3d_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(intr_width, intr_height, intr_fx, intr_fy,  intr_ppx, intr_ppy)
        depth_image_np = depth
        depth_image_o3d = o3d.geometry.Image(depth_image_np.astype(np.float32))
        pcd = o3d.geometry.PointCloud.create_from_depth_image(depth_image_o3d, o3d_camera_intrinsic)
        pcs = np.asarray(pcd.points)

        cols = image.reshape(-1, 3)
        pcd.colors = o3d.utility.Vector3dVector(cols / 256.)
        return pcd

if __name__ == "__main__":
    rs = RealSense()
    image, depth = rs.capture(once=True)
    cv2.imwrite("debug.png", image[:, :, ::-1])
    pcd = rs.get_pcd(image, depth / 1000.)
    o3d.io.write_point_cloud('debug.ply', pcd)
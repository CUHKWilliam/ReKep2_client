import cv2                                # state of the art computer vision algorithms library
import numpy as np                        # fundamental package for scientific computing
import matplotlib.pyplot as plt           # 2D plotting library producing publication quality figures
import pyrealsense2 as rs
import open3d as o3d
import time

## intrinsic:
intr_width = 640
intr_height = 480
intr_fx = 385.8968200683594
intr_fy = 385.8968200683594
intr_ppx = 325.1910400390625
intr_ppy = 237.3795928955078

extric_mat = np.array([[ 0.75269223,  0.0662071 , -0.15891368,  0.68731665],
       [ 0.06249172, -0.43039059,  0.91493665, -0.38448571],
       [ 0.0031097 , -0.47081213, -0.58728788,  0.27960673]])
depth_scale = 1000
depth_max = 1.

## TODO: set correct realsense camera
REALSENSE_INDEX = 0
class RealSense():
    def __init__(self, ):
        # Setup:
        pipe = rs.pipeline()
        cfg = rs.config()
        # Tell config that we will use a recorded device from filem to be used by the pipeline through playback.
        serial = rs.context().devices[REALSENSE_INDEX].get_info(rs.camera_info.serial_number)

        rs.config.enable_device(cfg, str(serial))
        # Configure the pipeline to stream the depth stream
        cfg.enable_stream(rs.stream.depth, intr_width, intr_height, rs.format.z16, 30)
        cfg.enable_stream(rs.stream.color, intr_width, intr_height, rs.format.rgb8, 30)

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
        self.extrinsic = extric_mat
        self.intrinsic = o3d.camera.PinholeCameraIntrinsic(intr_width, intr_height, intr_fx, intr_fy,  intr_ppx, intr_ppy).intrinsic_matrix
        self.depth_scale = depth_scale
        self.depth_max = depth_max
        self.width, self.height = intr_width, intr_height

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
        for _ in range(5):
            frameset = pipe.wait_for_frames()
            # pipe.wait_for_frames()
            frameset = self.align.process(frameset)
            color_frame = frameset.get_color_frame()
            depth_frame = frameset.get_depth_frame()
            if not once:
                break
            time.sleep(0.5)
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
        mask = np.zeros_like(depth).astype(np.bool_)
        depth[depth < 0.2] = 0.2
        depth[depth > 5] = 5
        mask[depth <= 0.2] = True
        mask[depth >= 5] = True
        
        depth_image_np = depth
        depth_image_o3d = o3d.geometry.Image(depth_image_np.astype(np.float32))
        pcd = o3d.geometry.PointCloud.create_from_depth_image(depth_image_o3d, 
            o3d.camera.PinholeCameraIntrinsic(intr_width, intr_height, intr_fx, intr_fy,  intr_ppx, intr_ppy))
        pcs = np.asarray(pcd.points)
        pcs = pcs.reshape(image.shape[0], image.shape[1], 3)
        
        pcs[mask] = np.nan
        pcs = pcs.reshape(-1, 3)
        cols = image.reshape(-1, 3)
        pcd.colors = o3d.utility.Vector3dVector(cols / 256.)
        return pcd

    def get_cam_obs(self, return_depth=False):
        image, depth = self.capture(once=True)
        pcd = self.get_pcd(image, depth / depth_scale)
        points = np.asarray(pcd.points)
        points = points.reshape(image.shape[0], image.shape[1], 3)
        if not return_depth:
            return image, points
        else:
            return image, depth, points

    def get_occlusion_func(self,):
        def occlusion_func(points):
            image, depth, _ = self.get_cam_obs(return_depth=True)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            depth_proj = pcd.project_to_depth_image(intr_width,
                                        intr_height,
                                        self.intrinsic,
                                        extric_mat,
                                        depth_scale=depth_scale,
                                        depth_max=depth_max)
            depth_proj = np.asarray(depth_proj.to_legacy())
            import ipdb;ipdb.set_trace()
        return occlusion_func
        

if __name__ == "__main__":
    rs = RealSense()
    image, depth = rs.capture(once=True)
    cv2.imwrite("debug.png", image[:, :, ::-1])
    pcd = rs.get_pcd(image, depth / depth_scale)
    o3d.io.write_point_cloud('debug.ply', pcd)
    
    # rs.pipe.start(rs.cfg)
    # for i in range(20):
    #     image, depth = rs.capture(once=False)
    #     cv2.imwrite('debug.png', image[:, :, ::-1])
    #     print("i:", i)
    # rs.pipe.stop()
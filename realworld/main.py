import torch
import numpy as np
import json
import os
import argparse
from constraint_generation import ConstraintGenerator2
from subgoal_solver import SubgoalSolver
from path_solver import PathSolver
from visualizer import Visualizer
import transform_utils as T
from utils import (
    bcolors,
    get_config,
    load_functions_from_txt,
    get_linear_interpolation_steps,
    spline_interpolate_poses,
    get_callable_grasping_cost_fn,
    print_opt_debug_dict,
    grasp,
)
import cv2
import ipdb
import subprocess
import open3d as o3d
import utils as utils
from scipy.spatial.transform import Rotation as R
from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE as env_dict


def release():
    utils.ENV.open_gripper()
    return

def mask_to_pc(mask):
    env = utils.ENV
    env.get_cam_obs()
    pcs = env.last_cam_obs[1]['points'][mask]
    return pcs

USE_ENV = True
class Main:
    def __init__(self, env_name, visualize=False):
        global_config = get_config(config_path="./configs/config.yaml")
        self.config = global_config['main']
        self.bounds_min = np.array(self.config['bounds_min'])
        self.bounds_max = np.array(self.config['bounds_max'])
        self.visualize = visualize
        # set random seed
        np.random.seed(self.config['seed'])
        torch.manual_seed(self.config['seed'])
        torch.cuda.manual_seed(self.config['seed'])
        # initialize keypoint proposer and constraint generator
        self.constraint_generator = ConstraintGenerator2(global_config['constraint_generator'])
        # initialize environment
        if USE_ENV:
            ## TODO:
            benchmark_env = env_dict[env_name]
            self.env = benchmark_env(seed=0)

        if USE_ENV:
            # initialize solvers
            import ipdb;ipdb.set_trace()
            self.subgoal_solver = SubgoalSolver(global_config['subgoal_solver'], self.env.reset_joint_pos)
            self.path_solver = PathSolver(global_config['path_solver'], self.env.reset_joint_pos)
            # initialize visualizer
            if self.visualize:
                self.visualizer = Visualizer(global_config['visualizer'], self.env)
        self.grasp_state = 0
        self.path_constraint_state = {}

        ## TODO: set mass for knife
        knife = self.env.og_env.scene.objects[-3]
        link_dict = knife.links
        for key in link_dict.keys():
            link_dict[key].mass = 2

    def perform_task(self, instruction, rekep_program_dir=None, disturbance_seq=None):
        if USE_ENV:
            import ipdb;ipdb.set_trace()
            self.env.reset()
            cam_obs = self.env.render(depth=False, camera_name=self.camera)
            rgb = cam_obs[1]['rgb']
            points = cam_obs[1]['points']
            mask = cam_obs[1]['seg']
        else:
            rgb = cv2.imread("./scene.png")
        # ====================================
        # = keypoint proposal and constraint generation
        # ====================================

        rekep_program_dir = self.constraint_generator.generate(rgb, instruction, rekep_program_dir=rekep_program_dir)
        print(f'{bcolors.HEADER}Constraints generated{bcolors.ENDC}')
        # ====================================
        # = execute
        # ====================================
        self._execute(rekep_program_dir, disturbance_seq)

    def _update_disturbance_seq(self, stage, disturbance_seq):
        if disturbance_seq is not None:
            if stage in disturbance_seq and not self.applied_disturbance[stage]:
                # set the disturbance sequence, the generator will yield and instantiate one disturbance function for each env.step until it is exhausted
                if USE_ENV:
                    self.env.disturbance_seq = disturbance_seq[stage](self.env)
                self.applied_disturbance[stage] = True

    def _execute(self, rekep_program_dir, disturbance_seq=None)
        # load metadata
        with open(os.path.join(rekep_program_dir, 'metadata.json'), 'r') as f:
            self.program_info = json.load(f)
        self.applied_disturbance = {stage: False for stage in range(1, self.program_info['num_stage'] + 1)}
        if USE_ENV:
            self.env.register_keypoints(self.program_info['object_to_segment'], rekep_program_dir)
        # self.applied_disturbance = {stage: False for stage in range(1, self.program_info['num_stage'] + 1)}
        # register keypoints to be tracked
        # load constraints
        self.constraint_fns = dict()
        self.constraint_fns_code = dict()
        functions_dict = {
            "grasp": grasp,
            "release": release,
            "env": self.env,
            "np": np,
            "subprocess": subprocess,
            "o3d": o3d,
            "mask_to_pc": mask_to_pc,
        }
        for stage in range(1, self.program_info['num_stage'] + 1):  # stage starts with 1
            stage_dict = dict()
            stage_dict_code = dict()
            for constraint_type in ['subgoal', 'path', 'targets']:
                load_path = os.path.join(rekep_program_dir, f'stage_{stage}_{constraint_type}_constraints.txt')
                if not os.path.exists(load_path):
                    func, code = [], []
                else:
                    ret = load_functions_from_txt(load_path, functions_dict, return_code=True) 
                    func, code = ret['func'], ret["code"]
                ## merge the target constraints and the sub-goal constraint
                stage_dict[constraint_type] = func
                stage_dict_code[constraint_type] = code
                if constraint_type == "path":
                    for func in stage_dict[constraint_type]:
                        self.path_constraint_state[str(func)] = 0 # set inactivate
            stage_dict["subgoal"] += stage_dict['targets']
            self.constraint_fns[stage] = stage_dict
            self.constraint_fns_code[stage] = stage_dict_code

        ## TODO: manual control
        # while True:
        #     ee_pose = self.env.get_ee_pose()
        #     ee_pos = ee_pose[:3]
        #     ee_quat = ee_pose[3:]
        #     ee_euler = R.from_quat(ee_quat).as_euler("XYZ")
        #     print("manual control:")
        #     ch = input()
        #     if ch == "c":
        #         break
        #     elif ch == "a":
        #         ee_pos = ee_pos + np.array([-0.05, 0, 0])
        #     elif ch == "d":
        #         ee_pos = ee_pos + np.array([0.05, 0, 0])
        #     elif ch == "w":
        #         ee_pos = ee_pos + np.array([0, -0.05, 0])
        #     elif ch == "s":
        #         ee_pos = ee_pos + np.array([0, 0.05, 0])
        #     elif ch == "q":
        #         ee_pos = ee_pos + np.array([0, 0, 0.05])
        #     elif ch == "r":
        #         ee_pos = ee_pos + np.array([0, 0, -0.05])
        #     elif ch == "i":
        #         ee_euler = ee_euler + np.array([0.2, 0, 0])
        #     elif ch == "k":
        #         ee_euler = ee_euler + np.array([-0.2, 0, 0])
        #     elif ch == "j":
        #         ee_euler = ee_euler + np.array([0, 0.2, 0])
        #     elif ch == "l":
        #         ee_euler = ee_euler + np.array([0, -0.2, 0])
        #     elif ch == "u":
        #         ee_euler = ee_euler + np.array([0, 0, 0.2])
        #     elif ch == "o":
        #         ee_euler = ee_euler + np.array([0, 0, -0.2])
        #     ee_quat = R.from_euler("XYZ", ee_euler).as_quat()
        #     self.env.execute_action(np.concatenate([ee_pos, ee_quat, np.array([0])], axis=0))
        #     cv2.imwrite("debug.png", self.env.video_cache[-1])
        # import ipdb;ipdb.set_trace()

        # main loop
        self.last_sim_step_counter = -np.inf
        self._update_stage(1)
        while True:
            utils.ENV = self.env
            if USE_ENV:
                self.curr_ee_pose = self.env.get_ee_pose()
                self.curr_joint_pos = self.env.get_arm_joint_postions()
            # ====================================
            # = decide whether to backtrack
            # ====================================
            backtrack = False
            if self.stage > 1:
                path_constraints = self.constraint_fns[self.stage]['path']
                for constraints in path_constraints:
                    if self.path_constraint_state[str(constraints)] == 0:
                        continue
                    violation = constraints()
                    if violation > self.config['constraint_tolerance']:
                        backtrack = True
                        break
            if backtrack:
                # determine which stage to backtrack to based on constraints
                for new_stage in range(self.stage - 1, 0, -1):
                    path_constraints = self.constraint_fns[new_stage]['path']
                    # if no constraints, we can safely backtrack
                    if len(path_constraints) == 0:
                        break
                    # otherwise, check if all constraints are satisfied
                    all_constraints_satisfied = True
                    for constraints in path_constraints:
                        violation = constraints()
                        if violation > self.config['constraint_tolerance']:
                            all_constraints_satisfied = False
                            break
                    if all_constraints_satisfied:   
                        break
                print(f"{bcolors.HEADER}[stage={self.stage}] backtrack to stage {new_stage}{bcolors.ENDC}")
                self._update_stage(new_stage)
            else:
                # apply disturbance
                self._update_disturbance_seq(self.stage, disturbance_seq)
                # ====================================
                # = get optimized plan
                # ====================================
                if USE_ENV:
                    if self.last_sim_step_counter == self.env.step_counter:
                        print(f"{bcolors.WARNING}sim did not step forward within last iteration (HINT: adjust action_steps_per_iter to be larger or the pos_threshold to be smaller){bcolors.ENDC}")
                next_subgoal = self._get_next_subgoal(from_scratch=self.first_iter)
                if next_subgoal is None:
                    ## release gripper
                    pass
                else:
                    next_path = self._get_next_path(next_subgoal, from_scratch=self.first_iter)
                    self.first_iter = False
                    self.action_queue = next_path.tolist()
                    if USE_ENV:
                        self.last_sim_step_counter = self.env.step_counter
                    # ====================================
                    # = execute
                    # ====================================
                    # determine if we proceed to the next stage
                    count = 0
                    while len(self.action_queue) > 0: #  and count < self.config['action_steps_per_iter']:
                        next_action = self.action_queue.pop(0)
                        precise = len(self.action_queue) == 0
                        pos_error, rot_error = self.env.execute_action(next_action, precise=precise)
                        count += 1
                cv2.imwrite('debug.png', self.env.video_cache[-1][:, :, ::-1])
                import ipdb;ipdb.set_trace()
                if len(self.action_queue) == 0:
                    if self.grasp_state == 1:
                        self._execute_grasp_action()
                    else:
                        self._execute_release_action()
                    # if completed, save video and return
                    if self.stage == self.program_info['num_stage']: 
                        if USE_ENV:
                            self.env.sleep(2.0)
                            save_path = self.env.save_video()
                        print(f"{bcolors.OKGREEN}Video saved to {save_path}\n\n{bcolors.ENDC}")
                        return
                    # progress to next stage
                    self._update_stage(self.stage + 1)

    def _get_next_subgoal(self, from_scratch):
        subgoal_constraints = self.constraint_fns[self.stage]['subgoal']
        path_constraints = self.constraint_fns[self.stage]['path']
        result = subgoal_constraints[0]()
        if isinstance(result, dict):
            ## grasping
            subgoal_pose = result['subgoal_pose']
            debug_dict = {}
            self.grasp_state = 1
            code = self.constraint_fns_code[self.stage]['subgoal']
            ## set moving part the part connected to the end-effector
            moving_part_names = []
            moving_part_name = code.split('grasp("')[1].split('")')[0]
            moving_part_obj_name = moving_part_name.split("of")[-1].strip()
            for key in self.env.part_to_pts_dict[-1].keys():
                # if "hinge" in key or "axis" in key or "frame" in key:
                #     continue
                if key.split("of")[-1].strip() == moving_part_obj_name:
                    moving_part_names.append(key)
            self.env.moving_part_names = moving_part_names
        elif result is None:
            self.grasp_state = 0
            self.env.moving_part_names = []
            subgoal_pose = None
            debug_dict = {}
        else:
            subgoal_pose, debug_dict = self.subgoal_solver.solve(
                                                    self.curr_ee_pose,
                                                    self.env.get_part_to_pts_dict(),
                                                    self.env.get_moving_part_names(),
                                                    subgoal_constraints,
                                                    path_constraints,
                                                    self.grasp_state > 0,
                                                    self.curr_joint_pos,
                                                    from_scratch=from_scratch)
        # subgoal_pose_homo = T.convert_pose_quat2mat(subgoal_pose)
        # if grasp stage, back up a bit to leave room for grasping
        # if self.is_grasp_stage:
        #     subgoal_pose[:3] += subgoal_pose_homo[:3, :3] @ np.array([-self.config['grasp_depth'] / 2.0, 0, 0])
        debug_dict['stage'] = self.stage
        print_opt_debug_dict(debug_dict)
        if self.visualize:
            if self.env.is_grasping and subgoal_pose is not None:
                self.visualizer.visualize_subgoal(subgoal_pose, self.env.get_part_to_pts_dict()[-1], self.env.get_moving_part_names())
                import ipdb;ipdb.set_trace()
        return subgoal_pose

    def _get_next_path(self, next_subgoal, from_scratch):
        path_constraints = self.constraint_fns[self.stage]['path']
        path, debug_dict = self.path_solver.solve(self.curr_ee_pose,
                                                    next_subgoal,
                                                    self.env.get_part_to_pts_dict(),
                                                    self.env.get_moving_part_names(),
                                                    path_constraints,
                                                    self.curr_joint_pos,
                                                    from_scratch=from_scratch)
        print_opt_debug_dict(debug_dict)
        processed_path = self._process_path(path)
        if self.visualize:
            self.visualizer.visualize_path(processed_path)
        return processed_path

    def _process_path(self, path):
        # spline interpolate the path from the current ee pose
        full_control_points = np.concatenate([
            self.curr_ee_pose.reshape(1, -1),
            path,
        ], axis=0)
        num_steps = get_linear_interpolation_steps(full_control_points[0], full_control_points[-1],
                                                    self.config['interpolate_pos_step_size'],
                                                    self.config['interpolate_rot_step_size'])
        dense_path = spline_interpolate_poses(full_control_points, num_steps)

        # add gripper action
        ee_action_seq = np.zeros((dense_path.shape[0], 8))
        ee_action_seq[:, :7] = dense_path
        if USE_ENV:
            ee_action_seq[:, 7] = self.env.get_gripper_null_action()
        return ee_action_seq

    def _update_stage(self, stage):
        # update stage
        self.stage = stage
        # clear action queue
        self.action_queue = []
        # update keypoint movable mask
        # self._update_keypoint_movable_mask()
        self.first_iter = True

    def _execute_grasp_action(self):
        if self.env.is_grasping:
            return
        pregrasp_pose = self.env.get_ee_pose()
        grasp_pose = pregrasp_pose.copy()
        grasp_pose[:3] += T.quat2mat(pregrasp_pose[3:]) @ np.array([self.config['grasp_depth'], 0, 0])
        grasp_action = np.concatenate([grasp_pose, [self.env.get_gripper_close_action()]])
        self.env.execute_action(grasp_action, precise=True)
        self.env.is_grasping = True
    
    def _execute_release_action(self):
        self.env.open_gripper()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='pen', help='task to perform')
    parser.add_argument('--use_cached_query', action='store_true', help='instead of querying the VLM, use the cached query')
    parser.add_argument('--apply_disturbance', action='store_true', help='apply disturbance to test the robustness')
    parser.add_argument('--visualize', action='store_true', help='visualize each solution before executing (NOTE: this is blocking and needs to press "ESC" to continue)')
    args = parser.parse_args()
    if args.apply_disturbance:
        assert args.task == 'pen' and args.use_cached_query, 'disturbance sequence is only defined for cached scenario'

    # ====================================
    # = pen task disturbance sequence
    # ====================================
    def stage1_disturbance_seq(env):
        """
        Move the pen in stage 0 when robot is trying to grasp the pen
        """
        pen = env.og_env.scene.object_registry("name", "pen_1")
        holder = env.og_env.scene.object_registry("name", "pencil_holder_1")
        # disturbance sequence
        pos0, orn0 = pen.get_position_orientation()
        pose0 = np.concatenate([pos0, orn0])
        pos1 = pos0 + np.array([-0.08, 0.0, 0.0])
        orn1 = T.quat_multiply(T.euler2quat(np.array([0, 0, np.pi/4])), orn0)
        pose1 = np.concatenate([pos1, orn1])
        pos2 = pos1 + np.array([0.10, 0.0, 0.0])
        orn2 = T.quat_multiply(T.euler2quat(np.array([0, 0, -np.pi/2])), orn1)
        pose2 = np.concatenate([pos2, orn2])
        control_points = np.array([pose0, pose1, pose2])
        pose_seq = spline_interpolate_poses(control_points, num_steps=25)
        def disturbance(counter):
            if counter < len(pose_seq):
                pose = pose_seq[counter]
                pos, orn = pose[:3], pose[3:]
                pen.set_position_orientation(pos, orn)
                counter += 1
        counter = 0
        while True:
            yield disturbance(counter)
            counter += 1
    
    def stage2_disturbance_seq(env):
        """
        Take the pen out of the gripper in stage 1 when robot is trying to reorient the pen
        """
        apply_disturbance = env.is_grasping()
        pen = env.og_env.scene.object_registry("name", "pen_1")
        holder = env.og_env.scene.object_registry("name", "pencil_holder_1")
        # disturbance sequence
        pos0, orn0 = pen.get_position_orientation()
        pose0 = np.concatenate([pos0, orn0])
        pose1 = np.array([-0.30, -0.15, 0.71, -0.7071068, 0, 0, 0.7071068])
        control_points = np.array([pose0, pose1])
        pose_seq = spline_interpolate_poses(control_points, num_steps=25)
        def disturbance(counter):
            if apply_disturbance:
                if counter < 20:
                    if counter > 15:
                        env.robot.release_grasp_immediately()  # force robot to release the pen
                    else:
                        pass  # do nothing for the other steps
                elif counter < len(pose_seq) + 20:
                    env.robot.release_grasp_immediately()  # force robot to release the pen
                    pose = pose_seq[counter - 20]
                    pos, orn = pose[:3], pose[3:]
                    pen.set_position_orientation(pos, orn)
                    counter += 1
        counter = 0
        while True:
            yield disturbance(counter)
            counter += 1
    
    def stage3_disturbance_seq(env):
        """
        Move the holder in stage 2 when robot is trying to drop the pen into the holder
        """
        pen = env.og_env.scene.object_registry("name", "pen_1")
        holder = env.og_env.scene.object_registry("name", "pencil_holder_1")
        # disturbance sequence
        pos0, orn0 = holder.get_position_orientation()
        pose0 = np.concatenate([pos0, orn0])
        pos1 = pos0 + np.array([-0.02, -0.15, 0.0])
        orn1 = orn0
        pose1 = np.concatenate([pos1, orn1])
        control_points = np.array([pose0, pose1])
        pose_seq = spline_interpolate_poses(control_points, num_steps=5)
        def disturbance(counter):
            if counter < len(pose_seq):
                pose = pose_seq[counter]
                pos, orn = pose[:3], pose[3:]
                holder.set_position_orientation(pos, orn)
                counter += 1
        counter = 0
        while True:
            yield disturbance(counter)
            counter += 1

    task = {
        "instruction": "open the door",
        "env_name": "door-open-v2-goal-observable",
    }
    instruction = task['instruction']
    main = Main(task['env_name'], visualize=args.visualize)
    main.perform_task(instruction,
                    rekep_program_dir=task['rekep_program_dir'] if args.use_cached_query else None,
                    disturbance_seq=task.get('disturbance_seq', None) if args.apply_disturbance else None)
    

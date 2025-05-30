from typing import List, Callable, Optional
import cv2
from collections import namedtuple
import numpy as np
import torch
import os 
import re

from matplotlib import pyplot as plt
from pyrep import PyRep
from pyrep.const import ObjectType
from pyrep.errors import ConfigurationPathError
from pyrep.objects import Dummy
from pyrep.objects.shape import Shape
from pyrep.objects.vision_sensor import VisionSensor

from rlbench.backend.exceptions import (
    WaypointError, BoundaryError, NoWaypointsError, DemoError)
from rlbench.backend.observation import Observation
from rlbench.backend.robot import Robot
from rlbench.backend.spawn_boundary import SpawnBoundary
from rlbench.backend.task import Task
from rlbench.backend.utils import rgb_handles_to_mask
from rlbench.demo import Demo
from rlbench.noise_model import NoiseModel
from rlbench.observation_config import ObservationConfig, CameraConfig


STEPS_BEFORE_EPISODE_START = 10

from PIL import Image
def save_depth_16bit(depth_array, path):
    """
    Save a depth map to 16-bit PNG.
    - depth_array: (H, W) float32 array with depth values in [0, max_depth]
    """
    depth_array = np.clip(depth_array, 0, 255)
    depth_normalized = (depth_array / 255 * 65535).astype(np.uint16)
    img = Image.fromarray(depth_normalized, mode='I;16')
    img.save(path)
def save_pil(img, path):
    """
    Save an image as PNG.
    - img: (H, W, 3) uint8 array
    """
    img = np.clip(img, 0, 255).astype(np.uint8)
    img = Image.fromarray(img)
    img.save(path)


class Scene(object):
    """Controls what is currently in the vrep scene. This is used for making
    sure that the tasks are easily reachable. This may be just replaced by
    environment. Responsible for moving all the objects. """

    def __init__(self,
                 pyrep: PyRep,
                 robot: Robot,
                 obs_config: ObservationConfig = ObservationConfig(),
                 robot_setup: str = 'panda'):
        self.pyrep = pyrep
        self.robot = robot
        self.robot_setup = robot_setup
        self.task = None
        self._obs_config = obs_config
        self._initial_task_state = None
        self._start_arm_joint_pos = robot.arm.get_joint_positions()
        self._starting_gripper_joint_pos = robot.gripper.get_joint_positions()
        self._workspace = Shape('workspace')
        self._workspace_boundary = SpawnBoundary([self._workspace])
        self._cam_over_shoulder_left = VisionSensor('cam_over_shoulder_left')
        self._cam_over_shoulder_right = VisionSensor('cam_over_shoulder_right')
        self._cam_overhead = VisionSensor('cam_overhead')
        self._cam_wrist = VisionSensor('cam_wrist')
        self._cam_front = VisionSensor('cam_front')
        self._cam_over_shoulder_left_mask = VisionSensor(
            'cam_over_shoulder_left_mask')
        self._cam_over_shoulder_right_mask = VisionSensor(
            'cam_over_shoulder_right_mask')
        self._cam_overhead_mask = VisionSensor('cam_overhead_mask')
        self._cam_wrist_mask = VisionSensor('cam_wrist_mask')
        self._cam_front_mask = VisionSensor('cam_front_mask')
        self._has_init_task = self._has_init_episode = False
        self._variation_index = 0

        self._initial_robot_state = (robot.arm.get_configuration_tree(),
                                     robot.gripper.get_configuration_tree())

        self._ignore_collisions_for_current_waypoint = False

        # Set camera properties from observation config
        self._set_camera_properties()

        x, y, z = self._workspace.get_position()
        minx, maxx, miny, maxy, _, _ = self._workspace.get_bounding_box()
        self._workspace_minx = x - np.fabs(minx) - 0.2
        self._workspace_maxx = x + maxx + 0.2
        self._workspace_miny = y - np.fabs(miny) - 0.2
        self._workspace_maxy = y + maxy + 0.2
        self._workspace_minz = z
        self._workspace_maxz = z + 1.0  # 1M above workspace

        self.target_workspace_check = Dummy.create()
        self._step_callback = None

        self._robot_shapes = self.robot.arm.get_objects_in_tree(
            object_type=ObjectType.SHAPE)

    def load(self, task: Task) -> None:
        """Loads the task and positions at the centre of the workspace.

        :param task: The task to load in the scene.
        """
        task.load()  # Load the task in to the scene

        # Set at the centre of the workspace
        task.get_base().set_position(self._workspace.get_position())

        self._initial_task_state = task.get_state()
        self.task = task
        self._initial_task_pose = task.boundary_root().get_orientation()
        self._has_init_task = self._has_init_episode = False
        self._variation_index = 0

    def unload(self) -> None:
        """Clears the scene. i.e. removes all tasks. """
        if self.task is not None:
            self.robot.gripper.release()
            if self._has_init_task:
                self.task.cleanup_()
            self.task.unload()
        self.task = None
        self._variation_index = 0

    def init_task(self) -> None:
        self.task.init_task()
        self._initial_task_state = self.task.get_state()
        self._has_init_task = True
        self._variation_index = 0

    def init_episode(self, index: int, randomly_place: bool=True,
                     max_attempts: int = 5) -> List[str]:
        """Calls the task init_episode and puts randomly in the workspace.
        """

        self._variation_index = index

        if not self._has_init_task:
            self.init_task()

        # Try a few times to init and place in the workspace
        attempts = 0
        descriptions = None
        while attempts < max_attempts:
            descriptions = self.task.init_episode(index)
            try:
                if (randomly_place and
                        not self.task.is_static_workspace()):
                    self._place_task()
                    if self.robot.arm.check_arm_collision():
                        raise BoundaryError()
                self.task.validate()
                break
            except (BoundaryError, WaypointError) as e:
                self.task.cleanup_()
                self.task.restore_state(self._initial_task_state)
                attempts += 1
                if attempts >= max_attempts:
                    raise e

        # Let objects come to rest
        [self.pyrep.step() for _ in range(STEPS_BEFORE_EPISODE_START)]
        self._has_init_episode = True
        return descriptions

    def reset(self) -> None:
        """Resets the joint angles. """
        self.robot.gripper.release()

        arm, gripper = self._initial_robot_state
        self.pyrep.set_configuration_tree(arm)
        self.pyrep.set_configuration_tree(gripper)
        self.robot.arm.set_joint_positions(self._start_arm_joint_pos, disable_dynamics=True)
        self.robot.arm.set_joint_target_velocities(
            [0] * len(self.robot.arm.joints))
        self.robot.gripper.set_joint_positions(
            self._starting_gripper_joint_pos, disable_dynamics=True)
        self.robot.gripper.set_joint_target_velocities(
            [0] * len(self.robot.gripper.joints))

        if self.task is not None and self._has_init_task:
            self.task.cleanup_()
            self.task.restore_state(self._initial_task_state)
        self.task.set_initial_objects_in_scene()

    def get_observation(self,
                        use_mono_depth=False,
                        depth_model=None,
                        depth_model_transform=None,
                        depth_predict_function=None) -> Observation:
        tip = self.robot.arm.get_tip()

        joint_forces = None
        if self._obs_config.joint_forces:
            fs = self.robot.arm.get_joint_forces()
            vels = self.robot.arm.get_joint_target_velocities()
            joint_forces = self._obs_config.joint_forces_noise.apply(
                np.array([-f if v < 0 else f for f, v in zip(fs, vels)]))

        ee_forces_flat = None
        if self._obs_config.gripper_touch_forces:
            ee_forces = self.robot.gripper.get_touch_sensor_forces()
            ee_forces_flat = []
            for eef in ee_forces:
                ee_forces_flat.extend(eef)
            ee_forces_flat = np.array(ee_forces_flat)

        lsc_ob = self._obs_config.left_shoulder_camera
        rsc_ob = self._obs_config.right_shoulder_camera
        oc_ob = self._obs_config.overhead_camera
        wc_ob = self._obs_config.wrist_camera
        fc_ob = self._obs_config.front_camera

        lsc_mask_fn, rsc_mask_fn, oc_mask_fn, wc_mask_fn, fc_mask_fn = [
            (rgb_handles_to_mask if c.masks_as_one_channel else lambda x: x
             ) for c in [lsc_ob, rsc_ob, oc_ob, wc_ob, fc_ob]]

        def get_rgb_depth(sensor: VisionSensor, get_rgb: bool, get_depth: bool,
                          get_pcd: bool, rgb_noise: NoiseModel,
                          depth_noise: NoiseModel, depth_in_meters: bool):
            rgb = depth = pcd = None
            if sensor is not None and (get_rgb or get_depth):
                sensor.handle_explicitly()
                if get_rgb:
                    rgb = sensor.capture_rgb()
                    if rgb_noise is not None:
                        rgb = rgb_noise.apply(rgb)
                    rgb = np.clip((rgb * 255.).astype(np.uint8), 0, 255)
                if get_depth or get_pcd:
                    if use_mono_depth:
                        # NEW: use depth model to predict the depth
                        fx, _ = sensor.get_focal_length()
                        intrinsics = sensor.get_intrinsic_matrix()
                        # in PyRep, the coordinate is OpenGL-style
                        # so we need to flip the focal lengths
                        fx = -fx
                        intrinsics[0, 0] = -intrinsics[0, 0]
                        intrinsics[1, 1] = -intrinsics[1, 1]
                        # print(f"fx: {fx}, intrinsics: {intrinsics}")
                        depth = depth_predict_function(
                            depth_model=depth_model, 
                            depth_model_transform=depth_model_transform,
                            rgb_imgs=rgb, 
                            f_px=fx,
                            intrinsics=intrinsics)
                    else:
                        depth = sensor.capture_depth(depth_in_meters)
                    if depth_noise is not None:
                        depth = depth_noise.apply(depth)
                if get_pcd:
                    depth_m = depth
                    if not depth_in_meters and not use_mono_depth:
                        near = sensor.get_near_clipping_plane()
                        far = sensor.get_far_clipping_plane()
                        depth_m = near + depth * (far - near)
                        # depth = depth_m    
                    pcd = sensor.pointcloud_from_depth(depth_m)
                    # ## YCH: 
                    # if not get_depth:
                    #     depth = None
            return rgb, depth, pcd

        ## YCH: function getting rgbs depths pcds at once with finetuned depth model, 
        ##      using GT depth for wrist camera
        def get_rgbs_depths(
                sensors: Optional[List[VisionSensor]],  
                get_rgbs: Optional[List[bool]],        
                get_depths: Optional[List[bool]],      
                get_pcds: Optional[List[bool]],        
                rgb_noises: Optional[List[NoiseModel]],     
                depth_noises: Optional[List[NoiseModel]],   
                depth_in_meters: Optional[List[bool]]  
            ):
            N = len(sensors)
            assert N == len(get_rgbs) == len(get_depths) == len(get_pcds) \
                    == len(rgb_noises) == len(depth_noises) == len(depth_in_meters)
            # used_camera_idxs = [4]
            # used_camera_idxs = [0, 1, 4]
            used_camera_idxs = [0, 1, 3, 4]
            # Phase 1: capture & denoise all RGBs
            rgbs = []
            for sensor, do_rgb, rgb_noise in zip(sensors, get_rgbs, rgb_noises):
                rgb = None
                if sensor is not None and do_rgb:
                    sensor.handle_explicitly()
                    rgb = sensor.capture_rgb()
                    if rgb_noise is not None:
                        rgb = rgb_noise.apply(rgb)
                    rgb = np.clip((rgb * 255.0).astype(np.uint8), 0, 255)
                rgbs.append(rgb)

            # ── Phase 2: capture / predict depths ─────────────────────────
            depths = []
            for i, (sensor, rgb, do_depth, depth_noise, depth_in_meter) in enumerate(
                zip(sensors, rgbs, get_pcds, depth_noises, depth_in_meters)
            ):
                depth = None
                if sensor is not None and do_depth:
                    # ensure sensor state updated
                    if not get_rgbs[i]:
                        sensor.handle_explicitly()

                    depth = sensor.capture_depth(depth_in_meter)

                    if depth is not None and depth_noise is not None:
                        depth = depth_noise.apply(depth)

                depths.append(depth)

            if use_mono_depth:
                rgbs_, intrinsics, fxs = [], [], []
                for idx in used_camera_idxs:
                    sensor = sensors[idx]
                    fx, _ = sensor.get_focal_length()
                    intrinsic = sensor.get_intrinsic_matrix()
                    fx = -fx
                    intrinsic[0,0] = -intrinsic[0,0]
                    intrinsic[1,1] = -intrinsic[1,1]
                    rgbs_.append(rgbs[idx])
                    fxs.append(fx)
                    intrinsics.append(intrinsic)
                
                dpreds = depth_predict_function(
                    depth_model=depth_model,
                    depth_model_transform=depth_model_transform,
                    rgb_imgs=np.array(rgbs_),
                    f_px=np.array(fxs),
                    intrinsics=np.array(intrinsics)
                )

                for i in range(len(depths)):
                    if i in used_camera_idxs:
                        depths[i] = dpreds[0]
                        dpreds = dpreds[1:] 
                    elif depths[i] is not None:
                        print(f"Used GT depth from sensor {i}")

                        near = sensors[i].get_near_clipping_plane()
                        far  = sensors[i].get_far_clipping_plane()
                        depths[i] = near + depths[i] * (far - near)

                    
            frame_folder = './eval_logs/vis/4'
            pattern = re.compile(r'^vis_(\d+)\.png$')

            ids = []
            for fn in os.listdir(frame_folder):
                m = pattern.match(fn)
                if m:
                    ids.append(int(m.group(1)))

            last_id = max(ids) if ids else 0
            next_id = last_id + 1
            for i in range(len(depths)):
                if depths[i] is None:
                    continue
                rgb_view    = rgbs[i]  # (C, H, W) -> (H, W, C)
                depth = depths[i]  # (H, W)
                cv2.imwrite(f'./eval_logs/vis/{i}/vis_{next_id:03d}.png', rgb_view[:, :, ::-1])
                # plt.figure(figsize=(10, 5))

                # # RGB View
                # plt.subplot(1, 2, 1)
                # plt.imshow(rgb_view)
                # plt.title("RGB View")
                # plt.axis("off")

                # plt.subplot(1, 2, 2)
                # plt.imshow(depth, cmap='viridis')
                # plt.title("Predicted Depth")
                # plt.colorbar()
                # plt.axis("off")
                # plt.tight_layout()
                # plt.savefig(f'./eval_logs/vis/{i}/vis_{next_id:03d}.png')
                # plt.close()
            
            # ── Phase 3: compute point‑clouds ─────────────────────────────
            pcds = []
            for i, (sensor, depth, do_pcd, depth_in_meter) in enumerate(
                zip(sensors, depths, get_pcds, depth_in_meters)
            ):
                pcd = None
                if sensor is not None and do_pcd and depth is not None:
                    # convert to meters if needed
                    depth_m = depth
                    if not depth_in_meter and not use_mono_depth:
                        print("calculating depth in meters")
                        near = sensor.get_near_clipping_plane()
                        far  = sensor.get_far_clipping_plane()
                        depth_m = near + depth * (far - near)
                    pcd = sensor.pointcloud_from_depth(depth_m)
                pcds.append(pcd)
            for i, get_depth in enumerate(get_depths):
                if not get_depth:
                    depths[i] = None

                elif use_mono_depth:
                    ## TRUN DEPTH FROM METERS INTO SCALE
                    near = sensors[i].get_near_clipping_plane()
                    far  = sensors[i].get_far_clipping_plane()
                    depths[i] = (depths[i] - near) / (far - near)
                    print(f"transform depth from meters into scale")
            
            return rgbs, depths, pcds
            
        def get_mask(sensor: VisionSensor, mask_fn):
            mask = None
            if sensor is not None:
                sensor.handle_explicitly()
                mask = mask_fn(sensor.capture_rgb())
            return mask

        # left_shoulder_rgb, left_shoulder_depth, left_shoulder_pcd = get_rgb_depth(
        #     self._cam_over_shoulder_left, lsc_ob.rgb, lsc_ob.depth, lsc_ob.point_cloud,
        #     lsc_ob.rgb_noise, lsc_ob.depth_noise, lsc_ob.depth_in_meters)
        # right_shoulder_rgb, right_shoulder_depth, right_shoulder_pcd = get_rgb_depth(
        #     self._cam_over_shoulder_right, rsc_ob.rgb, rsc_ob.depth, rsc_ob.point_cloud,
        #     rsc_ob.rgb_noise, rsc_ob.depth_noise, rsc_ob.depth_in_meters)
        # overhead_rgb, overhead_depth, overhead_pcd = get_rgb_depth(
        #     self._cam_overhead, oc_ob.rgb, oc_ob.depth, oc_ob.point_cloud,
        #     oc_ob.rgb_noise, oc_ob.depth_noise, oc_ob.depth_in_meters)
        # wrist_rgb, wrist_depth, wrist_pcd = get_rgb_depth(
        #     self._cam_wrist, wc_ob.rgb, wc_ob.depth, wc_ob.point_cloud,
        #     wc_ob.rgb_noise, wc_ob.depth_noise, wc_ob.depth_in_meters)
        # front_rgb, front_depth, front_pcd = get_rgb_depth(
        #     self._cam_front, fc_ob.rgb, fc_ob.depth, fc_ob.point_cloud,
        #     fc_ob.rgb_noise, fc_ob.depth_noise, fc_ob.depth_in_meters)
        
        ## YCH: get rgbs depths pcds at once with finetuned depth model
        rgbs, depths, pcds = get_rgbs_depths(
            sensors=[self._cam_over_shoulder_left, self._cam_over_shoulder_right, self._cam_overhead, self._cam_wrist, self._cam_front],
            get_rgbs=[lsc_ob.rgb, rsc_ob.rgb, oc_ob.rgb, wc_ob.rgb, fc_ob.rgb],
            get_depths=[lsc_ob.depth, rsc_ob.depth, oc_ob.depth, wc_ob.depth, fc_ob.depth],
            get_pcds=[lsc_ob.point_cloud, rsc_ob.point_cloud, oc_ob.point_cloud, wc_ob.point_cloud, fc_ob.point_cloud],
            rgb_noises=[lsc_ob.rgb_noise, rsc_ob.rgb_noise, oc_ob.rgb_noise, wc_ob.rgb_noise, fc_ob.rgb_noise],
            depth_noises=[lsc_ob.depth_noise, rsc_ob.depth_noise, oc_ob.depth_noise, wc_ob.depth_noise, fc_ob.depth_noise],
            depth_in_meters=[lsc_ob.depth_in_meters, rsc_ob.depth_in_meters, oc_ob.depth_in_meters, wc_ob.depth_in_meters, fc_ob.depth_in_meters]
        )
        left_shoulder_rgb, left_shoulder_depth, left_shoulder_pcd = rgbs[0], depths[0], pcds[0]
        right_shoulder_rgb, right_shoulder_depth, right_shoulder_pcd = rgbs[1], depths[1], pcds[1]
        overhead_rgb, overhead_depth, overhead_pcd = rgbs[2], depths[2], pcds[2]
        wrist_rgb, wrist_depth, wrist_pcd = rgbs[3], depths[3], pcds[3]
        front_rgb, front_depth, front_pcd = rgbs[4], depths[4], pcds[4]
        # ## YCH: collecting sim depth data
        # def get_next_frame_idx(directory):
        #     files = os.listdir(directory)
            
        #     pattern = re.compile(r'^(\d{5})\.npz$')
            
        #     indices = [
        #         int(pattern.match(f).group(1))
        #         for f in files if pattern.match(f)
        #     ]
            
        #     last_idx = max(indices, default=-1)
        #     return last_idx + 1
        
        # def save_multi_cam_npz(frame_idx: int,
        #                left_shoulder: tuple,
        #                right_shoulder: tuple,
        #                overhead: tuple,
        #                wrist: tuple,
        #                front: tuple,
        #                out_dir: str = './data'):
        #     # unpack
        #     l_rgb, l_depth = left_shoulder
        #     r_rgb, r_depth = right_shoulder
        #     o_rgb, o_depth = overhead
        #     w_rgb, w_depth = wrist
        #     f_rgb, f_depth = front

        #     # build filename
        #     fname = f"{out_dir}/{frame_idx:05d}.npz"

        #     # save
        #     np.savez_compressed(
        #         fname,
        #         left_shoulder_rgb   = l_rgb,
        #         left_shoulder_depth = l_depth,
        #         right_shoulder_rgb   = r_rgb,
        #         right_shoulder_depth = r_depth,
        #         overhead_rgb        = o_rgb,
        #         overhead_depth      = o_depth,
        #         wrist_rgb           = w_rgb,
        #         wrist_depth         = w_depth,
        #         front_rgb           = f_rgb,
        #         front_depth         = f_depth,
        #     )
        
        # out_dir = '/project2/yehhh/datasets/RLBench/sim-depth_m'
        # print(f"next frame idx: {get_next_frame_idx(out_dir)}")
        # save_multi_cam_npz(
        #     get_next_frame_idx(out_dir),
        #     (left_shoulder_rgb,  left_shoulder_depth),
        #     (right_shoulder_rgb, right_shoulder_depth),
        #     (overhead_rgb,       overhead_depth),
        #     (wrist_rgb,          wrist_depth),
        #     (front_rgb,          front_depth),
        #     out_dir=out_dir
        # )
        # left_shoulder_depth, right_shoulder_depth, overhead_depth, wrist_depth, front_depth = [None, None, None, None, None]


        left_shoulder_mask = get_mask(self._cam_over_shoulder_left_mask,
                                      lsc_mask_fn) if lsc_ob.mask else None
        right_shoulder_mask = get_mask(self._cam_over_shoulder_right_mask,
                                      rsc_mask_fn) if rsc_ob.mask else None
        overhead_mask = get_mask(self._cam_overhead_mask,
                                 oc_mask_fn) if oc_ob.mask else None
        wrist_mask = get_mask(self._cam_wrist_mask,
                              wc_mask_fn) if wc_ob.mask else None
        front_mask = get_mask(self._cam_front_mask,
                              fc_mask_fn) if fc_ob.mask else None

        obs = Observation(
            left_shoulder_rgb=left_shoulder_rgb,
            left_shoulder_depth=left_shoulder_depth,
            left_shoulder_point_cloud=left_shoulder_pcd,
            right_shoulder_rgb=right_shoulder_rgb,
            right_shoulder_depth=right_shoulder_depth,
            right_shoulder_point_cloud=right_shoulder_pcd,
            overhead_rgb=overhead_rgb,
            overhead_depth=overhead_depth,
            overhead_point_cloud=overhead_pcd,
            wrist_rgb=wrist_rgb,
            wrist_depth=wrist_depth,
            wrist_point_cloud=wrist_pcd,
            front_rgb=front_rgb,
            front_depth=front_depth,
            front_point_cloud=front_pcd,
            left_shoulder_mask=left_shoulder_mask,
            right_shoulder_mask=right_shoulder_mask,
            overhead_mask=overhead_mask,
            wrist_mask=wrist_mask,
            front_mask=front_mask,
            joint_velocities=(
                self._obs_config.joint_velocities_noise.apply(
                    np.array(self.robot.arm.get_joint_velocities()))
                if self._obs_config.joint_velocities else None),
            joint_positions=(
                self._obs_config.joint_positions_noise.apply(
                    np.array(self.robot.arm.get_joint_positions()))
                if self._obs_config.joint_positions else None),
            joint_forces=(joint_forces
                          if self._obs_config.joint_forces else None),
            gripper_open=(
                (1.0 if self.robot.gripper.get_open_amount()[0] > 0.95 else 0.0) # Changed from 0.9 to 0.95 because objects, the gripper does not close completely
                if self._obs_config.gripper_open else None),
            gripper_pose=(
                np.array(tip.get_pose())
                if self._obs_config.gripper_pose else None),
            gripper_matrix=(
                tip.get_matrix()
                if self._obs_config.gripper_matrix else None),
            gripper_touch_forces=(
                ee_forces_flat
                if self._obs_config.gripper_touch_forces else None),
            gripper_joint_positions=(
                np.array(self.robot.gripper.get_joint_positions())
                if self._obs_config.gripper_joint_positions else None),
            task_low_dim_state=(
                self.task.get_low_dim_state() if
                self._obs_config.task_low_dim_state else None),
            ignore_collisions=(
                np.array((1.0 if self._ignore_collisions_for_current_waypoint else 0.0))
                if self._obs_config.record_ignore_collisions else None),
            misc=self._get_misc())
        obs = self.task.decorate_observation(obs)
        return obs

    def step(self):
        self.pyrep.step()
        self.task.step()
        if self._step_callback is not None:
            self._step_callback()

    def register_step_callback(self, func):
        self._step_callback = func

    def get_demo(self, record: bool = True,
                 callable_each_step: Callable[[Observation], None] = None,
                 randomly_place: bool = True,
                use_mono_depth=False,
                depth_model=None,
                depth_model_transform=None,
                depth_predict_function=None
                ) -> Demo:
        """Returns a demo (list of observations)"""

        if not self._has_init_task:
            self.init_task()
        if not self._has_init_episode:
            self.init_episode(self._variation_index,
                              randomly_place=randomly_place)
        self._has_init_episode = False

        waypoints = self.task.get_waypoints()
        if len(waypoints) == 0:
            raise NoWaypointsError(
                'No waypoints were found.', self.task)
        demo = []
        if record:
            self.pyrep.step()  # Need this here or get_force doesn't work...
            demo.append(self.get_observation(use_mono_depth=use_mono_depth, depth_model=depth_model, depth_model_transform=depth_model_transform, depth_predict_function=depth_predict_function))
        while True:
            success = False
            self._ignore_collisions_for_current_waypoint = False
            for i, point in enumerate(waypoints):
                self._ignore_collisions_for_current_waypoint = point._ignore_collisions
                point.start_of_path()
                if point.skip:
                    continue
                grasped_objects = self.robot.gripper.get_grasped_objects()
                colliding_shapes = [s for s in self.pyrep.get_objects_in_tree(
                    object_type=ObjectType.SHAPE) if s not in grasped_objects
                                    and s not in self._robot_shapes and s.is_collidable()
                                    and self.robot.arm.check_arm_collision(s)]
                [s.set_collidable(False) for s in colliding_shapes]
                try:
                    path = point.get_path()
                    [s.set_collidable(True) for s in colliding_shapes]
                except ConfigurationPathError as e:
                    [s.set_collidable(True) for s in colliding_shapes]
                    raise DemoError(
                        'Could not get a path for waypoint %d.' % i,
                        self.task) from e
                ext = point.get_ext()
                path.visualize()

                done = False
                success = False
                while not done:
                    done = path.step()
                    self.step()
                    self._demo_record_step(demo, record, callable_each_step, 
                                            use_mono_depth=use_mono_depth, 
                                            depth_model=depth_model, 
                                            depth_model_transform=depth_model_transform, 
                                            depth_predict_function=depth_predict_function)
                    success, term = self.task.success()

                point.end_of_path()

                path.clear_visualization()
                print(f"scene.py 629")
                if len(ext) > 0:
                    contains_param = False
                    start_of_bracket = -1
                    gripper = self.robot.gripper
                    if 'open_gripper(' in ext:
                        gripper.release()
                        start_of_bracket = ext.index('open_gripper(') + 13
                        contains_param = ext[start_of_bracket] != ')'
                        if not contains_param:
                            done = False
                            while not done:
                                done = gripper.actuate(1.0, 0.04)
                                self.pyrep.step()
                                self.task.step()
                                if self._obs_config.record_gripper_closing:
                                    print(f"get obs at scene.py 645")
                                    self._demo_record_step(
                                        demo, record, callable_each_step)
                    elif 'close_gripper(' in ext:
                        start_of_bracket = ext.index('close_gripper(') + 14
                        contains_param = ext[start_of_bracket] != ')'
                        if not contains_param:
                            done = False
                            while not done:
                                done = gripper.actuate(0.0, 0.04)
                                self.pyrep.step()
                                self.task.step()
                                if self._obs_config.record_gripper_closing:
                                    print(f"get obs at scene.py 658")
                                    self._demo_record_step(
                                        demo, record, callable_each_step)

                    if contains_param:
                        rest = ext[start_of_bracket:]
                        num = float(rest[:rest.index(')')])
                        done = False
                        while not done:
                            done = gripper.actuate(num, 0.04)
                            self.pyrep.step()
                            self.task.step()
                            if self._obs_config.record_gripper_closing:
                                print(f"get obs at scene.py 671")
                                self._demo_record_step(
                                    demo, record, callable_each_step)

                    if 'close_gripper(' in ext:
                        for g_obj in self.task.get_graspable_objects():
                            gripper.grasp(g_obj)

                    self._demo_record_step(demo, record, callable_each_step, 
                                           use_mono_depth=use_mono_depth, 
                                            depth_model=depth_model, 
                                            depth_model_transform=depth_model_transform, 
                                            depth_predict_function=depth_predict_function)

            if not self.task.should_repeat_waypoints() or success:
                break

        # Some tasks may need additional physics steps
        # (e.g. ball rowling to goal)
        if not success:
            for _ in range(10):
                self.pyrep.step()
                self.task.step()
                self._demo_record_step(demo, record, callable_each_step, 
                                       use_mono_depth=use_mono_depth, 
                                            depth_model=depth_model, 
                                            depth_model_transform=depth_model_transform, 
                                            depth_predict_function=depth_predict_function)
                success, term = self.task.success()
                if success:
                    break

        success, term = self.task.success()
        if not success:
            raise DemoError('Demo was completed, but was not successful.',
                            self.task)
        return Demo(demo)

    def get_observation_config(self) -> ObservationConfig:
        return self._obs_config

    def check_target_in_workspace(self, target_pos: np.ndarray) -> bool:
        x, y, z = target_pos
        return (self._workspace_maxx > x > self._workspace_minx and
                self._workspace_maxy > y > self._workspace_miny and
                self._workspace_maxz > z > self._workspace_minz)

    def _demo_record_step(self, demo_list, record, func,
                            use_mono_depth=False,
                            depth_model=None,
                            depth_model_transform=None,
                            depth_predict_function=None
                            ):
        if record:
            demo_list.append(self.get_observation(
                                use_mono_depth=use_mono_depth, 
                                depth_model=depth_model, 
                                depth_model_transform=depth_model_transform, 
                                depth_predict_function=depth_predict_function))
        if func is not None:
            print(f"get obs at scene.py 712")
            func(self.get_observation())

    def _set_camera_properties(self) -> None:
        def _set_rgb_props(rgb_cam: VisionSensor,
                           rgb: bool, depth: bool, conf: CameraConfig):
            if not (rgb or depth or conf.point_cloud):
                rgb_cam.remove()
            else:
                rgb_cam.set_explicit_handling(1)
                rgb_cam.set_resolution(conf.image_size)
                rgb_cam.set_render_mode(conf.render_mode)

        def _set_mask_props(mask_cam: VisionSensor, mask: bool,
                            conf: CameraConfig):
                if not mask:
                    mask_cam.remove()
                else:
                    mask_cam.set_explicit_handling(1)
                    mask_cam.set_resolution(conf.image_size)
        _set_rgb_props(
            self._cam_over_shoulder_left,
            self._obs_config.left_shoulder_camera.rgb,
            self._obs_config.left_shoulder_camera.depth,
            self._obs_config.left_shoulder_camera)
        _set_rgb_props(
            self._cam_over_shoulder_right,
            self._obs_config.right_shoulder_camera.rgb,
            self._obs_config.right_shoulder_camera.depth,
            self._obs_config.right_shoulder_camera)
        _set_rgb_props(
            self._cam_overhead,
            self._obs_config.overhead_camera.rgb,
            self._obs_config.overhead_camera.depth,
            self._obs_config.overhead_camera)
        _set_rgb_props(
            self._cam_wrist, self._obs_config.wrist_camera.rgb,
            self._obs_config.wrist_camera.depth,
            self._obs_config.wrist_camera)
        _set_rgb_props(
            self._cam_front, self._obs_config.front_camera.rgb,
            self._obs_config.front_camera.depth,
            self._obs_config.front_camera)
        _set_mask_props(
            self._cam_over_shoulder_left_mask,
            self._obs_config.left_shoulder_camera.mask,
            self._obs_config.left_shoulder_camera)
        _set_mask_props(
            self._cam_over_shoulder_right_mask,
            self._obs_config.right_shoulder_camera.mask,
            self._obs_config.right_shoulder_camera)
        _set_mask_props(
            self._cam_overhead_mask,
            self._obs_config.overhead_camera.mask,
            self._obs_config.overhead_camera)
        _set_mask_props(
            self._cam_wrist_mask, self._obs_config.wrist_camera.mask,
            self._obs_config.wrist_camera)
        _set_mask_props(
            self._cam_front_mask, self._obs_config.front_camera.mask,
            self._obs_config.front_camera)

    def _place_task(self) -> None:
        self._workspace_boundary.clear()
        # Find a place in the robot workspace for task
        self.task.boundary_root().set_orientation(
            self._initial_task_pose)
        min_rot, max_rot = self.task.base_rotation_bounds()
        self._workspace_boundary.sample(
            self.task.boundary_root(),
            min_rotation=min_rot, max_rotation=max_rot)

    def _get_misc(self):
        def _get_cam_data(cam: VisionSensor, name: str):
            d = {}
            if cam.still_exists():
                d = {
                    '%s_extrinsics' % name: cam.get_matrix(),
                    '%s_intrinsics' % name: cam.get_intrinsic_matrix(),
                    '%s_near' % name: cam.get_near_clipping_plane(),
                    '%s_far' % name: cam.get_far_clipping_plane(),
                }
            return d
        misc = _get_cam_data(self._cam_over_shoulder_left, 'left_shoulder_camera')
        misc.update(_get_cam_data(self._cam_over_shoulder_right, 'right_shoulder_camera'))
        misc.update(_get_cam_data(self._cam_overhead, 'overhead_camera'))
        misc.update(_get_cam_data(self._cam_front, 'front_camera'))
        misc.update(_get_cam_data(self._cam_wrist, 'wrist_camera'))
        return misc
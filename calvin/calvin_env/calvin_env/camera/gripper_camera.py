import numpy as np
import math
import pybullet as p

from calvin_env.camera.camera import Camera


class GripperCamera(Camera):
    def __init__(self, fov, aspect, nearval, farval, width, height, robot_id, cid, name, objects=None):
        self.cid = cid
        self.robot_uid = robot_id
        links = {
            p.getJointInfo(self.robot_uid, i, physicsClientId=self.cid)[12].decode("utf-8"): i
            for i in range(p.getNumJoints(self.robot_uid, physicsClientId=self.cid))
        }
        self.gripper_cam_link = links["gripper_cam"]
        self.fov = fov
        self.aspect = aspect
        self.nearval = nearval
        self.farval = farval
        self.width = width
        self.height = height

        self.name = name

    def get_focal_length(self):
        """
        Compute the focal length in pixels from the vertical field of view and image height.
        Returns:
            (fx, fy): Tuple of focal lengths in pixels along x and y directions
        """
        fov_rad = math.radians(self.fov)  # convert FOV to radians
        fy = self.height / (2 * math.tan(fov_rad / 2))
        fx = fy * self.aspect  # aspect = width / height
        return fx, fy

    def get_intrinsic_matrix(self) -> np.ndarray:
        """
        Compute the 3x3 intrinsic camera matrix (K) based on pinhole model,
        using the FOV, aspect ratio, and image resolution.

        Returns:
            np.ndarray: The 3x3 intrinsic matrix K.
        """
        fov_rad = math.radians(self.fov)
        fy = self.height / (2 * math.tan(fov_rad / 2))
        fx = fy * self.aspect  # or: fx = fy * (self.width / self.height)

        cx = self.width / 2
        cy = self.height / 2

        K = np.array([
            [fx,  0, cx],
            [ 0, fy, cy],
            [ 0,  0,  1]
        ])
        return K

    def render(self):
        camera_ls = p.getLinkState(
            bodyUniqueId=self.robot_uid, linkIndex=self.gripper_cam_link, physicsClientId=self.cid
        )
        camera_pos, camera_orn = camera_ls[:2]
        cam_rot = p.getMatrixFromQuaternion(camera_orn)
        cam_rot = np.array(cam_rot).reshape(3, 3)
        cam_rot_y, cam_rot_z = cam_rot[:, 1], cam_rot[:, 2]
        # camera: eye position, target position, up vector
        self.view_matrix = p.computeViewMatrix(camera_pos, camera_pos + cam_rot_y, -cam_rot_z)
        self.projection_matrix = p.computeProjectionMatrixFOV(
            fov=self.fov, aspect=self.aspect, nearVal=self.nearval, farVal=self.farval
        )
        image = p.getCameraImage(
            width=self.width,
            height=self.height,
            viewMatrix=self.view_matrix,
            projectionMatrix=self.projection_matrix,
            physicsClientId=self.cid,
        )
        rgb_img, depth_img = self.process_rgbd(image, self.nearval, self.farval)
        return rgb_img, depth_img

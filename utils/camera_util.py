"""
Camera utilities for converting between different camera parameterizations.

This module provides functions to convert camera parameters from the dataset format
(quaternion + translation + fov) to the format expected by the voxel rasterizer
(view and projection matrices).

Parameter Conventions:
- Dataset quaternions: scipy format (x, y, z, w) 
- utils3d quaternions: (w, x, y, z) format
- Dataset extrinsics_local: World-to-Camera transformation (standard CV convention)
- Rasterizer expects: OpenGL view and projection matrices
- All operations are GPU-compatible and support batch dimensions
"""

import torch
import torch.nn.functional as F
import utils3d
from einops import rearrange


def worldmem_cam_to_minetest_cam(poses):
    """
    Convert WorldMem 5D poses to Minetest camera format.

    Args:
        poses: (T, 5) [x, y, z, pitch_deg, yaw_deg]

    Returns:
        cam_pos: (T, 3) position in ENU
        cam_dir: (T, 3) direction vector in ENU (unit length)
    """
    wm_x, wm_y, wm_z = poses[:, 0], poses[:, 1], poses[:, 2]
    wm_pitch, wm_yaw = poses[:, 3], poses[:, 4]

    cam_pos = torch.stack([wm_x, -wm_z, wm_y], dim=-1)

    pr = torch.deg2rad(wm_pitch)
    yr = torch.deg2rad(wm_yaw)
    wm_fwd_x = torch.sin(yr) * torch.cos(pr)
    wm_fwd_y = -torch.sin(pr)
    wm_fwd_z = torch.cos(yr) * torch.cos(pr)
    cam_dir = torch.stack([wm_fwd_x, -wm_fwd_z, wm_fwd_y], dim=-1)

    return cam_pos, cam_dir


def minetest_cam_to_worldmem_cam(cam_pos, cam_dir):
    """
    Convert Minetest camera format to WorldMem 5D poses.

    Args:
        cam_pos: (T, 3) position in ENU
        cam_dir: (T, 3) direction vector in ENU

    Returns:
        poses: (T, 5) [x, y, z, pitch_deg, yaw_deg]
    """
    mt_x, mt_y, mt_z = cam_pos[:, 0], cam_pos[:, 1], cam_pos[:, 2]

    wm_x = mt_x
    wm_y = mt_z
    wm_z = -mt_y

    wm_dir_x = cam_dir[:, 0]
    wm_dir_y = cam_dir[:, 2]
    wm_dir_z = -cam_dir[:, 1]

    norm = torch.sqrt(wm_dir_x**2 + wm_dir_y**2 + wm_dir_z**2)
    wm_pitch = torch.rad2deg(torch.arcsin((-wm_dir_y / norm).clamp(-1.0, 1.0)))
    wm_yaw = torch.rad2deg(torch.atan2(wm_dir_x, wm_dir_z))

    return torch.stack([wm_x, wm_y, wm_z, wm_pitch, wm_yaw], dim=-1)

def project_voxel_local_to_worldcam(camera):
    B, T = camera.shape[0], camera.shape[1]
    camera = rearrange(camera, "B T D -> (B T) D")
    R = rotation_6d_to_matrix(camera[..., :6])
    t = local_to_worldcam(camera[..., -4:-1], R)
    camera[..., -4:-1] = t
    camera = rearrange(camera, "(B T) D -> B T D", B=B, T=T)
    return camera

def camera_to_xyz_pry_fov(cameras, in_degrees=True, rotation_rep='6d'):
    """Convert camera tensor to roll, pitch, yaw representation."""
    if rotation_rep == '6d':
        rot_6d = cameras[..., :6]
        rot_mat = rotation_6d_to_matrix(rot_6d)
        pry = utils3d.matrix_to_euler_angles(rot_mat, 'ZYX')
    elif rotation_rep == 'cam_dir':
        cam_dir = cameras[..., :3]
        cam_pos = torch.zeros_like(cam_dir)
        ups = torch.tile(torch.tensor([0, 0, 1], dtype=cam_pos.dtype), cam_pos.shape[:-1] + (1,)).to(
            cam_pos.device
        )
        extrinsics = utils3d.torch.extrinsics_look_at(cam_pos, cam_dir, ups)
        rot_mat = extrinsics[..., :3, :3]
        pry = utils3d.matrix_to_euler_angles(rot_mat, 'ZYX')

    xyz = cameras[..., -4:-1]
    fov = cameras[..., -1]
    if in_degrees:
        fov = torch.rad2deg(fov)
        pry = torch.rad2deg(pry)
    return xyz, pry, fov

def pitch_yaw_to_cam_dir(pitch, yaw, eps=1e-8, in_degrees=True):
    if in_degrees:
        pitch = torch.deg2rad(pitch)
        yaw = torch.deg2rad(yaw)
    pitch = pitch - torch.pi / 2.0
    y = torch.cos(pitch) * torch.cos(yaw)
    x = torch.cos(pitch) * torch.sin(yaw)
    z = -torch.sin(pitch)
    cam_dir = torch.stack([x, y, z], dim=-1)
    cam_dir = cam_dir / cam_dir.norm(dim=-1, keepdim=True).clamp_min(eps)
    return cam_dir

def cam_dir_to_6d(cam_dir, eps=1e-8):
    up = torch.tile(torch.tensor([0, 0, 1], dtype=cam_dir.dtype), cam_dir.shape[:-1] + (1,)).to(
        cam_dir.device
    )
    x = torch.cross(-up, cam_dir, dim=-1)
    y = torch.cross(cam_dir, x, dim=-1)
    x = x / x.norm(dim=-1, keepdim=True).clamp_min(eps)
    y = y / y.norm(dim=-1, keepdim=True).clamp_min(eps)
    cam_dir = cam_dir / cam_dir.norm(dim=-1, keepdim=True).clamp_min(eps)
    R = torch.stack([x, y, cam_dir], dim=-2)
    r6d = matrix_to_rotation_6d(R)
    return r6d

def compute_cam_pose_local(cam_pos, voxel_center_pos, dataset_params):
    vox_preprocessing = dataset_params["voxel_info"]["active_voxel_preprocessing"]
    assert (
        vox_preprocessing["pad"] == False
        and vox_preprocessing["scale"] == False
        and vox_preprocessing["crop"] is not None
    ), "Only cropping is supported when computing intrinsics and extrinsics"
    assert all(vox_preprocessing["crop"]["left"]) == 0, (
        "Only right crop is supported when computing intrinsics and extrinsics"
    )

    new_voxel_center_pos = (
        voxel_center_pos - 0.5
    )  # shift from coordinate being in the center of the voxel to the corner
    cam_pos_local = cam_pos - new_voxel_center_pos
    voxel_grid_size = torch.tensor(
        dataset_params["voxel_info"]["dims"][:3], dtype=torch.float32
    ).to(cam_pos.device)
    cam_pos_local = cam_pos_local / voxel_grid_size.reshape(1, 1, 3)

    return cam_pos_local


def compute_extrinsics(cam_pos, cam_dir, eps=1e-8):
    f = cam_dir / (cam_dir.norm(dim=-1, keepdim=True).clamp_min(eps))  # unit forward
    look_at_pos = cam_pos + f
    ups = torch.tile(torch.tensor([0, 0, 1], dtype=cam_pos.dtype), cam_pos.shape[:-1] + (1,)).to(
        cam_pos.device
    )
    return utils3d.torch.extrinsics_look_at(cam_pos, look_at_pos, ups)

def local_to_worldcam(pos, R):
    """
    Rotate pos vector if not in World-to-Camera format (standard CV convention)

    Args:
        pos: tensor of shape (B, 3) encoding a camera location in world frame
        R: tensor of shape (B, 3, 3) encoding a rotation matrix

    """
    t = -torch.matmul(R, pos[..., None])
    t = t[:, :, 0]
    return t

def camera_params_to_matrices(
    camera_params: torch.Tensor,
    image_width: int = 640,
    image_height: int = 360,
    near: float = 0.001,
    far: float = 10.0,
    device: torch.device = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convert camera parameters to view and projection matrices for the rasterizer.

    Args:
        camera_params: Camera parameters tensor of shape (..., 8) or (..., 10):
                      - 8D: [qx, qy, qz, qw, tx, ty, tz, fov_radians]
                            where quaternion is in (x, y, z, w) format from scipy
                      - 10D: [r1, r2, r3, r4, r5, r6, tx, ty, tz, fov_radians]
                             where first 6 dims are 6D rotation representation
        image_width: Width of the rendered image
        image_height: Height of the rendered image
        near: Near clipping plane distance
        far: Far clipping plane distance
        device: Device to put matrices on (inferred from input if None)

    Returns:
        view_matrix: View matrix of shape (..., 4, 4) for OpenGL convention
        projection_matrix: Projection matrix of shape (..., 4, 4) for OpenGL convention
    """
    if device is None:
        device = camera_params.device

    image_width = torch.tensor(image_width, device=device)
    image_height = torch.tensor(image_height, device=device)
    near = torch.tensor(near, device=device)
    far = torch.tensor(far, device=device)

    # Detect format based on last dimension size
    param_dim = camera_params.shape[-1]

    if param_dim == 8:
        # Quaternion format: [qx, qy, qz, qw, tx, ty, tz, fov]
        quaternions = camera_params[..., :4]  # (x, y, z, w) from scipy
        translations = camera_params[..., 4:7]
        fovs = camera_params[..., 7]

        # Convert scipy quaternion (x,y,z,w) to utils3d quaternion (w,x,y,z)
        quaternions_utils3d = torch.stack([
            quaternions[..., 3],  # w
            quaternions[..., 0],  # x
            quaternions[..., 1],  # y
            quaternions[..., 2]   # z
        ], dim=-1)
        rotation_matrices = utils3d.torch.quaternion_to_matrix(quaternions_utils3d)

    elif param_dim == 10:
        # 6D rotation format: [r1, r2, r3, r4, r5, r6, tx, ty, tz, fov]
        rotation_6d = camera_params[..., :6]
        translations = camera_params[..., 6:9]
        fovs = camera_params[..., 9]
        rotation_matrices = rotation_6d_to_matrix(rotation_6d)

    else:
        raise ValueError(f"camera_params must have 8 (quaternion) or 10 (6D rotation) dims, got {param_dim}")
    
    # Create 4x4 transformation matrices from rotation and translation
    # The dataset's extrinsics_local is already in World-to-Camera format (standard CV convention)
    # No inversion needed
    extrinsics_matrix = utils3d.torch.to4x4(rotation_matrices, translations)
    
    # Convert to OpenGL view matrix format
    view_matrix = utils3d.torch.extrinsics_to_view(extrinsics_matrix)

    # Create projection matrix from FOV
    projection_matrix = utils3d.torch.perspective_from_fov(
        fov=fovs,  # fov_x
        width=image_width,
        height=image_height,
        near=near,
        far=far
    )
    
    return view_matrix, projection_matrix

def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """
    from https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/rotation_conversions.html
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalization per Section B of [1].
    Args:
        d6: 6D rotation representation, of size (*, 6)

    Returns:
        batch of rotation matrices of size (*, 3, 3)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """

    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)

def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    """
    adapted from https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/rotation_conversions.html
    Converts rotation matrices to 6D rotation representation by Zhou et al. [1]
    by dropping the last row. Note that 6D representation is not unique.
    Args:
        matrix: batch of rotation matrices of size (*, 3, 3)

    Returns:
        6D rotation representation, of size (*, 6)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    batch_dim = matrix.shape[:-2]
    return matrix[..., :2, :].reshape(batch_dim + (6,))


def angular_diff_deg(a, b):
    """
    Minimal signed angular difference (degrees) between a and b.
    a, b: tensors in degrees, typically in [-180, 180]
    returns: angle in [-180, 180]
    """
    return (a - b + 180.0) % 360.0 - 180.0

def compute_dpitch_dyaw_from_cam_dir(cam_dir):
    # note: pitch/yaw may have some offset depending on axis convention but dpitch/dyaw remain correct
    f = cam_dir / cam_dir.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    pitch = torch.rad2deg(torch.atan2(f[..., 2], torch.sqrt(f[..., 0]**2 + f[..., 1]**2)))
    dpitch = pitch[..., 1:] - pitch[..., :-1]
    yaw   = torch.rad2deg(torch.atan2(f[..., 1], f[..., 0]))
    dyaw = angular_diff_deg(yaw[..., 1:], yaw[..., :-1])
    return dpitch.numpy(), dyaw.numpy()

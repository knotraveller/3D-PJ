import math
import numpy as np
from mathutils import Matrix

def get_intrinsics_from_fov(resolution, fov_deg):
    W = H = resolution
    fov = math.radians(fov_deg)

    fx = fy = 0.5 * W / math.tan(0.5 * fov)
    cx = W / 2.0
    cy = H / 2.0

    K = np.array([
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)

    return K


def matrix_to_numpy(mat):
    return np.array(mat, dtype=np.float32)


def get_camera_matrices(camera, resolution, fov_deg):
    """
    返回 Blender convention 下的 c2w / w2c / K。
    camera.matrix_world 是 Blender 的 camera-to-world。
    """
    c2w = matrix_to_numpy(camera.matrix_world)
    w2c = matrix_to_numpy(camera.matrix_world.inverted())
    K = get_intrinsics_from_fov(resolution, fov_deg)

    return K, c2w, w2c


def get_rays_blender(camera, resolution, fov_deg):
    """
    生成每个像素的世界坐标射线。

    返回:
        rays_o: [H, W, 3]
        rays_d: [H, W, 3]

    坐标约定:
        Blender camera local:
            +X right
            +Y up
            -Z forward
    """
    H = W = resolution
    K, c2w, w2c = get_camera_matrices(camera, resolution, fov_deg)

    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    # 像素中心
    u, v = np.meshgrid(
        np.arange(W, dtype=np.float32) + 0.5,
        np.arange(H, dtype=np.float32) + 0.5,
        indexing="xy"
    )

    x_cam = (u - cx) / fx
    y_cam = -(v - cy) / fy
    z_cam = -np.ones_like(x_cam)

    dirs_cam = np.stack([x_cam, y_cam, z_cam], axis=-1)  # [H, W, 3]
    dirs_cam = dirs_cam / np.linalg.norm(dirs_cam, axis=-1, keepdims=True)

    R_c2w = c2w[:3, :3]
    t_c2w = c2w[:3, 3]

    # [H, W, 3] @ [3, 3]^T
    rays_d = dirs_cam @ R_c2w.T
    rays_d = rays_d / np.linalg.norm(rays_d, axis=-1, keepdims=True)

    rays_o = np.broadcast_to(t_c2w.reshape(1, 1, 3), rays_d.shape)

    return rays_o.astype(np.float32), rays_d.astype(np.float32)

def get_plucker_rays(camera, resolution, fov_deg):
    rays_o, rays_d = get_rays_blender(camera, resolution, fov_deg)

    # moment = origin × direction
    rays_m = np.cross(rays_o, rays_d)

    # [H, W, 6]
    rays = np.concatenate([rays_d, rays_m], axis=-1).astype(np.float32)

    return rays

def get_zero123pp_views(ref_azimuth=0.0, input_elevation=0.0):
    relative_azimuths = [30, 90, 150, 210, 270, 330]
    target_elevations = [20, -10, 20, -10, 20, -10]

    views = []

    views.append({
        "name": "cond",
        "azimuth": ref_azimuth,
        "elevation": input_elevation,
        "relative_azimuth": 0.0,
    })

    for i, (rel_az, elev) in enumerate(zip(relative_azimuths, target_elevations)):
        views.append({
            "name": f"target_{i:03d}",
            "azimuth": ref_azimuth + rel_az,
            "elevation": elev,
            "relative_azimuth": rel_az,
        })

    return views
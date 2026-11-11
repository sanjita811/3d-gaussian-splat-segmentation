from __future__ import annotations

import json
import math
import struct
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn.functional as F

from gaussian_scene import Camera


# ─────────────────────────────────────────────────────────────────────────────
#  cameras.json  (written by 3DGS trainer / viewer)
# ─────────────────────────────────────────────────────────────────────────────

def _load_cameras_json(json_path, device, max_cameras=200):
    with open(json_path) as f:
        data = json.load(f)
    cameras = []
    for entry in data[:max_cameras]:
        img_w = int(entry.get("width",  entry.get("w", 800)))
        img_h = int(entry.get("height", entry.get("h", 600)))
        fx = float(entry.get("fx", entry.get("fl_x", img_w * 0.8)))
        fy = float(entry.get("fy", entry.get("fl_y", img_h * 0.8)))
        cx = float(entry.get("cx", img_w / 2.0))
        cy = float(entry.get("cy", img_h / 2.0))

        c2w = np.array(entry["transform_matrix"], dtype=np.float32)
        R_c2w = c2w[:3, :3]
        t_c2w = c2w[:3, 3]
        R_w2c = R_c2w.T
        t_w2c = (-R_c2w.T @ t_c2w).astype(np.float32)

        cameras.append(Camera(
            R=torch.from_numpy(R_w2c), t=torch.from_numpy(t_w2c),
            fx=fx, fy=fy, cx=cx, cy=cy,
            width=img_w, height=img_h, device=device,
        ))
    return cameras


# ─────────────────────────────────────────────────────────────────────────────
#  COLMAP binary
# ─────────────────────────────────────────────────────────────────────────────

def _read_colmap_cameras_bin(path):
    cams = {}
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            cam_id   = struct.unpack("<I", f.read(4))[0]
            model_id = struct.unpack("<I", f.read(4))[0]
            w = struct.unpack("<Q", f.read(8))[0]
            h = struct.unpack("<Q", f.read(8))[0]
            np_ = {0:3, 1:4, 2:4, 3:5, 4:5, 5:8, 6:8, 7:12}.get(model_id, 4)
            params = struct.unpack(f"<{np_}d", f.read(8 * np_))
            cams[cam_id] = {"w":w, "h":h, "model_id":model_id, "params":params}
    return cams

def _read_colmap_images_bin(path):
    images = []
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            img_id = struct.unpack("<I", f.read(4))[0]
            qvec   = struct.unpack("<4d", f.read(32))
            tvec   = struct.unpack("<3d", f.read(24))
            cam_id = struct.unpack("<I", f.read(4))[0]
            name = b""
            while True:
                c = f.read(1)
                if c == b"\x00": break
                name += c
            num_pts = struct.unpack("<Q", f.read(8))[0]
            f.read(num_pts * 24)
            images.append({"qvec":qvec, "tvec":tvec, "cam_id":cam_id, "name":name.decode()})
    return images

def _qvec_to_rotmat(qvec):
    qw, qx, qy, qz = qvec
    return np.array([
        [1-2*(qy**2+qz**2),   2*(qx*qy-qw*qz),   2*(qx*qz+qw*qy)],
        [2*(qx*qy+qw*qz),   1-2*(qx**2+qz**2),   2*(qy*qz-qw*qx)],
        [2*(qx*qz-qw*qy),   2*(qy*qz+qw*qx),   1-2*(qx**2+qy**2)],
    ], dtype=np.float32)

def _load_colmap_cameras(sparse_dir, device, max_cameras=200,
                         target_long_edge=800):
    """Load COLMAP cameras with UNIFORM rescaling to preserve aspect ratio.

    CRITICAL: render_width/render_height are NOT accepted as separate params.
    Instead we use target_long_edge and derive render_w, render_h per-camera
    by uniformly scaling so the longest edge = target_long_edge.

    Why uniform scaling matters
    ---------------------------
    The previous code accepted render_width=800, render_height=600 and
    applied NON-UNIFORM scaling:
        sx = render_width  / orig_w
        sy = render_height / orig_h
    For a PORTRAIT camera (e.g. bonsai: 1520x2028):
        sx = 800/1520 = 0.526
        sy = 600/2028 = 0.296   ← 2x different!
    This stretches the camera frustum, making cam.project() return pixel
    coordinates that don't align with what the renderer produces.
    The segmentation mask and the Gaussian projections are in different
    coordinate spaces → lifting scores are computed on wrong pixels.

    With uniform scaling (sx == sy):
        scale = 800 / max(1520, 2028) = 800/2028 = 0.395
        render_w = round_even(1520 * 0.395) = 598
        render_h = round_even(2028 * 0.395) = 800
        sx = sy = 0.395 → fx, fy both scaled correctly
    """
    cb = sparse_dir / "cameras.bin"
    ib = sparse_dir / "images.bin"
    if not cb.exists() or not ib.exists():
        return []
    colmap_cams   = _read_colmap_cameras_bin(cb)
    colmap_images = _read_colmap_images_bin(ib)
    cameras = []
    for img in colmap_images[:max_cameras]:
        cc = colmap_cams[img["cam_id"]]
        p  = cc["params"]
        if cc["model_id"] == 0:   # SIMPLE_PINHOLE
            fx = fy = p[0]; cx, cy = p[1], p[2]
        elif cc["model_id"] == 2: # SIMPLE_RADIAL
            fx = fy = p[0]; cx, cy = p[1], p[2]
        else:                     # PINHOLE, OPENCV, FULL_OPENCV
            fx, fy = p[0], p[1]; cx, cy = p[2], p[3]

        orig_w, orig_h = int(cc["w"]), int(cc["h"])

        # Uniform scale: longest edge → target_long_edge
        scale  = target_long_edge / max(orig_w, orig_h)
        out_w  = int(orig_w * scale)
        out_h  = int(orig_h * scale)
        # Round to nearest even (required by some video/image encoders)
        out_w  = max(2, (out_w  // 2) * 2)
        out_h  = max(2, (out_h  // 2) * 2)
        # Recompute exact scale from rounded output dims
        sx     = out_w / orig_w
        sy     = out_h / orig_h

        fx = fx * sx;  fy = fy * sy
        cx = cx * sx;  cy = cy * sy

        R = _qvec_to_rotmat(img["qvec"])
        t = np.array(img["tvec"], dtype=np.float32)

        # Resolve original training image path.
        img_name  = img.get("name", "")
        scene_dir = sparse_dir.parent.parent if sparse_dir.name == "0" else sparse_dir.parent
        image_path = None
        for img_subdir in ["images_4", "images_2", "images"]:
            candidate = scene_dir / img_subdir / img_name
            if candidate.exists():
                image_path = candidate
                break

        cameras.append(Camera(
            R=torch.from_numpy(R), t=torch.from_numpy(t),
            fx=float(fx), fy=float(fy), cx=float(cx), cy=float(cy),
            width=out_w, height=out_h, device=device,
            image_path=image_path,
        ))
    return cameras


# ─────────────────────────────────────────────────────────────────────────────
#  NeRF-style transforms.json
# ─────────────────────────────────────────────────────────────────────────────

def _load_transforms_json(tf_path, device, max_cameras=200):
    with open(tf_path) as f:
        tf = json.load(f)
    # Prefer per-frame intrinsics; fall back to top-level fov
    frames = tf.get("frames", [])[:max_cameras]
    cameras = []
    for frame in frames:
        w   = int(frame.get("w", tf.get("w", 800)))
        h   = int(frame.get("h", tf.get("h", 600)))
        if "fl_x" in frame or "fl_x" in tf:
            fx = float(frame.get("fl_x", tf.get("fl_x")))
            fy = float(frame.get("fl_y", tf.get("fl_y", fx)))
        else:
            fov_x = float(tf.get("camera_angle_x", math.pi/3))
            fx = fy = 0.5 * w / math.tan(0.5 * fov_x)
        cx = float(frame.get("cx", tf.get("cx", w/2.0)))
        cy = float(frame.get("cy", tf.get("cy", h/2.0)))
        c2w    = np.array(frame["transform_matrix"], dtype=np.float32)
        R_c2w  = c2w[:3,:3]; t_c2w = c2w[:3,3]
        R_w2c  = R_c2w.T
        t_w2c  = (-R_c2w.T @ t_c2w).astype(np.float32)
        cameras.append(Camera(
            R=torch.from_numpy(R_w2c), t=torch.from_numpy(t_w2c),
            fx=fx, fy=fy, cx=cx, cy=cy,
            width=w, height=h, device=device,
        ))
    return cameras


# ─────────────────────────────────────────────────────────────────────────────
#  Public API
# ─────────────────────────────────────────────────────────────────────────────


def load_cameras(
    scene_dir,
    width: int = 800,
    height: int = 600,
    device=torch.device("cpu"),
    max_cameras: int = 200,
    target_long_edge: int = 800,
) -> List[Camera]:
    """Load cameras with UNIFORM aspect-ratio-preserving scaling.

    width/height kept for API compat but IGNORED for COLMAP paths.
    Render resolution is derived per-camera from original dimensions
    using uniform scaling so sx == sy and intrinsics stay valid.
    """
    scene_dir = Path(scene_dir)

    colmap_present = any(
        (scene_dir / sd / "cameras.bin").exists()
        for sd in ["sparse/0", "sparse", "colmap/sparse/0"]
    )
    if not colmap_present:
        for p in [scene_dir / "cameras.json",
                  scene_dir / "camera_calibration" / "cameras.json"]:
            if p.exists():
                cams = _load_cameras_json(p, device, max_cameras)
                if cams:
                    print(f"[camera_utils] Loaded {len(cams)} cameras from {p}")
                    return cams

    for sd_rel in ["sparse/0", "sparse", "colmap/sparse/0"]:
        sparse_dir_path = scene_dir / sd_rel
        cams = _load_colmap_cameras(
            sparse_dir_path, device, max_cameras,
            target_long_edge=target_long_edge,
        )
        if cams:
            n_with_img = sum(1 for c in cams if c.image_path is not None)
            s = cams[0]
            print(f"[camera_utils] Loaded {len(cams)} cameras from {sparse_dir_path} "
                  f"(render={s.width}x{s.height}, uniform scale, "
                  f"{n_with_img}/{len(cams)} have original image paths)")
            return cams

    for name in ["transforms_train.json", "transforms.json", "transforms_test.json"]:
        p = scene_dir / name
        if p.exists():
            cams = _load_transforms_json(p, device, max_cameras)
            if cams:
                print(f"[camera_utils] Loaded {len(cams)} cameras from {p}")
                return cams

    print(f"[camera_utils] WARNING: No camera files found in {scene_dir}")
    print(f"  Contents: {list(scene_dir.iterdir()) if scene_dir.exists() else 'DIR NOT FOUND'}")
    return []

def make_orbit_cameras(
    center: torch.Tensor,
    radius: float,
    num_frames: int,
    elevation_deg: float,
    fx: float, fy: float,
    width: int, height: int,
    device: torch.device,
) -> List[Camera]:
    """Orbit cameras around `center`. Uses standard OpenCV convention."""
    cameras = []
    elevation = math.radians(elevation_deg)
    # World up = -Y in OpenCV/3DGS convention (Y points down in camera space)
    world_up = torch.tensor([0.0, -1.0, 0.0], device=device, dtype=torch.float32)

    for i in range(num_frames):
        azimuth = 2 * math.pi * i / num_frames
        cam_pos = center.float() + torch.tensor([
            radius * math.cos(elevation) * math.cos(azimuth),
            -radius * math.sin(elevation),   # negative: up in world = -Y
            radius * math.cos(elevation) * math.sin(azimuth),
        ], device=device, dtype=torch.float32)

        forward = F.normalize(center.float() - cam_pos, dim=0)  # points toward scene
        right   = F.normalize(torch.cross(forward, world_up, dim=0), dim=0)
        up      = torch.cross(right, forward, dim=0)  # recompute for orthogonality

        # OpenCV R_w2c: rows are right, down, forward
        # "down" in camera space = -up in world space
        R_w2c = torch.stack([right, -up, forward], dim=0)  # (3,3)
        t_w2c = R_w2c @ (-cam_pos)

        cameras.append(Camera(
            R=R_w2c, t=t_w2c,
            fx=fx, fy=fy,
            cx=width/2.0, cy=height/2.0,
            width=width, height=height, device=device,
        ))
    return cameras
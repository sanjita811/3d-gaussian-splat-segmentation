from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from plyfile import PlyData
from rich.console import Console

console = Console()

# ─────────────────────────────────────────────────────────────────────────────
#  Spherical Harmonics helpers  (degree-0 DC only)
# ─────────────────────────────────────────────────────────────────────────────
SH_C0 = 0.28209479177387814   # 1 / (2√π)

def sh_to_rgb(sh_dc: torch.Tensor) -> torch.Tensor:
    return (sh_dc * SH_C0 + 0.5).clamp(0.0, 1.0)

def rgb_to_sh(rgb: torch.Tensor) -> torch.Tensor:
    return (rgb - 0.5) / SH_C0


# ─────────────────────────────────────────────────────────────────────────────
#  Camera
# ─────────────────────────────────────────────────────────────────────────────
class Camera:
    """Pinhole camera.  R, t are world→camera (OpenCV convention)."""

    def __init__(self, R, t, fx, fy, cx, cy, width, height, device, image_path=None):
        self.R = R.to(device)
        self.t = t.to(device)
        self.fx, self.fy = fx, fy
        self.cx, self.cy = cx, cy
        self.width  = width
        self.height = height
        self.device = device
        # Path to the original training image for this camera (used by the
        # segmenter instead of the 3DGS render — real photos give much better
        # GroundingDINO + SAM2 results than washed-out splat renders).
        self.image_path = image_path  # Path | None

    @property
    def world_center(self):
        return -(self.R.T @ self.t)

    def project(self, xyz_world: torch.Tensor):
        """(N,3) world → (N,2) pixel uv,  (N,) depth z."""
        xyz_cam = (xyz_world @ self.R.T) + self.t   # (N,3)
        z  = xyz_cam[:, 2].clamp(min=1e-6)
        u  = self.fx * xyz_cam[:, 0] / z + self.cx
        v  = self.fy * xyz_cam[:, 1] / z + self.cy
        return torch.stack([u, v], dim=-1), z


# ─────────────────────────────────────────────────────────────────────────────
#  GaussianScene
# ─────────────────────────────────────────────────────────────────────────────
class GaussianScene:

    def __init__(self, ply_path: Path, device: torch.device):
        self.device   = device
        self.ply_path = Path(ply_path)
        self._load_ply()

    def _load_ply(self):
        console.log(f"Loading scene: [cyan]{self.ply_path}[/cyan]")
        ply    = PlyData.read(str(self.ply_path))
        vertex = ply["vertex"]

        def col(name):
            return torch.tensor(np.array(vertex[name]), dtype=torch.float32)

        self.xyz     = torch.stack([col("x"), col("y"), col("z")], dim=-1).to(self.device)
        N            = self.xyz.shape[0]
        self.opacity = torch.sigmoid(col("opacity")).to(self.device)

        scales      = torch.stack([col(f"scale_{i}") for i in range(3)], dim=-1)
        self.scales = torch.exp(scales).to(self.device)
        # No scale clamping. Mip-NeRF 360 scenes (bicycle, bonsai, etc.) contain
        # large-scale Gaussians for branches, spokes, and background planes that are
        # valid geometry. Clamping them collapses their alpha contribution and breaks
        # the multi-view lifting step (Gaussians never accumulate enough score to be
        # segmented). Log the scale range for diagnostics only.
        scale_max_val = self.scales.max().item()
        console.log(f"  Scale range: max={scale_max_val:.3f}")

        # FIX: quaternion convention — official 3DGS PLY stores (w,x,y,z)
        # as rot_0..rot_3, which matches our _quat_to_rotmat expectation.
        self.quats = F.normalize(
            torch.stack([col(f"rot_{i}") for i in range(4)], dim=-1), dim=-1
        ).to(self.device)

        sh_names = sorted([p.name for p in vertex.properties if p.name.startswith("f_dc")])
        if sh_names:
            sh_dc = torch.stack([col(n) for n in sh_names], dim=-1)
        else:
            try:
                sh_dc = torch.stack([col("red"), col("green"), col("blue")], dim=-1).float() / 255.0
                sh_dc = rgb_to_sh(sh_dc)
            except Exception:
                sh_dc = torch.zeros(N, 3)
        self.sh_dc = sh_dc.to(self.device)

        sh_rest_names = sorted([p.name for p in vertex.properties if p.name.startswith("f_rest")])
        self.sh_rest  = (
            torch.stack([col(n) for n in sh_rest_names], dim=-1).to(self.device)
            if sh_rest_names else None
        )

        console.log(f"  Loaded [bold]{N:,}[/bold] Gaussians")
        self.seg_mask = torch.zeros(N, dtype=torch.bool, device=self.device)

    @property
    def num_gaussians(self): return self.xyz.shape[0]

    @property
    def rgb(self): return sh_to_rgb(self.sh_dc)

    def recolor(self, rgb: tuple):
        if not self.seg_mask.any():
            console.log("[yellow]No Gaussians segmented – recolor skipped[/yellow]")
            return
        target = torch.tensor(rgb, dtype=torch.float32, device=self.device)
        self.sh_dc[self.seg_mask] = rgb_to_sh(target).unsqueeze(0)
        if self.sh_rest is not None:
            self.sh_rest[self.seg_mask] = 0.0
        console.log(f"  Recolored {self.seg_mask.sum().item():,} Gaussians → RGB{rgb}")

    def remove_segmented(self):
        keep      = ~self.seg_mask
        n_removed = self.seg_mask.sum().item()
        self.xyz     = self.xyz[keep]
        self.opacity = self.opacity[keep]
        self.scales  = self.scales[keep]
        self.quats   = self.quats[keep]
        self.sh_dc   = self.sh_dc[keep]
        if self.sh_rest is not None:
            self.sh_rest = self.sh_rest[keep]
        self.seg_mask = self.seg_mask[keep]
        console.log(f"  Removed {n_removed:,} Gaussians – {self.num_gaussians:,} remaining")

    def save_ply(self, out_path: Path):
        from plyfile import PlyElement, PlyData as PD
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        xyz           = self.xyz.cpu().numpy()
        opacity_logit = torch.logit(self.opacity.clamp(1e-6, 1-1e-6)).cpu().numpy()
        scales_log    = torch.log(self.scales.clamp(min=1e-8)).cpu().numpy()
        quats         = self.quats.cpu().numpy()
        sh_dc         = self.sh_dc.cpu().numpy()
        dtype_list = [
            ("x","f4"),("y","f4"),("z","f4"),("opacity","f4"),
            ("scale_0","f4"),("scale_1","f4"),("scale_2","f4"),
            ("rot_0","f4"),("rot_1","f4"),("rot_2","f4"),("rot_3","f4"),
            ("f_dc_0","f4"),("f_dc_1","f4"),("f_dc_2","f4"),
        ]
        N   = xyz.shape[0]
        arr = np.empty(N, dtype=dtype_list)
        arr["x"], arr["y"], arr["z"] = xyz[:,0], xyz[:,1], xyz[:,2]
        arr["opacity"] = opacity_logit
        for i in range(3): arr[f"scale_{i}"] = scales_log[:,i]
        for i in range(4): arr[f"rot_{i}"]   = quats[:,i]
        for i in range(3): arr[f"f_dc_{i}"]  = sh_dc[:,i]
        PD([PlyElement.describe(arr, "vertex")]).write(str(out_path))
        console.log(f"  Saved → [cyan]{out_path}[/cyan]")


# ─────────────────────────────────────────────────────────────────────────────
#  Rasteriser helpers
# ─────────────────────────────────────────────────────────────────────────────

def _quat_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    """(N,4) wxyz quaternion → (N,3,3) rotation matrices.

    FIX: original code called .reshape(-1,3,3) on a flat stack of 9 scalars
    per Gaussian which transposes the matrix (row-major ≠ column-major).
    We build each row explicitly to be unambiguous.
    """
    w, x, y, z = q[:,0], q[:,1], q[:,2], q[:,3]
    # Row 0
    r00 = 1 - 2*(y*y + z*z);  r01 = 2*(x*y - w*z);  r02 = 2*(x*z + w*y)
    # Row 1
    r10 = 2*(x*y + w*z);       r11 = 1 - 2*(x*x + z*z); r12 = 2*(y*z - w*x)
    # Row 2
    r20 = 2*(x*z - w*y);       r21 = 2*(y*z + w*x);  r22 = 1 - 2*(x*x + y*y)
    # Stack rows first, then reshape → correct (N,3,3) layout
    row0 = torch.stack([r00, r01, r02], dim=-1)   # (N,3)
    row1 = torch.stack([r10, r11, r12], dim=-1)
    row2 = torch.stack([r20, r21, r22], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)  # (N,3,3)


def _world_covariance(scales: torch.Tensor, quats: torch.Tensor) -> torch.Tensor:
    """3D world-space covariance:  Σ_world = R_g S Sᵀ R_gᵀ   shape (N,3,3)."""
    R_g = _quat_to_rotmat(quats)       # (N,3,3)
    S   = torch.diag_embed(scales)     # (N,3,3)
    RS  = R_g @ S                      # (N,3,3)
    return RS @ RS.transpose(-1, -2)   # (N,3,3)


# ─────────────────────────────────────────────────────────────────────────────
#  render_view
# ─────────────────────────────────────────────────────────────────────────────

def render_view(
    scene: GaussianScene,
    cam: Camera,
    background: Optional[torch.Tensor] = None,
    max_gaussians: int = 100_000,
    tile_size: int = 16,
    **kwargs,
) -> dict:
    """Tile-based alpha-compositing rasteriser.  MPS / CPU compatible.

    Returns dict with keys "rgb" (H,W,3), "depth" (H,W), "alpha_score" (N,).
    """
    device = scene.device
    H, W   = cam.height, cam.width

    if background is None:
        background = torch.ones(3, device=device)

    # ── 1. Project & cull ────────────────────────────────────────────────────
    uv, z = cam.project(scene.xyz)          # (N,2), (N,)

    margin = max(H, W) * 0.5
    valid  = (
        (z > 0.1)
        & (uv[:,0] > -margin) & (uv[:,0] < W + margin)
        & (uv[:,1] > -margin) & (uv[:,1] < H + margin)
    )

    if valid.sum() > max_gaussians:
        opac_tmp = scene.opacity.clone()
        opac_tmp[~valid] = -1.0
        _, top_idx = opac_tmp.topk(max_gaussians)
        mask2 = torch.zeros(scene.num_gaussians, dtype=torch.bool, device=device)
        mask2[top_idx] = True
        valid = valid & mask2

    idx      = torch.where(valid)[0]
    uv_v     = uv[idx]
    z_v      = z[idx]
    opac_v   = scene.opacity[idx]
    rgb_v    = scene.rgb[idx]
    scales_v = scene.scales[idx]
    quats_v  = scene.quats[idx]
    M        = idx.shape[0]

    if M == 0:
        return {
            "rgb":         background.view(1,1,3).expand(H,W,3).clone(),
            "depth":       torch.zeros(H, W, device=device),
            "alpha_score": torch.zeros(scene.num_gaussians, device=device),
        }

    # ── 2. 2D covariance (EWA splatting) ─────────────────────────────────────
    # Camera-space positions (needed for correct Jacobian)
    xyz_cam_v = (scene.xyz[idx] @ cam.R.T) + cam.t   # (M,3)
    xc = xyz_cam_v[:,0]   # (M,)
    yc = xyz_cam_v[:,1]
    # z_v == xyz_cam_v[:,2]  (already computed above, same values)

    # World-space 3D covariance
    Sigma_world = _world_covariance(scales_v, quats_v)   # (M,3,3)

    # Rotate into camera space:  Σ_cam = R_cam Σ_world R_camᵀ
    R  = cam.R.unsqueeze(0)                              # (1,3,3)
    Sigma_cam = R @ Sigma_world @ R.transpose(-1,-2)     # (M,3,3)

    # EWA perspective Jacobian (Zwicker et al. 2002, eq. 29):
    #   J = [[fx/z,  0,      -fx·x/z²],
    #        [0,     fy/z,   -fy·y/z²]]
    fx, fy = cam.fx, cam.fy
    z2 = z_v * z_v
    J  = torch.zeros(M, 2, 3, device=device)
    J[:,0,0] =  fx / z_v
    J[:,0,2] = -fx * xc / z2
    J[:,1,1] =  fy / z_v
    J[:,1,2] = -fy * yc / z2

    Sigma2D = J @ Sigma_cam @ J.transpose(-1,-2)     # (M,2,2)
    Sigma2D[:,0,0] += 0.3                             # low-pass filter (prevent aliasing)
    Sigma2D[:,1,1] += 0.3

    cov_xx = Sigma2D[:,0,0]
    cov_xy = Sigma2D[:,0,1]
    cov_yy = Sigma2D[:,1,1]

    det     = (cov_xx * cov_yy - cov_xy**2).clamp(min=1e-6)
    inv_det = 1.0 / det
    inv_xx  =  cov_yy * inv_det
    inv_xy  = -cov_xy * inv_det
    inv_yy  =  cov_xx * inv_det

    # Per-Gaussian screen-space bounding radius (3σ)
    trace   = cov_xx + cov_yy
    lambda1 = trace/2 + torch.sqrt(((trace/2)**2 - det).clamp(min=0))
    radii   = (3.0 * lambda1.sqrt()).ceil().int().clamp(1, max(H,W)//2)

    # ── 3. Depth sort (back-to-front) ────────────────────────────────────────
    order    = torch.argsort(z_v, descending=True)
    uv_v     = uv_v[order]
    opac_v   = opac_v[order]
    rgb_v    = rgb_v[order]
    inv_xx   = inv_xx[order]
    inv_xy   = inv_xy[order]
    inv_yy   = inv_yy[order]
    radii    = radii[order]
    orig_idx = idx[order]

    ux = uv_v[:,0]
    uy = uv_v[:,1]

    # ── 4. Tile rasterisation ─────────────────────────────────────────────────
    T     = tile_size
    n_ty  = math.ceil(H / T)
    n_tx  = math.ceil(W / T)

    r_np  = radii.cpu().numpy()
    ux_np = ux.cpu().numpy()
    uy_np = uy.cpu().numpy()

    canvas_rgb        = torch.zeros(H, W, 3, device=device)
    transmittance     = torch.ones(H, W,    device=device)
    gauss_alpha_score = torch.zeros(scene.num_gaussians, device=device)

    for ty in range(n_ty):
        y0 = ty * T;  y1 = min(y0 + T, H)
        for tx in range(n_tx):
            x0 = tx * T;  x1 = min(x0 + T, W)

            # Gaussians whose bbox overlaps this tile
            overlap = (
                (ux_np + r_np >= x0) & (ux_np - r_np < x1)
                & (uy_np + r_np >= y0) & (uy_np - r_np < y1)
            )
            if not overlap.any():
                continue

            t_idx  = torch.from_numpy(overlap.nonzero()[0]).to(device)
            K      = t_idx.shape[0]
            t_ux   = ux[t_idx]
            t_uy   = uy[t_idx]
            t_opac = opac_v[t_idx]
            t_rgb  = rgb_v[t_idx]
            t_ixx  = inv_xx[t_idx]
            t_ixy  = inv_xy[t_idx]
            t_iyy  = inv_yy[t_idx]

            bh = y1 - y0;  bw = x1 - x0

            # FIX: use .view() with explicit shapes to avoid non-contiguous broadcast
            px = torch.arange(x0, x1, device=device, dtype=torch.float32)  # (bw,)
            py = torch.arange(y0, y1, device=device, dtype=torch.float32)  # (bh,)

            dx = px.view(1, 1, bw) - t_ux.view(K, 1, 1)   # (K, 1, bw)
            dy = py.view(1, bh, 1) - t_uy.view(K, 1, 1)   # (K, bh, 1)
            # Both broadcast correctly to (K, bh, bw)

            maha = (
                t_ixx.view(K,1,1) * dx * dx
                + 2.0 * t_ixy.view(K,1,1) * dx * dy
                + t_iyy.view(K,1,1) * dy * dy
            )   # (K, bh, bw)

            g_alpha = t_opac.view(K,1,1) * torch.exp(-0.5*maha) * (maha < 8.0)
            # (K, bh, bw)

            # Vectorised alpha-compositing within tile:
            # T_before[k] = Π_{j<k}(1 - α_j)  for each pixel
            # Use cumprod then shift right by 1.
            one_minus = (1.0 - g_alpha)                      # (K, bh, bw)
            cumprod   = torch.cumprod(one_minus, dim=0)       # (K, bh, bw)
            T_before  = torch.cat([
                torch.ones(1, bh, bw, device=device),
                cumprod[:-1]
            ], dim=0)                                          # (K, bh, bw)

            weights = g_alpha * T_before                       # (K, bh, bw)

            # Colour accumulation: Σ_k w_k · rgb_k
            # rgb: (K,3) → (K,1,1,3); weights: (K,bh,bw,1)
            colour_contrib = (weights.unsqueeze(-1) * t_rgb.view(K,1,1,3)).sum(dim=0)
            canvas_rgb[y0:y1, x0:x1] += colour_contrib        # (bh,bw,3)

            # Residual transmittance for this tile
            T_final = cumprod[-1]                              # (bh, bw)
            transmittance[y0:y1, x0:x1] *= T_final

            # Alpha score for mask lifting: max weight per Gaussian
            peak_per_gauss = weights.view(K, -1).max(dim=1).values  # (K,)
            gauss_alpha_score.scatter_reduce_(
                0, orig_idx[t_idx], peak_per_gauss, reduce="amax", include_self=True
            )

    final_rgb = canvas_rgb + transmittance.unsqueeze(-1) * background

    return {
        "rgb":         final_rgb.clamp(0.0, 1.0),
        "depth":       1.0 - transmittance,
        "alpha_score": gauss_alpha_score,
    }
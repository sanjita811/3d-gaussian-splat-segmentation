"""
mask_lifting.py
───────────────
Multi-view mask lifting: accumulates per-Gaussian segmentation scores across
N rendered views using alpha-weighted confidence fusion.

Core algorithm
──────────────
For each view v:
  1. Render the scene → per-Gaussian alpha score map A_v (shape N)
     where A_v[i] = max pixel alpha contribution of Gaussian i in this view
  2. Run 2D segmenter → binary mask M_v (H, W)
  3. Render → alpha-weighted mask score:
       For each Gaussian i:
         contribution_v[i] = A_v[i] * M_v[projected_pixel_of_i]
         (i.e. how much does Gaussian i contribute to *masked* pixels)
4. Accumulate:
     score[i] += contribution_v[i]
     views_hit[i] += (A_v[i] > alpha_min)   ← did this Gaussian appear in this view?

After all views:
  Normalise score by views_hit (avoid bias toward Gaussians visible in many views).
  Apply threshold:  seg_mask[i] = (score_norm[i] > threshold) & (views_hit[i] >= min_views)
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch
from rich.console import Console
from rich.progress import track

from gaussian_scene import GaussianScene, render_view
from camera_utils import Camera

console = Console()


# ──────────────────────────────────────────────────────────────────────────────
#  Per-view alpha-weighted mask score
# ──────────────────────────────────────────────────────────────────────────────

def _alpha_weighted_mask_score(
    scene: GaussianScene,
    cam: Camera,
    mask_2d: np.ndarray,     # (H, W) bool
    alpha_min: float = 0.01,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns
        contribution (N,) – alpha-weighted mask score per Gaussian
        appeared     (N,) bool – Gaussians visible in this view above alpha_min

    FIX: H, W are now read from mask_2d.shape (the actual rendered image size)
    rather than cam.height / cam.width. After the camera_utils fix, both are
    identical (COLMAP intrinsics are rescaled to render resolution), but reading
    from the mask guarantees they can never diverge even if the camera objects
    were constructed elsewhere.
    """
    device = scene.device

    # FIX: derive H, W from the actual mask, not from cam attributes.
    # cam.height/cam.width now always equal the render resolution (after the
    # _load_colmap_cameras rescaling fix), but reading from mask_2d is the
    # ground truth — it is the actual image the segmenter operated on.
    H, W = mask_2d.shape[:2]

    # Render to get alpha_score per Gaussian
    result = render_view(scene, cam, return_alpha_map=True)
    alpha_score = result["alpha_score"]   # (N,)  max pixel alpha contribution

    # Project Gaussian centres to 2D to look up mask value
    uv, z = cam.project(scene.xyz)        # (N, 2), (N,)

    # Round to integer pixel coords
    px = uv[:, 0].round().long()           # (N,)
    py = uv[:, 1].round().long()           # (N,)

    # Mask tensor on device
    mask_t = torch.from_numpy(mask_2d.astype(np.float32)).to(device)  # (H, W)

    # For each Gaussian, look up whether its projected pixel is in the mask.
    # Clamp to valid range; out-of-image Gaussians get 0.
    valid = (z > 0) & (px >= 0) & (px < W) & (py >= 0) & (py < H)
    px_safe = px.clamp(0, W - 1)
    py_safe = py.clamp(0, H - 1)

    pixel_mask_val = mask_t[py_safe, px_safe]         # (N,)
    pixel_mask_val[~valid] = 0.0

    # Weighted contribution: alpha * mask
    contribution = alpha_score * pixel_mask_val        # (N,)
    appeared     = alpha_score > alpha_min             # (N,) bool

    return contribution, appeared


# ──────────────────────────────────────────────────────────────────────────────
#  Multi-view lifting
# ──────────────────────────────────────────────────────────────────────────────

class MultiViewLifter:
    """
    Accumulates per-Gaussian segmentation evidence across multiple views.

    Usage:
        lifter = MultiViewLifter(scene, cfg)
        for cam, mask in zip(cameras, masks):
            lifter.accumulate(cam, mask)
        lifter.threshold()   # writes scene.seg_mask
    """

    def __init__(self, scene: GaussianScene, cfg: dict):
        self.scene     = scene
        self.cfg       = cfg
        self.device    = scene.device
        N = scene.num_gaussians

        self.score_sum  = torch.zeros(N, device=self.device)
        self.views_hit  = torch.zeros(N, dtype=torch.long, device=self.device)
        self.n_views    = 0

    def accumulate(self, cam: Camera, mask_2d: np.ndarray, seg_score: float = 1.0):
        """Add evidence from one view.

        Args:
            cam:       Camera for this view.
            mask_2d:   (H, W) bool – segmented pixels in the render.
            seg_score: Confidence score from the 2D segmenter (scales contribution).
        """
        alpha_min = self.cfg["lifting"]["alpha_min"]
        contribution, appeared = _alpha_weighted_mask_score(
            self.scene, cam, mask_2d, alpha_min
        )
        self.score_sum  += contribution * seg_score
        self.views_hit  += appeared.long()
        self.n_views    += 1

    def compute_scores(self) -> torch.Tensor:
        """Return normalised per-Gaussian score in [0, 1]."""
        # Normalise by number of views in which each Gaussian appeared
        denom = self.views_hit.float().clamp(min=1.0)
        score_norm = self.score_sum / denom
        return score_norm

    def threshold(self) -> torch.Tensor:
        """Apply threshold and min-views filter → write scene.seg_mask.

        Returns the binary mask (N,) bool.
        """
        thresh     = self.cfg["lifting"]["score_threshold"]
        min_views  = self.cfg["lifting"]["min_views_hit"]

        score_norm = self.compute_scores()
        seg_mask   = (score_norm > thresh) & (self.views_hit >= min_views)

        self.scene.seg_mask = seg_mask
        n_seg = seg_mask.sum().item()
        frac  = 100 * n_seg / self.scene.num_gaussians
        console.log(
            f"  Segmented [bold]{n_seg:,}[/bold] / {self.scene.num_gaussians:,} "
            f"Gaussians ({frac:.1f}%)  [threshold={thresh}, min_views={min_views}]"
        )
        return seg_mask


# ──────────────────────────────────────────────────────────────────────────────
#  Single-view baseline  (for ablation)
# ──────────────────────────────────────────────────────────────────────────────

def single_view_lift(
    scene: GaussianScene,
    cam: Camera,
    mask_2d: np.ndarray,
    cfg: dict,
) -> torch.Tensor:
    """
    Ablation baseline: lift segmentation from a single view only.
    Projects each Gaussian centre to 2D and checks if it falls in the mask.
    No alpha weighting.
    """
    device = scene.device
    # FIX: read H, W from mask, not cam
    H, W = mask_2d.shape[:2]

    uv, z = cam.project(scene.xyz)
    px = uv[:, 0].round().long()
    py = uv[:, 1].round().long()
    valid = (z > 0) & (px >= 0) & (px < W) & (py >= 0) & (py < H)

    mask_t = torch.from_numpy(mask_2d.astype(bool)).to(device)
    px_safe = px.clamp(0, W - 1)
    py_safe = py.clamp(0, H - 1)

    in_mask = mask_t[py_safe, px_safe] & valid
    scene.seg_mask = in_mask

    n_seg = in_mask.sum().item()
    console.log(f"  [Baseline] Single-view lift: {n_seg:,} Gaussians segmented")
    return in_mask


# ──────────────────────────────────────────────────────────────────────────────
#  Full pipeline helper
# ──────────────────────────────────────────────────────────────────────────────

def run_lifting_pipeline(
    scene:      GaussianScene,
    cameras:    List[Camera],
    segmenter,              # GroundedSAM2 or SAM3Segmenter
    prompt:     str,
    cfg:        dict,
    render_dir: "Optional[Path]" = None,  # save renders + masks if set
) -> torch.Tensor:
    """
    Full multi-view lifting pipeline.

    Renders each camera, runs 2D segmentation, accumulates lifting scores.
    Returns the final binary seg_mask (N,) written to scene.seg_mask.
    """
    from pathlib import Path
    import imageio

    lifter = MultiViewLifter(scene, cfg)
    max_views = cfg["render"].get("num_views", len(cameras))
    cameras   = cameras[:max_views]

    background = torch.tensor(cfg["render"]["background"], device=scene.device)

    for view_idx, cam in enumerate(track(cameras, description="Lifting views")):
        # Render at cam resolution (which now equals render_width x render_height
        # after the camera_utils fix)
        result = render_view(scene, cam, background=background)
        rgb_np = (result["rgb"].cpu().numpy() * 255).astype(np.uint8)

        # 2D segmentation — mask shape matches rgb_np shape
        mask, score = segmenter.segment(rgb_np, prompt)

        # Optionally save renders and masks
        if render_dir is not None:
            rdir = Path(render_dir)
            rdir.mkdir(parents=True, exist_ok=True)
            imageio.imwrite(str(rdir / f"render_{view_idx:04d}.png"), rgb_np)
            mask_vis = (mask * 255).astype(np.uint8)
            imageio.imwrite(str(rdir / f"mask_{view_idx:04d}.png"), mask_vis)

        # Accumulate — seg_score floored at 0.1 so zero-confidence views still
        # contribute a small signal (avoids missing the object on hard frames)
        lifter.accumulate(cam, mask, seg_score=max(score, 0.1))

    seg_mask = lifter.threshold()
    return seg_mask

"""
debug_seg.py
───────────────
Diagnostic tool for the renderer. Run from the project root:

    python scripts/debug_seg.py --scene bicycle

It prints a full diagnostic of every stage and saves:
  - debug_raw_render.png   : what the rasteriser actually produces
  - debug_depth.png        : depth map (should show scene structure)
  - debug_gaussians.png    : 2D scatter of projected Gaussian centres (first camera)
  - debug_stats.txt        : all numerical stats in one file

Inspect debug_stats.txt and the images if a render looks wrong.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

# ── make sure project scripts are importable ──────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from gaussian_scene import GaussianScene, render_view, sh_to_rgb
from camera_utils import load_cameras

# ─────────────────────────────────────────────────────────────────────────────

def save_png(tensor_hwc: torch.Tensor, path: Path):
    """Save (H,W,3) float [0,1] tensor as PNG."""
    try:
        from PIL import Image
        arr = (tensor_hwc.clamp(0,1).cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(arr).save(str(path))
        print(f"  Saved → {path}")
    except ImportError:
        print("  [PIL not installed – skipping PNG save, install Pillow]")

def save_depth_png(tensor_hw: torch.Tensor, path: Path):
    try:
        from PIL import Image
        d = tensor_hw.cpu().numpy()
        d = (d - d.min()) / (d.max() - d.min() + 1e-8)
        arr = (d * 255).astype(np.uint8)
        Image.fromarray(arr).save(str(path))
        print(f"  Saved → {path}")
    except ImportError:
        pass

def banner(title):
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print('─'*60)

# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene",   default="bicycle")
    ap.add_argument("--config",  default="configs/default.yaml")
    ap.add_argument("--max_g",   type=int, default=50_000,
                    help="max Gaussians passed to renderer")
    ap.add_argument("--out_dir", default="debug_out")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        cfg_path = SCRIPT_DIR.parent / args.config
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    out_dir = Path(args.out_dir);  out_dir.mkdir(exist_ok=True)
    stats_lines = []

    def log(msg):
        print(msg)
        stats_lines.append(str(msg))

    # ── Device ────────────────────────────────────────────────────────────────
    d = cfg.get("device", "cpu")
    if d == "mps" and not torch.backends.mps.is_available():
        d = "cpu"
    device = torch.device(d)
    log(f"Device: {device}")

    # ── Scene loading ─────────────────────────────────────────────────────────
    banner("1. PLY Loading")
    scene_dir = Path(cfg["scenes_dir"]) / args.scene
    candidates = [
        scene_dir / "point_cloud" / "iteration_30000" / "point_cloud.ply",
        scene_dir / "point_cloud" / "iteration_7000"  / "point_cloud.ply",
        *sorted(scene_dir.rglob("*.ply")),
    ]
    ply_path = next((p for p in candidates if p.exists()), None)
    if ply_path is None:
        print(f"ERROR: no .ply found under {scene_dir}")
        sys.exit(1)
    log(f"PLY path : {ply_path}")

    scene = GaussianScene(ply_path, device)
    N = scene.num_gaussians
    log(f"Gaussians: {N:,}")

    # ── PLY attribute stats ───────────────────────────────────────────────────
    banner("2. Attribute Statistics")

    xyz = scene.xyz.cpu()
    log(f"xyz range  x: [{xyz[:,0].min():.3f}, {xyz[:,0].max():.3f}]")
    log(f"           y: [{xyz[:,1].min():.3f}, {xyz[:,1].max():.3f}]")
    log(f"           z: [{xyz[:,2].min():.3f}, {xyz[:,2].max():.3f}]")

    scales = scene.scales.cpu()
    log(f"\nscales (exp'd)  min={scales.min():.5f}  max={scales.max():.3f}"
        f"  mean={scales.mean():.5f}  median={scales.median():.5f}")
    log(f"  99th pct: {scales.view(-1).kthvalue(int(0.99*scales.numel())).values.item():.4f}")
    log(f"  Gaussians with ANY scale > 1.0  : {(scales > 1.0).any(dim=1).sum().item():,}")
    log(f"  Gaussians with ANY scale > 0.5  : {(scales > 0.5).any(dim=1).sum().item():,}")
    log(f"  Gaussians with ALL scales < 0.01: {(scales < 0.01).all(dim=1).sum().item():,}")

    # Raw log-scale values (as stored in PLY before exp)
    import math
    log(f"\n  Raw scale_log stats (before exp):")
    raw_scales_log = torch.log(scales.clamp(min=1e-8))
    log(f"    min={raw_scales_log.min():.3f}  max={raw_scales_log.max():.3f}"
        f"  mean={raw_scales_log.mean():.3f}")

    opac = scene.opacity.cpu()
    log(f"\nopacity  min={opac.min():.4f}  max={opac.max():.4f}"
        f"  mean={opac.mean():.4f}  >0.5: {(opac>0.5).sum().item():,}")

    rgb_v = scene.rgb.cpu()
    log(f"\nrgb (from SH DC)  min={rgb_v.min():.3f}  max={rgb_v.max():.3f}"
        f"  mean={rgb_v.mean():.3f}")
    log(f"  sh_dc raw: min={scene.sh_dc.min():.3f}  max={scene.sh_dc.max():.3f}")

    q = scene.quats.cpu()
    log(f"\nquats  norm_min={q.norm(dim=1).min():.5f}  norm_max={q.norm(dim=1).max():.5f}")
    log(f"  w-component: min={q[:,0].min():.3f}  max={q[:,0].max():.3f}")

    # ── Camera loading ────────────────────────────────────────────────────────
    banner("3. Camera")
    cameras = load_cameras(scene_dir, device=device)
    cam = cameras[0]
    log(f"Num cameras loaded : {len(cameras)}")
    log(f"Image size         : {cam.width} × {cam.height}")
    log(f"fx={cam.fx:.2f}  fy={cam.fy:.2f}  cx={cam.cx:.2f}  cy={cam.cy:.2f}")
    log(f"Camera world pos   : {cam.world_center.cpu().numpy()}")
    log(f"R[0]               : {cam.R[0].cpu().numpy()}")
    log(f"t                  : {cam.t.cpu().numpy()}")

    # ── Projection check ──────────────────────────────────────────────────────
    banner("4. Projection Sanity")
    uv, z = cam.project(scene.xyz)
    log(f"Depth range   : min={z.min():.3f}  max={z.max():.3f}  mean={z.mean():.3f}")
    in_front = (z > 0.1)
    in_frame = (uv[:,0] > 0) & (uv[:,0] < cam.width) & \
               (uv[:,1] > 0) & (uv[:,1] < cam.height)
    log(f"In front of cam     : {in_front.sum().item():,} / {N:,}")
    log(f"Projecting into frame: {(in_front & in_frame).sum().item():,} / {N:,}")

    # Save projected Gaussian scatter (sample 5000)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        vis_mask = in_front & in_frame
        uv_vis   = uv[vis_mask].cpu().numpy()
        sample   = np.random.choice(len(uv_vis), min(5000, len(uv_vis)), replace=False)
        fig, ax  = plt.subplots(figsize=(8, 6))
        ax.scatter(uv_vis[sample,0], uv_vis[sample,1], s=1, alpha=0.3, c="blue")
        ax.set_xlim(0, cam.width);  ax.set_ylim(cam.height, 0)
        ax.set_title(f"Projected Gaussian centres (sample {len(sample)})")
        ax.set_xlabel("u (pixels)");  ax.set_ylabel("v (pixels)")
        p = out_dir / "debug_gaussians.png"
        fig.savefig(str(p), dpi=100, bbox_inches="tight")
        plt.close(fig)
        log(f"\nSaved Gaussian scatter → {p}")
    except ImportError:
        log("  [matplotlib not installed – skipping scatter plot]")

    # ── 2D covariance check ───────────────────────────────────────────────────
    banner("5. 2D Covariance / Radius Check")
    from gaussian_scene import _world_covariance, _quat_to_rotmat

    sample_idx = torch.randperm(min(N, 1000))[:200].to(device)
    s_s = scene.scales[sample_idx]
    q_s = scene.quats[sample_idx]
    Sw  = _world_covariance(s_s, q_s)

    R   = cam.R.unsqueeze(0)
    Sc  = R @ Sw @ R.transpose(-1,-2)

    z_s = z[sample_idx]
    xc  = (scene.xyz[sample_idx] @ cam.R.T + cam.t)[:,0]
    yc  = (scene.xyz[sample_idx] @ cam.R.T + cam.t)[:,1]
    fx, fy = cam.fx, cam.fy
    z2  = z_s * z_s
    J   = torch.zeros(len(sample_idx), 2, 3, device=device)
    J[:,0,0] =  fx / z_s;  J[:,0,2] = -fx * xc / z2
    J[:,1,1] =  fy / z_s;  J[:,1,2] = -fy * yc / z2

    S2  = J @ Sc @ J.transpose(-1,-2)
    S2[:,0,0] += 0.3;  S2[:,1,1] += 0.3

    cov_xx = S2[:,0,0];  cov_yy = S2[:,1,1]
    det    = (cov_xx*cov_yy - S2[:,0,1]**2).clamp(min=1e-6)
    trace  = cov_xx + cov_yy
    lam1   = trace/2 + torch.sqrt(((trace/2)**2 - det).clamp(min=0))
    radii  = (3.0 * lam1.sqrt()).cpu()

    log(f"Screen-space radii (sample of 200 in-front Gaussians):")
    log(f"  min={radii.min():.2f}  max={radii.max():.2f}"
        f"  mean={radii.mean():.2f}  median={radii.median():.2f}")
    log(f"  Radii > 100px : {(radii > 100).sum().item()}")
    log(f"  Radii > 50px  : {(radii > 50).sum().item()}")
    log(f"  Radii > 10px  : {(radii > 10).sum().item()}")
    if radii.max() > 200:
        log("  *** WARNING: some Gaussians have enormous screen radii.")
        log("  *** This usually means scale values are too large (not log-space in PLY?)")
        log("  *** or the camera is very close to the scene origin.")

    # ── Render ────────────────────────────────────────────────────────────────
    banner("6. Test Render")
    bg = torch.ones(3, device=device)
    import time
    t0 = time.time()
    result = render_view(scene, cam, background=bg, max_gaussians=args.max_g)
    elapsed = time.time() - t0
    log(f"Render time : {elapsed:.2f}s  (max_gaussians={args.max_g})")

    img = result["rgb"]
    dep = result["depth"]
    asc = result["alpha_score"]

    log(f"Output rgb  : min={img.min():.4f}  max={img.max():.4f}  mean={img.mean():.4f}")
    log(f"Depth       : min={dep.min():.4f}  max={dep.max():.4f}  mean={dep.mean():.4f}")
    log(f"alpha_score : nonzero={( asc > 0).sum().item():,}  max={asc.max():.4f}")

    if img.mean() > 0.95:
        log("  *** WARNING: render is nearly all white – most Gaussians may be")
        log("  *** outside the frame or have near-zero opacity/scale.")
    if img.mean() < 0.05:
        log("  *** WARNING: render is nearly all black – opacity or RGB may be wrong.")
    if dep.max() < 0.01:
        log("  *** WARNING: depth map is flat zero – no Gaussians are being composited.")

    save_png(img,                             out_dir / "debug_raw_render.png")
    save_depth_png(dep,                       out_dir / "debug_depth.png")

    # ── GIF timing breakdown ──────────────────────────────────────────────────
    banner("7. GIF Timing Estimate")
    n_gif_frames = cfg.get("orbit", {}).get("num_frames", 60)
    log(f"Orbit frames in config : {n_gif_frames}")
    log(f"Time per frame         : {elapsed:.2f}s")
    log(f"Estimated total        : {elapsed * n_gif_frames:.0f}s  "
        f"({elapsed * n_gif_frames / 60:.1f} min)")
    if elapsed > 5:
        log("  *** GIF is slow because render_view takes >5s per frame.")
        log("  *** Reduce max_gaussians or tile_size in config to speed up.")
        log(f"  *** At current speed, reduce orbit.num_frames to "
            f"{max(6, int(30/elapsed))} for a 30s total budget.")

    # ── Write stats file ──────────────────────────────────────────────────────
    stats_path = out_dir / "debug_stats.txt"
    stats_path.write_text("\n".join(stats_lines))
    print(f"\n{'='*60}")
    print(f"  Stats written to {stats_path}.")
    print(f"  Check {out_dir}/ for debug images.")
    print('='*60)

if __name__ == "__main__":
    main()
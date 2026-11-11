"""
ablation.py
───────────
Ablation studies comparing:
  A) SAM3 vs Grounded-SAM2 as the 2D backbone
  B) Multi-view alpha-weighted fusion vs single-view lifting
  C) Number of views N sensitivity

Run:
    python scripts/ablation.py --scene garden --prompt "the plant" --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path
from typing import List

import numpy as np
import torch
import yaml
from rich.console import Console
from rich.table import Table

console = Console()


def _get_device(cfg: dict) -> torch.device:
    d = cfg.get("device", "cpu")
    if d == "mps" and not torch.backends.mps.is_available():
        console.log("[yellow]MPS not available – falling back to CPU[/yellow]")
        d = "cpu"
    return torch.device(d)


def _load_scene_and_cameras(scene_name: str, cfg: dict, device: torch.device):
    """Load scene + cameras (shared across ablation runs)."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from gaussian_scene import GaussianScene
    from camera_utils import load_cameras

    scene_dir = Path(cfg["scenes_dir"]) / scene_name
    # Find the .ply file (point_cloud/iteration_30000/point_cloud.ply is standard)
    ply_candidates = [
        scene_dir / "point_cloud" / "iteration_30000" / "point_cloud.ply",
        scene_dir / "point_cloud" / "iteration_7000" / "point_cloud.ply",
        *sorted(scene_dir.rglob("*.ply")),
    ]
    ply_path = next((p for p in ply_candidates if p.exists()), None)
    if ply_path is None:
        raise FileNotFoundError(f"No .ply found in {scene_dir}")

    scene = GaussianScene(ply_path, device)

    cameras = load_cameras(
        scene_dir,
        width=cfg["render"]["image_width"],
        height=cfg["render"]["image_height"],
        device=device,
        max_cameras=cfg["render"].get("num_views", 24),
    )
    return scene, cameras


def _copy_scene(scene):
    """Deep-copy a GaussianScene for independent ablation runs."""
    import copy
    from gaussian_scene import GaussianScene
    new = object.__new__(GaussianScene)
    new.device    = scene.device
    new.ply_path  = scene.ply_path
    new.xyz       = scene.xyz.clone()
    new.opacity   = scene.opacity.clone()
    new.scales    = scene.scales.clone()
    new.quats     = scene.quats.clone()
    new.sh_dc     = scene.sh_dc.clone()
    new.sh_rest   = scene.sh_rest.clone() if scene.sh_rest is not None else None
    new.seg_mask  = scene.seg_mask.clone()
    return new


# ──────────────────────────────────────────────────────────────────────────────
#  Ablation A:  Backbone comparison  (SAM3 vs Grounded-SAM2)
# ──────────────────────────────────────────────────────────────────────────────

def ablation_backbone(scene, cameras, prompt: str, cfg: dict, device: torch.device) -> dict:
    """Run lifting with both backbones and compare results."""
    from segmentation import build_segmenter
    from mask_lifting import run_lifting_pipeline
    from evaluate import eval_clip_similarity, project_seg_mask_to_2d
    from gaussian_scene import render_view

    results = {}
    background = torch.tensor(cfg["render"]["background"], device=device)
    eval_cameras = cameras[:cfg["evaluation"]["num_eval_views"]]

    for backbone_name in ["grounded_sam2", "sam3"]:
        console.rule(f"[bold]Backbone: {backbone_name}[/bold]")
        cfg_copy = copy.deepcopy(cfg)
        cfg_copy["segmentation"]["backbone"] = backbone_name

        try:
            segmenter = build_segmenter(cfg_copy, device)
        except Exception as e:
            console.log(f"[red]Skipping {backbone_name}: {e}[/red]")
            results[backbone_name] = {"error": str(e)}
            continue

        scene_copy = _copy_scene(scene)
        t0 = time.time()
        run_lifting_pipeline(scene_copy, cameras, segmenter, prompt, cfg_copy)
        elapsed = time.time() - t0

        # Render eval views after lifting (mask overlay)
        renders, masks_2d = [], []
        for cam in eval_cameras:
            r = render_view(scene_copy, cam, background=background)
            rgb = (r["rgb"].cpu().numpy() * 255).astype(np.uint8)
            renders.append(rgb)
            masks_2d.append(project_seg_mask_to_2d(scene_copy, cam))

        clip_sim = 0.0
        try:
            from evaluate import eval_clip_similarity
            clip_sim = eval_clip_similarity(renders, masks_2d, prompt,
                                            cfg["evaluation"]["clip_model"], device)
        except Exception:
            pass

        n_seg = int(scene_copy.seg_mask.sum().item())
        results[backbone_name] = {
            "n_gaussians_segmented": n_seg,
            "fraction_segmented":    n_seg / scene_copy.num_gaussians,
            "clip_similarity":       clip_sim,
            "runtime_seconds":       elapsed,
        }

    return results


# ──────────────────────────────────────────────────────────────────────────────
#  Ablation B:  Multi-view vs single-view
# ──────────────────────────────────────────────────────────────────────────────

def ablation_multiview_vs_singleview(
    scene, cameras, prompt: str, cfg: dict, device: torch.device
) -> dict:
    """Compare multi-view alpha-weighted fusion vs single-view baseline."""
    from segmentation import build_segmenter
    from mask_lifting import run_lifting_pipeline, single_view_lift
    from gaussian_scene import render_view
    from evaluate import project_seg_mask_to_2d, eval_clip_similarity

    segmenter  = build_segmenter(cfg, device)
    background = torch.tensor(cfg["render"]["background"], device=device)
    eval_cams  = cameras[:cfg["evaluation"]["num_eval_views"]]

    results = {}

    # ── Multi-view ────────────────────────────────────────────────────────────
    console.rule("[bold]Multi-view alpha-weighted fusion[/bold]")
    scene_mv = _copy_scene(scene)
    run_lifting_pipeline(scene_mv, cameras, segmenter, prompt, cfg)

    # ── Single-view baseline ──────────────────────────────────────────────────
    console.rule("[bold]Single-view baseline[/bold]")
    scene_sv = _copy_scene(scene)
    # Pick the camera that likely sees the object best (front-ish view)
    best_cam = cameras[0]
    result   = render_view(scene_sv, best_cam, background=background)
    rgb_np   = (result["rgb"].cpu().numpy() * 255).astype(np.uint8)
    mask_2d, _ = segmenter.segment(rgb_np, prompt)
    single_view_lift(scene_sv, best_cam, mask_2d, cfg)

    # ── Evaluate both ─────────────────────────────────────────────────────────
    for label, sc in [("multi_view", scene_mv), ("single_view", scene_sv)]:
        renders, masks_2d = [], []
        for cam in eval_cams:
            r = render_view(sc, cam, background=background)
            renders.append((r["rgb"].cpu().numpy() * 255).astype(np.uint8))
            masks_2d.append(project_seg_mask_to_2d(sc, cam))

        clip_sim = 0.0
        try:
            clip_sim = eval_clip_similarity(renders, masks_2d, prompt,
                                            cfg["evaluation"]["clip_model"], device)
        except Exception:
            pass

        n_seg = int(sc.seg_mask.sum().item())
        results[label] = {
            "n_gaussians_segmented": n_seg,
            "fraction_segmented":    n_seg / sc.num_gaussians,
            "clip_similarity":       clip_sim,
        }

    return results


# ──────────────────────────────────────────────────────────────────────────────
#  Ablation C:  Number of views sensitivity
# ──────────────────────────────────────────────────────────────────────────────

def ablation_num_views(
    scene, cameras, prompt: str, cfg: dict, device: torch.device,
    view_counts: List[int] = [1, 4, 8, 16, 24],
) -> dict:
    """Run lifting with different numbers of views and measure outcome."""
    from segmentation import build_segmenter
    from mask_lifting import run_lifting_pipeline

    segmenter = build_segmenter(cfg, device)
    results   = {}

    for n in view_counts:
        if n > len(cameras):
            continue
        console.rule(f"[bold]Views = {n}[/bold]")
        scene_n = _copy_scene(scene)
        cfg_n   = copy.deepcopy(cfg)
        cfg_n["render"]["num_views"] = n
        run_lifting_pipeline(scene_n, cameras, segmenter, prompt, cfg_n)
        n_seg = int(scene_n.seg_mask.sum().item())
        results[n] = {
            "n_gaussians_segmented": n_seg,
            "fraction_segmented":    n_seg / scene_n.num_gaussians,
        }

    return results


# ──────────────────────────────────────────────────────────────────────────────
#  CLI
# ──────────────────────────────────────────────────────────────────────────────

def print_results_table(results: dict, title: str):
    table = Table(title=title, show_header=True)
    table.add_column("Condition", style="cyan")
    table.add_column("# Gaussians", justify="right")
    table.add_column("Fraction", justify="right")
    table.add_column("CLIP Sim", justify="right")
    table.add_column("Runtime (s)", justify="right")

    for k, v in results.items():
        if "error" in v:
            table.add_row(str(k), "[red]ERROR[/red]", "-", "-", "-")
        else:
            table.add_row(
                str(k),
                f"{v.get('n_gaussians_segmented', '-'):,}" if isinstance(v.get('n_gaussians_segmented'), int) else "-",
                f"{v.get('fraction_segmented', 0):.3f}" if 'fraction_segmented' in v else "-",
                f"{v.get('clip_similarity', 0):.4f}" if 'clip_similarity' in v else "-",
                f"{v.get('runtime_seconds', 0):.1f}" if 'runtime_seconds' in v else "-",
            )
    console.print(table)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene",  required=True, help="Scene name, e.g. garden")
    parser.add_argument("--prompt", required=True, help="Text prompt, e.g. 'the plant'")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--ablation", choices=["backbone", "multiview", "views", "all"],
                        default="all")
    parser.add_argument("--out_dir", default="outputs")
    args = parser.parse_args()

    import sys
    sys.path.insert(0, str(Path(__file__).parent))

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = _get_device(cfg)
    console.print(f"\n[bold]Ablation Study[/bold] – scene=[cyan]{args.scene}[/cyan], "
                  f"prompt=[green]{args.prompt}[/green], device=[yellow]{device}[/yellow]\n")

    scene, cameras = _load_scene_and_cameras(args.scene, cfg, device)
    out_dir = Path(args.out_dir) / args.scene / "ablations"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    if args.ablation in ("backbone", "all"):
        r = ablation_backbone(scene, cameras, args.prompt, cfg, device)
        all_results["backbone"] = r
        print_results_table(r, "Backbone Comparison (SAM3 vs Grounded-SAM2)")

    if args.ablation in ("multiview", "all"):
        r = ablation_multiview_vs_singleview(scene, cameras, args.prompt, cfg, device)
        all_results["multiview_vs_singleview"] = r
        print_results_table(r, "Multi-view vs Single-view Ablation")

    if args.ablation in ("views", "all"):
        r = ablation_num_views(scene, cameras, args.prompt, cfg, device)
        all_results["num_views_sensitivity"] = r
        print_results_table(r, "Number of Views Sensitivity")

    # Save JSON
    out_json = out_dir / "ablation_results.json"
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    console.log(f"\nResults saved → [cyan]{out_json}[/cyan]")


if __name__ == "__main__":
    main()

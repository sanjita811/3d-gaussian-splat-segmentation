# Training-Free Multi-View Semantic Segmentation of 3D Gaussian Splatting Scenes

Segment and edit objects in a **pretrained 3D Gaussian Splatting (3DGS) scene** with nothing more than a text prompt. The pipeline needs no retraining, no per-scene optimisation and no ground-truth labels.

> *"potted plant"* → recolor red · *"bicycle"* → recolor blue · *"change the flowers to green"* → natural-language edit

---

## Overview

3DGS renders photorealistic novel views in real time, but it is *semantically blind*: no Gaussian knows whether it belongs to the tree, the bicycle or the tablecloth. Existing methods add semantics by retraining the scene or by embedding features during optimisation. This project works on **any pretrained scene, as is**.

Given a scene and a prompt, the pipeline:

1. **Renders** the scene from multiple viewpoints.
2. **Segments** each render in 2D with **GroundingDINO** (text → boxes) and **SAM2** (boxes → pixel masks).
3. **Lifts** the 2D masks to 3D by projecting every Gaussian into each view and accumulating an opacity-weighted vote.
4. **Thresholds and filters** the per-Gaussian scores to get a binary 3D segment.
5. **Edits** the segment: recolor, remove, or apply a natural-language instruction.
6. **Evaluates** the result with CLIP similarity, LPIPS (background preservation) and, optionally, mIoU.

Every edit is applied to the 3D Gaussians themselves, so the result is **consistent across viewpoints by construction**. No per-frame post-processing is needed.

## Method

### Multi-view mask lifting

For each view *v*, project Gaussian centre μᵢ to pixel (uᵢᵛ, vᵢᵛ), look up the 2D mask value mᵢᵛ ∈ {0,1}, and accumulate:

```
sᵢ += αᵢ · mᵢᵛ
```

where αᵢ is the Gaussian's opacity (in the PyTorch renderer here, αᵢ is its peak per-view alpha contribution). The implementation in [`scripts/mask_lifting.py`](scripts/mask_lifting.py) also weights by segmentation confidence and normalises by visibility:

```
score[i] = Σ_v alpha_v[i] · mask_v[proj_v(i)] · conf_v  /  Σ_v 1[alpha_v[i] > alpha_min]
seg[i]   = (score[i] > threshold) AND (views_hit[i] ≥ min_views)
```

### Filtering

Back-projection is ambiguous: Gaussians *behind* the object fall into the same masked pixels. Our experiments combined a percentile score threshold with lightweight spatial and colour filters, such as a height cutoff and XZ radius for the bonsai and a brightness filter to separate the silver bicycle from the black bench.

### Edits

| Edit | Operation |
|------|-----------|
| **Recolor** | Set the colour of every selected Gaussian to a target RGB |
| **Remove** | Render only the unselected Gaussians |
| **NL instruction** | Parse `"change the X to Y"`, segment `X`, recolor to `Y` |

## Results

- **Bonsai**, prompt `"potted plant"` (272,956 Gaussians). The trunk, branches and pot are recolored red, consistently across all orbit views.
- **Bicycle**, prompt `"bicycle"` (567,714 Gaussians). A brightness filter separates the silver bicycle from the black bench in the same depth plane, so the bicycle is recolored blue while the bench, grass and path stay unchanged.
- **Natural-language edit:** `"change the flowers to green"` on the bonsai scene re-segments the flowers and recolors them green.

## Repository structure

```
.
├── README.md
├── LICENSE
├── requirements.txt
├── setup.sh                 # one-shot env + checkpoints + scene download (Apple Silicon)
├── configs/
│   └── default.yaml         # all pipeline hyperparameters
├── scripts/
│   ├── run_pipeline.py      # ★ main entry point
│   ├── gaussian_scene.py    # PLY loader + pure-PyTorch tile rasteriser (MPS-friendly)
│   ├── camera_utils.py      # cameras.json / COLMAP / transforms.json loaders, orbit cameras
│   ├── segmentation.py      # Grounded-SAM2 and SAM-style backends
│   ├── mask_lifting.py      # multi-view alpha-weighted lifting
│   ├── evaluate.py          # CLIP / LPIPS / mIoU
│   ├── visualize.py         # orbit GIFs, overlays, comparison strips
│   ├── ablation.py          # backbone / multi- vs single-view / #views ablations
│   ├── fetch_cameras.py     # download COLMAP cameras for Mip-NeRF 360 scenes
│   ├── get_ply.sh           # (optional) train your own scenes with nerfstudio
│   └── debug_seg.py         # renderer diagnostics
```

Created at runtime (git-ignored): `.venv/`, `third_party/`, `checkpoints/`, `scenes/`, `outputs/`.

## Getting started

### Prerequisites

- macOS on Apple Silicon (MPS). The renderer is pure PyTorch, so CPU/CUDA also work if you set `device` in `configs/default.yaml`.
- Python 3.11 (`brew install python@3.11`)
- About 20 GB of free disk space for scenes and checkpoints

### Setup

```bash
git clone https://github.com/sanjita811/3d-gaussian-splat-segmentation.git
cd 3d-gaussian-splat-segmentation
bash setup.sh
source .venv/bin/activate
```

`setup.sh` creates a virtualenv, installs PyTorch and the requirements, installs SAM2 and GroundingDINO from source, downloads their checkpoints, and fetches pretrained Mip-NeRF 360 scenes (garden, room, bonsai, bicycle, counter, kitchen) from the [`dylanebert/3dgs`](https://huggingface.co/datasets/dylanebert/3dgs) Hugging Face dataset.

To use the original training cameras instead of auto-generated orbit cameras, run:

```bash
python scripts/fetch_cameras.py --scene bicycle
```

### Run

```bash
# Recolor the bonsai red
PYTORCH_ENABLE_MPS_FALLBACK=1 python scripts/run_pipeline.py \
    --scene bonsai --prompt "potted plant" --edit recolor --recolor_rgb 1.0 0.1 0.1

# Remove the bicycle
python scripts/run_pipeline.py --scene bicycle --prompt "bicycle" --edit remove

# Fewer views for a faster run
python scripts/run_pipeline.py --scene garden --prompt "the table" --num_views 8
```

Outputs go to `outputs/{scene}/{prompt}/`: `comparison.png`, before/after orbit GIFs, `metrics.json` and per-view renders and masks.

### Ablations

```bash
python scripts/ablation.py --scene garden --prompt "the plant" --ablation all
# --ablation backbone | multiview | views | all
```

Results go to `outputs/{scene}/ablations/ablation_results.json`.

## Configuration

Key knobs in [`configs/default.yaml`](configs/default.yaml):

| Parameter | Default | Effect |
|-----------|---------|--------|
| `render.num_views` | 24 | Views used for lifting |
| `render.image_width/height` | 800 × 600 | Render resolution fed to the segmenter |
| `segmentation.grounding_dino.box_threshold` | 0.25 | Detection confidence |
| `lifting.score_threshold` | 0.15 | Per-Gaussian score cutoff |
| `lifting.min_views_hit` | 2 | Min views a Gaussian must be visible in |
| `lifting.alpha_min` | 0.005 | Alpha needed to count as "visible" |
| `editing.recolor_rgb` | [1.0, 0.2, 0.2] | Default recolor target |

## Limitations and future work

- **Occluded-Gaussian contamination:** interior or behind-surface Gaussians collect votes. *Fix:* depth-aware lifting that gates votes on a per-view Z-buffer.
- **Scene-specific filters:** the spatial thresholds are tuned by hand. *Fix:* Otsu thresholding on the score distribution.
- **Single elevation ring:** the tops and undersides of objects are under-sampled. *Fix:* multiple elevation rings.
- **No quantitative mIoU:** the scenes ship without per-Gaussian labels.
- **Next steps:** CLIP/DINOv2 feature-field lifting, SH-based style transfer, and extension to 4D Gaussians.

## Acknowledgements

Built on [3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) (Kerbl et al., 2023), [SAM 2](https://github.com/facebookresearch/sam2) (Ravi et al., 2024), [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO) (Liu et al., 2023), [Mip-NeRF 360](https://jonbarron.info/mipnerf360/) (Barron et al., 2022), [gsplat](https://github.com/nerfstudio-project/gsplat), and pretrained scenes from [`dylanebert/3dgs`](https://huggingface.co/datasets/dylanebert/3dgs).

## License

[MIT](LICENSE)

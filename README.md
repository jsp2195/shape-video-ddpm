# PC_AE_Video_DDPM

Controlled synthetic shape-transformation latent video diffusion for point clouds.

This repository contains a self-contained pipeline, centered on
[`DDPM_Video__PCAE.py`](DDPM_Video__PCAE.py), for training a PointCloud
Autoencoder, compressing point clouds into compact latent grids, constructing
synthetic latent video trajectories, training latent video DDPMs, and decoding
sampled latent trajectories back into point-cloud video frames.

The project supports two related generation regimes:

1. **Endpoint-conditioned latent video diffusion**: learns to generate middle
   latent frames between a start and end point-cloud shape.
2. **Endpoint + control-conditioned latent video diffusion**: learns to generate
   middle latent frames conditioned on start/end shapes plus a 16-dimensional
   synthetic deformation control vector.

<p align="center">
  <img src="ae_reconstruction_best.png" width="720"
       alt="PointCloudAE reconstruction at the best epoch: ground truth (top), 32x32 latent grids (middle), reconstructions (bottom).">
</p>

<p align="center">
  <em>PointCloudAE reconstructions at the current best epoch
  (full_32, epoch 18, val loss 0.2522).
  Top: ground-truth point clouds. Middle: 32&times;32 latent grids.
  Bottom: AE reconstructions.</em>
</p>

<p align="center">
  <img src="latent_video_sample.gif" width="480"
       alt="Decoded point-cloud video sampled from the endpoint-conditioned latent video DDPM.">
</p>

<p align="center">
  <em>Decoded point-cloud video from the endpoint-conditioned latent video
  DDPM (pilot_32, 10k pairs, sample_0003). 10 frames, each frame is the AE
  decoder applied to a sampled latent grid.</em>
</p>

---

## Table of contents

1. [Project summary](#1-project-summary)
2. [Input data](#2-input-data)
3. [Repository structure](#3-repository-structure)
4. [Model architecture](#4-model-architecture)
5. [Controlled deformation modes](#5-controlled-deformation-modes)
6. [Control vector schema](#6-control-vector-schema)
7. [CLI commands](#7-cli-commands)
8. [Canonical workflow](#8-canonical-workflow)
9. [Outputs and checkpoints](#9-outputs-and-checkpoints)
10. [Validation checklist](#10-validation-checklist)
11. [Monitoring](#11-monitoring)
12. [Scientific limitations](#12-scientific-limitations)
13. [Recommended next experiments](#13-recommended-next-experiments)
14. [Reproducibility](#14-reproducibility)
15. [Files outside the main pipeline](#15-files-outside-the-main-pipeline)
16. [License](#license)

---

## 1. Project summary

This project trains and evaluates a controlled synthetic shape-transformation
latent video DDPM for point clouds.

The pipeline:

- trains a PointCloud Autoencoder on normalized point clouds;
- encodes every point cloud into a `[latent_size, latent_size]` latent grid;
- builds synthetic latent video datasets using either:
  - linear interpolation between AE-latent endpoints; or
  - controlled synthetic geometric deformations encoded through the frozen AE;
- trains an endpoint-conditioned latent video DDPM;
- trains a control-conditioned latent video DDPM;
- samples latent video trajectories;
- decodes each generated latent frame back into a point cloud;
- writes point-cloud frames, preview images, GIF animations, optional MP4s, and
  audit reports.

The core design is intentionally staged:

```text
Point clouds
  → PointCloudAE training
  → frozen AE encoder
  → latent grids
  → synthetic latent videos
  → latent video DDPM training
  → sampled latent trajectories
  → frozen AE decoder
  → point-cloud video frames
```

The AE and DDPM are trained separately. The AE learns the shape representation;
the DDPM learns a latent video prior over shape transitions.

---

## 2. Input data

Primary data path:

```text
data/normalized_rotated_point_clouds6.npy
```

Expected shape:

```text
(N, 1000, 6)
```

Current known dataset shape:

```text
(11892, 1000, 6)
```

Each point cloud contains 1000 points. Each point has six features:

```text
[x, y, z, feature_3, feature_4, feature_5]
```

In the current workflow, the last three channels are treated as normal-like or
auxiliary geometric features.

Large data files are intentionally excluded from Git. Keep them local or store
them in Git LFS, Hugging Face Datasets, an object store, or another external
artifact system.

---

## 3. Repository structure

Main script:

```text
DDPM_Video__PCAE.py
```

Optional helper scripts:

```text
evaluate_pilot_samples.py
make_pointcloud_video_previews.py
```

Expected local artifact directories:

```text
data/
checkpoints/
outputs/
logs/
```

These directories are typically ignored by Git except for placeholders.

### Important files

| File | Purpose |
| --- | --- |
| `DDPM_Video__PCAE.py` | Main training, dataset-building, sampling, and audit script |
| `README.md` | Project documentation |
| `LICENSE` | License file |
| `evaluate_pilot_samples.py` | Computes sanity metrics for sampled pilot videos |
| `make_pointcloud_video_previews.py` | Rebuilds GIF/MP4 previews from decoded frame `.npy` files |
| `data/.gitkeep` | Placeholder for local data directory |

---

## 4. Model architecture

### Stage 1 — PointCloudAE

The PointCloudAE compresses individual point clouds into compact latent grids.

Input:

```text
[1000, 6]
```

Latent representation:

```text
[latent_size, latent_size]
```

Typical latent size:

```text
[32, 32]
```

Output reconstruction:

```text
[1000, 6]
```

The AE is trained independently of the DDPM. This is the first-stage latent
representation model.

### Stage 2 — Latent encoding

After AE training, the frozen encoder maps every point cloud into a latent grid.

Example output:

```text
outputs/latent_videos_full_32/encoded_features.npy
```

Expected encoded feature shape:

```text
(N, 32, 32)
```

### Stage 3 — Linear endpoint latent videos

The simplest video dataset is built by interpolating between two encoded
point-cloud latents:

```text
z_start → z_1 → z_2 → ... → z_end
```

This creates endpoint-conditioned training data for a baseline latent video DDPM.

### Stage 4 — Controlled synthetic shape-transformation videos

Controlled videos are built in point-cloud space first, then encoded into latent
space.

For each source point cloud:

1. sample a deformation mode;
2. sample deformation parameters;
3. generate a sequence of transformed point-cloud frames;
4. encode each transformed frame through the frozen AE;
5. save the resulting latent video tensor;
6. save the 16-dimensional control vector;
7. save the deformation-mode label.

Expected controlled latent video shape:

```text
(num_sequences, num_frames, latent_size, latent_size)
```

Expected control tensor shape:

```text
(num_sequences, 16)
```

### Stage 5 — Endpoint-conditioned latent video DDPM

The endpoint-conditioned model learns to denoise/generate middle latent frames
given the start and end latent frames.

Training target:

```text
middle latent frames
```

Conditioning:

```text
z_start, z_end
```

Practical description:

```text
Given shape A and shape B, generate a plausible latent video trajectory between them.
```

### Stage 6 — Control-conditioned latent video DDPM

The control-conditioned model extends the endpoint-conditioned DDPM by adding a
control vector.

Training target:

```text
middle latent frames
```

Conditioning:

```text
z_start, z_end, control_vector
```

The control vector is embedded with an MLP and added to the diffusion timestep
embedding.

Practical description:

```text
Given shape A, shape B, and a requested synthetic deformation, generate the latent video trajectory.
```

### Stage 7 — Sampling and decoding

At sampling time, the DDPM generates latent middle frames. The final latent video
is then decoded frame-by-frame through the frozen AE decoder.

Each sample directory contains:

```text
frame_000.npy
frame_001.npy
...
frame_009.npy
latent_video.npz
preview.png
animation.gif
```

For controlled sampling, it also contains:

```text
control_vector.npy
control_config.json
```

---

## 5. Controlled deformation modes

The controlled dataset builder supports seven synthetic deformation modes.

### `cavity_expansion`

Radially pushes a shell of points outward from a sampled center. The strength
ramps smoothly across frames.

### `cavity_contraction`

The inverse of cavity expansion. Points in the selected shell are pulled inward.

### `anisotropic_expansion`

Applies directionally biased expansion along a sampled axis. This creates
lopsided or axis-sensitive shape motion.

### `twist`

Rotates points around a sampled axis. Rotation strength increases across frames.

### `hybrid_cavity_twist`

Combines cavity displacement with twist. This creates richer compound motion
than simple interpolation.

### `bulge_then_relax`

Applies a transient deformation that grows and then relaxes. The motion is
non-monotonic, producing a bulge-like trajectory.

### `two_stage_morph`

Applies a staged transformation: cavity deformation over the first part of the
sequence, followed by twist over the second part.

---

## 6. Control vector schema

The control-conditioned DDPM uses a fixed 16-dimensional control vector.

| idx | name | Description |
| --- | --- | --- |
| 0 | `mode_id` | Deformation mode ID normalized to `[0, 1]` |
| 1 | `cavity_radius` | Radius of the affected shell |
| 2 | `shell_width` | Width of the radial shell mask |
| 3 | `max_strength` | Main deformation strength |
| 4 | `twist_strength` | Twist angle or twist amplitude |
| 5 | `anisotropy_strength` | Directional expansion strength |
| 6 | `center_x` | Deformation center x-coordinate |
| 7 | `center_y` | Deformation center y-coordinate |
| 8 | `center_z` | Deformation center z-coordinate |
| 9 | `axis_x` | Deformation axis x-component |
| 10 | `axis_y` | Deformation axis y-component |
| 11 | `axis_z` | Deformation axis z-component |
| 12 | `num_frames` | Number of frames, normalized |
| 13 | `latent_size` | Latent grid size, normalized |
| 14 | `phase_or_relaxation_strength` | Used by non-monotonic modes |
| 15 | `composite_strength` | Reserved or compound-mode strength |

---

## 7. CLI commands

The main script exposes these commands:

| Command | Purpose |
| --- | --- |
| `train_ae` | Train the PointCloudAE |
| `encode_dataset` | Encode all point clouds into AE latent grids |
| `build_latent_videos` | Build linear endpoint-interpolation latent videos |
| `train_video_ddpm` | Train endpoint-conditioned latent video DDPM |
| `sample_video` | Sample one endpoint-conditioned latent video |
| `build_controlled_shape_videos` | Build controlled synthetic latent video dataset |
| `train_controlled_video_ddpm` | Train control-conditioned latent video DDPM |
| `sample_controlled_video` | Sample one controlled latent video |
| `audit_controlled_videos` | Write markdown audit report for controlled run |
| `run_all` | Convenience wrapper for the older endpoint-conditioned pipeline |

Check command availability with:

```bash
python3 DDPM_Video__PCAE.py --help
```

---

## 8. Canonical workflow

The following commands define the recommended full 32-latent workflow.

### 8.1 Train the full PointCloudAE

```bash
python3 DDPM_Video__PCAE.py --seed 123 train_ae \
  --pointcloud_path data/normalized_rotated_point_clouds6.npy \
  --ae_ckpt_dir checkpoints/ae_video_pcae_full_32 \
  --latent_size 32 \
  --ae_batch_size 16 \
  --ae_epochs 500 \
  --patience 50 \
  --ae_lr 1e-4 \
  --val_split 0.1
```

Expected outputs:

```text
checkpoints/ae_video_pcae_full_32/ae_latest.pth
checkpoints/ae_video_pcae_full_32/ae_best.pth
checkpoints/ae_video_pcae_full_32/ae_final.pth
checkpoints/ae_video_pcae_full_32/ae_meta.json
checkpoints/ae_video_pcae_full_32/ae_epoch_losses.txt
```

### 8.2 Encode the full dataset

```bash
python3 DDPM_Video__PCAE.py --seed 123 encode_dataset \
  --pointcloud_path data/normalized_rotated_point_clouds6.npy \
  --ae_ckpt_dir checkpoints/ae_video_pcae_full_32 \
  --encoded_features outputs/latent_videos_full_32/encoded_features.npy \
  --encode_batch_size 64
```

Expected output shape:

```text
(11892, 32, 32)
```

### 8.3 Build controlled synthetic latent videos

```bash
python3 DDPM_Video__PCAE.py --seed 123 build_controlled_shape_videos \
  --pointcloud_path data/normalized_rotated_point_clouds6.npy \
  --ae_ckpt_dir checkpoints/ae_video_pcae_full_32 \
  --controlled_video_path outputs/controlled_latent_videos_full_32/controlled_shape_videos_50k.npy \
  --control_path outputs/controlled_latent_videos_full_32/controlled_shape_controls_50k.npy \
  --num_sequences 50000 \
  --num_frames 10 \
  --latent_size 32 \
  --encode_batch_size 64 \
  --deformation_modes cavity_expansion,cavity_contraction,anisotropic_expansion,twist,hybrid_cavity_twist,bulge_then_relax,two_stage_morph \
  --deformation_mode_probs 0.22,0.08,0.20,0.15,0.15,0.10,0.10 \
  --control_dim 16 \
  --use_best_ae
```

Expected outputs:

```text
outputs/controlled_latent_videos_full_32/controlled_shape_videos_50k.npy
outputs/controlled_latent_videos_full_32/controlled_shape_controls_50k.npy
outputs/controlled_latent_videos_full_32/controlled_shape_videos_50k.labels.npy
outputs/controlled_latent_videos_full_32/controlled_shape_videos_50k.meta.json
```

Expected shapes:

```text
controlled_shape_videos_50k.npy:   (50000, 10, 32, 32)
controlled_shape_controls_50k.npy: (50000, 16)
labels:                            (50000,)
```

### 8.4 Train the controlled DDPM

```bash
python3 DDPM_Video__PCAE.py --seed 123 train_controlled_video_ddpm \
  --latent_video_path outputs/controlled_latent_videos_full_32/controlled_shape_videos_50k.npy \
  --control_path outputs/controlled_latent_videos_full_32/controlled_shape_controls_50k.npy \
  --video_ddpm_ckpt_dir checkpoints/controlled_video_ddpm_full_32_50k \
  --batch_size 8 \
  --epochs 500 \
  --patience 25 \
  --lr 1e-4 \
  --timesteps 1000 \
  --beta_schedule cosine \
  --base_channels 32 \
  --control_dim 16 \
  --val_split 0.1
```

The command name is exactly:

```text
train_controlled_video_ddpm
```

### 8.5 Sample one controlled video

```bash
python3 DDPM_Video__PCAE.py --seed 123 sample_controlled_video \
  --encoded_features outputs/latent_videos_full_32/encoded_features.npy \
  --pointcloud_path data/normalized_rotated_point_clouds6.npy \
  --ae_ckpt_dir checkpoints/ae_video_pcae_full_32 \
  --video_ddpm_ckpt_dir checkpoints/controlled_video_ddpm_full_32_50k \
  --start_index 0 \
  --end_index 10 \
  --num_frames 10 \
  --sample_dir outputs/controlled_samples_full_32_50k \
  --sample_id 0 \
  --mode cavity_expansion \
  --cavity_radius 0.35 \
  --shell_width 0.15 \
  --max_strength 0.10
```

### 8.6 Audit a controlled run

```bash
python3 DDPM_Video__PCAE.py audit_controlled_videos \
  --sample_dir outputs/controlled_samples_full_32_50k \
  --latent_video_path outputs/controlled_latent_videos_full_32/controlled_shape_videos_50k.npy \
  --control_path outputs/controlled_latent_videos_full_32/controlled_shape_controls_50k.npy \
  --video_ddpm_ckpt_dir checkpoints/controlled_video_ddpm_full_32_50k \
  --ae_ckpt_dir checkpoints/ae_video_pcae_full_32 \
  --output_report outputs/controlled_samples_full_32_50k/audit_report.md
```

---

## 9. Outputs and checkpoints

### AE checkpoints

```text
checkpoints/ae_video_pcae_full_32/
  ae_best.pth
  ae_final.pth
  ae_latest.pth
  ae_meta.json
  ae_epoch_losses.txt
  epoch_viz/
```

### Endpoint-conditioned DDPM checkpoints

```text
checkpoints/video_ddpm_pcae_pilot_32_10k/
  video_ddpm_best.pt
  video_ddpm_final.pt
  video_ddpm_latest.pt
  video_ddpm_meta.json
  video_ddpm_epoch_losses.txt
```

### Controlled DDPM checkpoints

```text
checkpoints/controlled_video_ddpm_full_32_50k/
  controlled_video_ddpm_best.pt
  controlled_video_ddpm_final.pt
  controlled_video_ddpm_latest.pt
  controlled_video_ddpm_meta.json
  controlled_video_ddpm_epoch_losses.txt
```

### Encoded features

```text
outputs/latent_videos_full_32/encoded_features.npy
outputs/latent_videos_pilot_32/encoded_features.npy
```

### Latent videos

```text
outputs/latent_videos_pilot_32/latent_interpolation_videos_10k.npy
outputs/controlled_latent_videos_full_32/controlled_shape_videos_50k.npy
```

### Controlled sample outputs

```text
outputs/controlled_samples_full_32_50k/sample_0000/
  frame_000.npy
  frame_001.npy
  ...
  frame_009.npy
  latent_video.npz
  control_vector.npy
  control_config.json
  preview.png
  animation.gif
```

### Pilot and smoke artifacts

The repo may retain local pilot/smoke artifacts that informed the full workflow:

```text
checkpoints/ae_video_pcae_pilot_32/
checkpoints/video_ddpm_pcae_pilot_32_10k/
checkpoints/controlled_video_ddpm_smoke_32/
outputs/latent_videos_pilot_32/
outputs/video_samples_pilot_32_10k/
outputs/controlled_latent_videos_32/
outputs/controlled_samples_smoke_32/
```

These artifacts are useful locally but are generally excluded from Git.

---

## 10. Validation checklist

For each run, verify:

- encoded features have shape `(N, latent_size, latent_size)`;
- latent videos have shape `(N, T, latent_size, latent_size)`;
- control vectors have shape `(N, 16)`;
- decoded frames have shape `(1000, 6)`;
- all arrays are finite;
- `preview.png` exists;
- `animation.gif` exists;
- `*_epoch_losses.txt` shows bounded train/validation losses;
- `*_meta.json` exists and matches the command configuration;
- audit report exists for controlled runs.

Quick checks:

```bash
python3 - <<'PY'
from pathlib import Path
import numpy as np

paths = [
    "outputs/latent_videos_full_32/encoded_features.npy",
    "outputs/controlled_latent_videos_full_32/controlled_shape_videos_50k.npy",
    "outputs/controlled_latent_videos_full_32/controlled_shape_controls_50k.npy",
]

for p in paths:
    p = Path(p)
    if p.exists():
        x = np.load(p, mmap_mode="r")
        print(p, x.shape, x.dtype)

print("done")
PY
```

---

## 11. Monitoring

### GPU status

```bash
watch -n 10 'nvidia-smi'
```

### AE training

```bash
tail -f logs/ae_full_32.log
tail -n 20 checkpoints/ae_video_pcae_full_32/ae_epoch_losses.txt
```

### Controlled DDPM training

```bash
tail -f logs/controlled_video_ddpm_full_32_50k.log
tail -n 20 checkpoints/controlled_video_ddpm_full_32_50k/controlled_video_ddpm_epoch_losses.txt
```

### Combined monitor

```bash
watch -n 10 '
echo "=== GPU ==="
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw --format=csv
echo
echo "=== PROCESS ==="
ps -ef | grep "DDPM_Video__PCAE.py" | grep -v grep || echo "no process"
echo
echo "=== AE LOSSES ==="
tail -n 5 checkpoints/ae_video_pcae_full_32/ae_epoch_losses.txt 2>/dev/null || echo "no AE loss file"
echo
echo "=== CONTROLLED DDPM LOSSES ==="
tail -n 5 checkpoints/controlled_video_ddpm_full_32_50k/controlled_video_ddpm_epoch_losses.txt 2>/dev/null || echo "no controlled DDPM loss file"
'
```

---

## 12. Scientific limitations

- This is a controlled synthetic shape-transformation model.
- Deformations are synthetic; they are not measured physical trajectories.
- Do not claim physical realism without external validation.
- Do not claim state-of-the-art performance without controlled comparison.
- Model quality is bounded by AE reconstruction quality.
- Endpoint/control conditioning can be ambiguous for multi-stage transformations.
- Linear interpolation and synthetic deformation are useful generation scaffolds,
  not real observed temporal dynamics.
- Generated point-cloud videos should be treated as synthetic samples, not
  forecasts.

---

## 13. Recommended next experiments

- Train a `latent_size = 64` AE and compare reconstruction and downstream DDPM
  quality against `latent_size = 32`.
- Compare endpoint-only DDPM against control-conditioned DDPM on identical
  endpoint pairs.
- Scale controlled trajectories to 100k if disk and wallclock allow.
- Add explicit mode embeddings or class conditioning instead of using only a
  normalized mode ID in the control vector.
- Add Chamfer / EMD / F-score reconstruction metrics.
- Add quantitative trajectory metrics for generated videos.
- Explore tokenized latent point diffusion as a follow-up architecture.

---

## 14. Reproducibility

- Use `--seed 123` for the canonical workflow.
- Target hardware is a single CUDA GPU.
- CPU fallback exists but is not practical for full training.
- Large artifacts are intentionally local and should not be committed to normal
  Git.
- Use Git LFS, Hugging Face, object storage, or a release artifact if large
  datasets/checkpoints need to be shared.

---

## 15. Files outside the main pipeline

`cleanup_archive/` may contain files from the original working repo that are not
part of this cleaned pipeline, such as:

- old Kinetics video DDPM components;
- old `models/`, `diffusion/`, `utils/`, `metrics/`, `configs/`, `runs/`, and
  `scripts/` folders;
- older `DDPM_PCAE` versions;
- unused point-cloud notebooks;
- abandoned datasets or old checkpoints.

Nothing in `cleanup_archive/` is imported by `DDPM_Video__PCAE.py`.

Inspect it before deleting it permanently.

---

## License

See [LICENSE](LICENSE).

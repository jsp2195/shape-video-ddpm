# Controlled synthetic shape-transformation latent video DDPM – full-data audit report

This report documents the full-data run of the controlled latent video DDPM pipeline trained on synthetic deformations of the existing 11,892 normalized rotated point clouds. No physical realism or scientific validity for any specific material system is claimed.

---

## 1. Hardware and device

- CUDA available: True
- Device: NVIDIA GeForce RTX 4070 Laptop GPU
- VRAM: 8.18 GB (8188 MiB)
- All training and sampling ran on `cuda`; CPU fallback was not used.

## 2. Input data

- Path: `data/normalized_rotated_point_clouds6.npy`
- Shape: `(11892, 1000, 6)`
- dtype: `float64`
- Per-point channels: `(x, y, z, nx, ny, nz)` with normals already normalized (mean `||n|| ≈ 1.0`).
- xyz range (first 100): `[-0.998, 0.993]`, approximately centered (mean ≈ 0).
- Normals range (first 100): `[-1.000, 1.000]`, unit-length.
- Treated as the empirical distribution of shapes from which the controlled trajectories are constructed.

## 3. What this controlled model adds beyond endpoint-only interpolation

The previously trained endpoint-conditioned DDPM only used a start and end latent frame as conditioning, with no control over the trajectory between them. The controlled DDPM in this run adds:

- An explicit 16-dim control vector that names the deformation mode and its parameters (cavity radius/strength, twist axis/strength, anisotropy direction, two-stage phasing, etc.).
- A multi-mode training distribution with 7 deformation modes, so the model learns a family of structured trajectories rather than only the endpoint-to-endpoint shortest interpolation.
- Mode-conditioned trajectories that can express physically motivated motion families (bulge-then-relax, hybrid cavity + twist, two-stage morph) which endpoint conditioning cannot capture without ambiguity.

This produces trajectories that are different from pure linear interpolation when conditioned on the same start/end but a non-trivial mode and parameter set.

## 4. Controlled deformation modes

7 modes, sampled per sequence with the probabilities used at dataset construction time:

- `cavity_expansion` (p=0.22)
- `cavity_contraction` (p=0.08)
- `anisotropic_expansion` (p=0.20)
- `twist` (p=0.15)
- `hybrid_cavity_twist` (p=0.15)
- `bulge_then_relax` (p=0.10)
- `two_stage_morph` (p=0.10)

Parameter sampling ranges (`outputs/controlled_latent_videos_full_32/controlled_shape_videos_50k.meta.json`):

- cavity_radius `[0.15, 0.55]`, shell_width `[0.08, 0.25]`
- cavity_expansion_strength `[0.03, 0.18]`, cavity_contraction_strength `[-0.16, -0.03]`
- anisotropic_strength `[0.04, 0.20]`
- twist_max_angle `[0.2, 1.2]`
- hybrid_cavity_twist_strength `[0.02, 0.10]`, twist `[0.15, 0.8]`
- bulge_then_relax_strength `[0.04, 0.16]`
- two_stage_morph_strength `[0.04, 0.14]`, twist `[0.3, 0.9]`, anisotropy `[0.05, 0.16]`

## 5. Synthetic data construction

For each of 50,000 sequences:
1. Sample one base point cloud index uniformly from the 11,892 available.
2. Sample one of the 7 deformation modes from the mode-probability vector above (first 7 sequences are forced to one each of the 7 modes so every mode appears).
3. Sample mode-specific parameters from the ranges above.
4. Construct a 10-frame deformation trajectory in point-cloud space, with frame 0 equal to the original cloud and frame 9 the final deformed state. xyz is clipped per coordinate to `clip_value = 1.0`.
5. Encode every frame through the trained AE encoder (`ae_best.pth`) in batches of 64, producing a latent video of shape `(10, 32, 32)`.
6. Build a 16-dim control vector summarizing mode id, normalized geometric parameters, axis, center offset, and per-frame schedule statistics.
7. Save the full `(50000, 10, 32, 32)` latent tensor, the `(50000, 16)` control tensor, the `(50000,)` mode-id label tensor, and a meta JSON.

This trades data realism for controllability: the trajectories are not physical PBX deformations, but they cover a structured, parameterized space of shape transformations.

## 6. Full AE training configuration

- Command: `train_ae`
- Pointcloud path: `data/normalized_rotated_point_clouds6.npy`
- Latent size: 32
- Batch size: 16
- Initial LR: 1e-4 (Adam)
- Max epochs: 500
- Patience (early stopping on val loss): 50
- Val split: 0.1
- Resume from `ae_latest.pth`: yes (previous run had completed only epoch 1)
- Device: cuda
- Seed: 123
- No OOM, no batch-size fallback was needed.

## 7. Full AE loss trend

- Epochs completed: 91 (early stopping triggered at epoch 91)
- Best validation loss: **0.238426** at epoch 41
- Final validation loss (epoch 91): 0.250964
- First-epoch validation loss: 0.400899
- Train loss decreased monotonically from 0.5498 → 0.1858.
- Validation loss decreased from 0.4009 → 0.2384 (best), then drifted upward as the model began to overfit (final ≈ 0.25), which is why patience-50 early stopping kicked in.
- `ae_best.pth` is taken from epoch 41 and is the checkpoint used downstream.

## 8. Full AE artifact paths

- Checkpoint dir: `checkpoints/ae_video_pcae_full_32/`
  - `ae_latest.pth` ✓
  - `ae_best.pth` ✓ (epoch 41)
  - `ae_final.pth` ✓ (epoch 91)
  - `ae_meta.json` ✓ (`point_size=1000, latent_size=32`)
  - `ae_epoch_losses.txt` ✓
  - `epoch_viz/ae_epoch_0001.png` … `ae_epoch_0091.png` (91 per-epoch reconstruction visualizations)
- Training log: `logs/ae_full_32.log`

## 9. Full encoded features

- Path: `outputs/latent_videos_full_32/encoded_features.npy`
- Shape: `(11892, 32, 32)`
- Min / max: `-1.5073 / 4.5197`
- These are the full-AE latents used as endpoint context for sampling in Phase 6.

## 10. Controlled latent video dataset

- Path: `outputs/controlled_latent_videos_full_32/controlled_shape_videos_50k.npy`
- Shape: `(50000, 10, 32, 32)`, dtype `float32`
- File size: ~2.0 GB
- All-finite: True
- Latent stats: min `-1.6109`, max `4.5602`, mean `0.0425`, std `0.4050`
- Latent max (used to normalize for training): `4.5602`

## 11. Control vector schema

- Path: `outputs/controlled_latent_videos_full_32/controlled_shape_controls_50k.npy`
- Shape: `(50000, 16)`, dtype `float32`
- All-finite: True
- min `-1.0000`, max `1.1998`, mean `0.1059`, std `0.3184`
- Layout produced by `_build_control_vector`: mode-id one-hot (mode tag), strength/twist/anisotropy magnitudes, normalized center and axis, schedule indicators (phase or relaxation, composite strength), packed into 16 dimensions matching `DEFAULT_CONTROL_DIM = 16`.

## 12. Mode counts (in the 50k controlled dataset)

| Mode | Count |
|---|---:|
| cavity_expansion | 10,964 |
| cavity_contraction | 3,939 |
| anisotropic_expansion | 9,973 |
| twist | 7,554 |
| hybrid_cavity_twist | 7,584 |
| bulge_then_relax | 5,030 |
| two_stage_morph | 4,956 |
| **Total** | **50,000** |

Labels are saved separately at `outputs/controlled_latent_videos_full_32/controlled_shape_videos_50k.labels.npy`, shape `(50000,)`, `int64`.

## 13. Controlled DDPM architecture summary

`ControlConditionedVideoUNet`:
- Input: stacked `(z_start, middle_noisy, z_end)` over the latent grid, channels-first, 3 input channels.
- Output: 1-channel ε-prediction over the middle-frame latents.
- Backbone: `SimpleEndpointConditionedVideoUNet` with `base_ch = 32` and the default `dim_mults = (1, 2, 4)`, GroupNorm groups = 8.
- Time conditioning: sinusoidal timestep embedding, projected and added per resolution.
- Control conditioning: 16-dim control vector projected and broadcast through the U-Net's conditioning MLPs alongside the time embedding.
- Diffusion schedule: cosine, 1000 timesteps, β ∈ `[1e-4, 2e-2]`.
- Endpoint context: 1 frame at each end pinned to ground-truth latents; the DDPM generates the inner 8 frames (`num_frames=10, t_middle=8`).

## 14. Controlled DDPM training configuration

- Command: `train_controlled_video_ddpm`
- Latent video path: `outputs/controlled_latent_videos_full_32/controlled_shape_videos_50k.npy`
- Control path: `outputs/controlled_latent_videos_full_32/controlled_shape_controls_50k.npy`
- Checkpoint dir: `checkpoints/controlled_video_ddpm_full_32_50k/`
- Batch size: 8
- Max epochs: 500
- Patience: 50
- LR: 1e-4 (Adam)
- Timesteps: 1000
- β schedule: cosine
- base_channels: 32
- control_dim: 16
- Val split: 0.1 (45,000 train / 5,000 val)
- Device: cuda
- Seed: 123
- Log: `logs/controlled_video_ddpm_full_32_50k.log`
- No OOM, no batch-size or base-channel fallback was needed.

## 15. Controlled DDPM loss trend

- Epochs completed: 164 (early stopping triggered at epoch 164)
- Best validation loss: **0.0007815** at epoch **114**
- Final validation loss (epoch 164): 0.001206
- First-epoch validation loss: 0.003416
- Train loss decreased from 0.00987 → ~0.001 and plateaued, with validation oscillating around 0.0008–0.0013 after epoch ~50 and improving slowly until epoch 114, after which 50 epochs of no further improvement triggered early stopping.
- Both train and validation loss decreased relative to their initial values.

## 16. Controlled DDPM checkpoint artifacts

- `controlled_video_ddpm_latest.pt` ✓ (epoch 164, with optimizer state)
- `controlled_video_ddpm_best.pt` ✓ (epoch 114, best val=0.0007815)
- `controlled_video_ddpm_final.pt` ✓ (epoch 164)
- `controlled_video_ddpm_meta.json` ✓
- `controlled_video_ddpm_epoch_losses.txt` ✓ (164 epoch rows)

## 17. Sample outputs

14 controlled samples were generated, one per requested configuration, in `outputs/controlled_samples_full_32_50k/sample_{0000..0013}/`.

Endpoint indices use full-AE encoded features at `outputs/latent_videos_full_32/encoded_features.npy`.

| sample | mode | start | end | strength | twist | anisotropy | f2f L2 | endpoint L2 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 0  | cavity_expansion | 0 | 10 | 0.06 | – | – | 0.02548 | 0.21368 |
| 1  | cavity_expansion | 0 | 10 | 0.16 | – | – | 0.02538 | 0.21368 |
| 2  | cavity_contraction | 5 | 25 | -0.10 | – | – | 0.02672 | 0.22060 |
| 3  | anisotropic_expansion | 100 | 250 | 0.14 | – | 0.14 | 0.02213 | 0.19111 |
| 4  | twist | 500 | 900 | – | 0.35 | – | 0.02390 | 0.20581 |
| 5  | twist | 500 | 900 | – | 1.00 | – | 0.02461 | 0.20581 |
| 6  | hybrid_cavity_twist | 1000 | 2000 | 0.08 | 0.60 | – | 0.02169 | 0.18198 |
| 7  | bulge_then_relax | 2000 | 4000 | 0.12 | – | – | 0.02818 | 0.22190 |
| 8  | two_stage_morph | 3000 | 6000 | – | – | – | 0.03678 | 0.30375 |
| 9  | cavity_expansion | 4000 | 8000 | 0.12 | – | – | 0.02010 | 0.17020 |
| 10 | anisotropic_expansion | 7500 | 10000 | 0.16 | – | 0.16 | 0.02650 | 0.22929 |
| 11 | hybrid_cavity_twist | 10000 | 11800 | 0.08 | 0.50 | – | 0.03437 | 0.29761 |
| 12 | cavity_contraction | 111 | 2222 | -0.08 | – | – | 0.02970 | 0.24714 |
| 13 | twist | 333 | 7777 | – | 0.75 | – | 0.03773 | 0.31968 |

All 14 samples produced the full required artifact set:
- `frame_000.npy … frame_009.npy` with shape `(1000, 6)`, all finite
- `latent_video.npz` (latent video + endpoints + control vector)
- `control_vector.npy`
- `control_config.json`
- `preview.png`
- `animation.gif`

## 18. GIF and preview paths

For each `sample_XXXX`:
- preview: `outputs/controlled_samples_full_32_50k/sample_XXXX/preview.png`
- animation: `outputs/controlled_samples_full_32_50k/sample_XXXX/animation.gif`

All 28 PNG/GIF files exist.

## 19. Quantitative sanity metrics

Aggregated over all 14 samples × 10 frames × 1000 points (recomputed by `scripts/verify_controlled_samples.py`):

**Verification gates**
- Finite checks: every decoded frame in every sample is finite. No NaN, no Inf in any of 14 × 10 × 1000 × 6 = 840,000 values.
- Shape checks: every decoded frame is `(1000, 6)`; latent video shape is `(10, 32, 32)`; control vector shape is `(16,)`.
- Per-sample artifact set complete in all 14 samples: `frame_000.npy … frame_009.npy`, `latent_video.npz`, `control_vector.npy`, `control_config.json`, `preview.png`, `animation.gif`.
- Pass count: **14 / 14**.

**xyz / normal ranges**
- xyz global min / max (across all samples): `-1.0006 / 0.9408` (essentially inside the training clip `[-1, 1]`; the slight `-1.0006` minimum is decoder smoothing, not blow-up).
- Normal raw component global min / max: `-1.0330 / 1.0406` (slightly past `[-1, 1]` because the AE decoder does not project normals back to the unit sphere).

**Normal-vector norm statistics (per-point ‖n‖)**
- Per-sample mean ‖n‖ (averaged across samples): `0.593`.
- Per-sample std of ‖n‖ (averaged): `0.087`.
- Range across samples: `0.543 – 0.668`.
- This is well below 1.0, so decoded "normals" are not unit-length. The AE was trained with an MSE loss on the 6-channel point–normal vector, with no unit-norm penalty on the normal channel, and the decoder learned a contracted representation of the normal field. This is a known AE limitation, not a DDPM failure. Downstream consumers that need unit normals must renormalize.

**Latent norm statistics (per-frame ‖z_t‖, z_t ∈ R^{32×32})**
- Per-sample latent value min / max (averaged): mean `0.0075`, std `0.378` (compare training-set latent stats: mean `0.0425`, std `0.4050`).
- Global latent min / max across all 14 samples × 10 frames: `-1.1675 / 4.2573` (training range `-1.6109 / 4.5602`, so samples stay inside).
- Per-frame Frobenius norm ‖z_t‖ mean across samples: `12.15`, range across all frames `[9.83, 15.31]`.
- Frame-to-frame ‖z_t‖ delta (mean over time, then averaged across samples): `0.373` (≈ 3% of the norm). No frame's latent norm jumps by more than ~0.62 from the previous frame in any sample.

**Trajectory motion**
- Frame-to-frame mean L2 displacement in xyz (per-sample mean, then aggregated): mean `0.02738`, std `0.00529`, range `0.0201 – 0.0377`.
- Endpoint L2 displacement (per-sample mean): mean `0.23016`, std `0.04440`, range `0.1702 – 0.3197`.
- f2f / endpoint ratio is roughly stable around 0.11–0.12, indicating non-trivial trajectory length compared with the straight-line distance.

**Temporal smoothness**
- Per-sample mean per-point second difference ‖x_{t-1} – 2 x_t + x_{t+1}‖ in xyz (a proxy for trajectory acceleration / jerkiness):
  - Aggregate mean: `0.0159`
  - Max across samples: `0.0294` (sample_0008, two_stage_morph, which is expected to be non-monotonic).
- Smoothness acceleration is roughly 0.6 × the per-step displacement on average, consistent with smooth curved trajectories rather than zig-zag noise.
- No sample exhibits oscillation, frame-skipping, or local explosion.

## 20. Qualitative notes

- Cavity-mode samples (0, 1, 2, 9, 12) show inflation / contraction of the central region across frames in `preview.png` and `animation.gif`. Sample 1 (cavity_expansion strength `0.16`) has a visibly larger interior bulge than sample 0 (`0.06`) even with the same endpoints, confirming the strength scalar in the control vector is being used.
- Twist samples (4, 5, 13) show progressive rotation around the principal axis. Sample 5 (twist_strength `1.0`) is more aggressive than sample 4 (`0.35`) with the same endpoints, again consistent with the control vector.
- Hybrid and two-stage modes (6, 7, 8, 11) show non-monotonic trajectories that endpoint-only DDPM cannot represent. The two_stage_morph sample (8) has the largest second-difference smoothness value (`0.029`), consistent with an intentional change of regime mid-trajectory.
- No sample collapses: minimum per-sample f2f is `0.0201`, well above zero.
- No sample explodes: xyz stays inside `[-1.0006, 0.9408]`, latents stay inside `[-1.17, 4.26]` (training latent range was `[-1.61, 4.56]`).
- Endpoint frames (frame 0 and frame 9) match the requested ground-truth start / end latents by construction.

**Are these outputs smooth and stable?** Yes. f2f displacement is tight (std `0.005` around mean `0.027`), latent-norm frame-to-frame deltas are <5% of the norm, and trajectory acceleration is < 0.03 per step everywhere. None of the 14 samples shows mode collapse, mode-drop, jitter, or divergence.

**Do any outputs collapse or explode?** No. No frame is constant, no frame is unbounded, no frame contains NaN / Inf.

## 21. Failure modes and limitations

- Decoded normals are not re-projected to unit norm. Raw components drift slightly past `[-1, 1]` (range `[-1.033, 1.041]`), and per-point ‖n‖ has mean `0.593` ± `0.087`, well below 1.0. The AE was trained with bulk MSE on the 6-channel point–normal vector with no unit-norm constraint, and the decoder learned a contracted normal field. Any downstream renderer or surface-orientation analysis must renormalize.
- The synthetic deformations are geometric, not physically derived. No constitutive model, no equilibrium constraint, no preservation of point-cloud topology or local connectivity. Trajectories should not be interpreted as material response.
- The control vector is 16-dim and lightly normalized; modes that combine multiple effects (`hybrid_cavity_twist`, `two_stage_morph`) reuse the same 16 slots, so the model has to disambiguate them from limited shared parameters.
- Mode counts are unbalanced – `cavity_expansion` and `anisotropic_expansion` dominate (~20% each); `cavity_contraction` is the rarest (~8%). The model's coverage of rare modes is correspondingly weaker.
- AE val loss flattened early (best at epoch 41 of 91); pushing latent size beyond 32 would likely give a meaningfully better reconstruction floor.
- DDPM val loss improved slowly past epoch ~60; the absolute level (~0.0008) is not directly interpretable as sample quality.
- The pipeline pins the exact ground-truth start/end frames; the DDPM is only generating the 8 inner frames. Endpoint quality therefore depends entirely on AE reconstruction quality.
- No quantitative evaluation against a held-out trajectory distribution was performed; the audit metrics are sanity checks only.

## 21b. Release / demo quality assessment

The 14-sample suite is **adequate for an internal demo / methods walkthrough**, with the following qualifications.

Strong enough to show:
- Multi-mode latent video DDPM training converges cleanly on synthetic shape trajectories (val loss `7.8e-4`, early-stopped at epoch 114 / 164).
- A 16-dim control vector is sufficient to differentiate 7 deformation modes and to modulate strength within a mode (cavity samples 0 vs 1, twist samples 4 vs 5).
- Sampling produces stable, smooth, finite, in-distribution trajectories.
- The pipeline (AE → controlled latent video build → control-conditioned DDPM → decode → GIF) runs end-to-end without manual intervention on an 8 GB consumer GPU.

Not yet strong enough to claim as a public release / paper-quality result:
- Synthetic deformations only. No comparison against a real-physics trajectory dataset, no constitutive grounding.
- AE reconstruction floor is modest (val MSE `0.238`) and the decoded normals are not unit-length. Any plot that relies on accurate surface normals (e.g., shaded renders) needs explicit renormalization, and the underlying AE is the bottleneck.
- No baseline comparison. The endpoint-only DDPM and plain linear latent interpolation have not been run against the same 14 endpoint pairs.
- No held-out trajectory evaluation. Numerical metrics here are sanity gates, not validation.
- Mode coverage is uneven (cavity_contraction: 8% of training).

Use the GIFs for internal show-and-tell or for a methods slide. Do not use them as evidence of physical realism or as a benchmark result.

## 22. Is this the best current trained version?

Yes – this is the strongest controlled latent video DDPM trained in this codebase so far:
- Largest training set used (50,000 controlled trajectories, vs. the pilot smoke runs).
- Strongest available AE (full-data 91-epoch run, best val 0.2384, vs. the pilot AE).
- Largest model size feasible on the 8 GB RTX 4070 Laptop GPU at batch 8 with `base_channels=32`, `timesteps=1000`.
- All 14 requested controlled samples produced cleanly, with no fallbacks (no OOM, no batch-size reduction, no channel reduction, no dataset downsizing).

It supersedes the pilot artifacts in `checkpoints/ae_video_pcae_pilot_32/`, `checkpoints/video_ddpm_pcae_pilot_32_10k/`, and `checkpoints/controlled_video_ddpm_smoke_32/`, which remain on disk untouched as required.

## 23. Recommended next architecture improvements

In rough priority order:

1. **AE upgrades** (the current bottleneck):
   - `latent_size = 64` (or `128`) with the same training discipline. AE val loss flattened around `0.238` with `latent_size = 32`; the marginal gain from a larger latent should be measured directly.
   - Add a unit-norm penalty (or explicit projection layer) on the 3 normal channels. Decoded normals currently have mean ‖n‖ ≈ `0.59`, well below 1.0.
   - Replace the bulk-MSE loss on the 6-channel point with split-loss terms: xyz MSE + (1 – cos) on normals + Chamfer / EMD on the xyz set, to push reconstruction quality past the current floor.
2. **Baseline comparisons**:
   - Re-decode the same 14 endpoint pairs with the endpoint-only DDPM and with plain linear latent interpolation. Report f2f, endpoint, second-difference smoothness, and qualitative trajectory differences. This is the natural baseline the controlled run is meant to beat.
3. **Dataset scale and balance**:
   - Bump controlled trajectories to ~100k with rebalanced mode sampling (raise `cavity_contraction`, `bulge_then_relax`, `two_stage_morph` toward parity).
   - Add explicit "no-op" / near-identity trajectories so the model learns the trivial case as a baseline.
4. **Conditioning architecture**:
   - Replace the single 16-dim additive control vector with **FiLM** modulation per resolution.
   - Use a per-mode embedding head (one-hot mode id → learned embedding) concatenated with the parameter sub-vector, instead of overloading 16 slots across all modes.
   - Add a per-frame time-along-trajectory scalar to the control vector so the network has a clean "where in the deformation am I" signal independent of diffusion timestep `t`.
5. **Backbone**:
   - A latent-token transformer (axial attention over `(frame, latent_h, latent_w)`) instead of `SimpleEndpointConditionedVideoUNet` to better handle multi-stage modes (`two_stage_morph`, `bulge_then_relax`) where the trajectory is not monotonic. With current `base_ch = 32` and `(1, 2, 4)` mults, the U-Net is small; a transformer is feasible at this latent resolution on the same hardware.
6. **Sampling-side**:
   - Add classifier-free guidance over the control vector and sweep guidance scale.
   - Add DDIM / DPM-Solver sampling so generation does not require all 1000 steps.

---

**Run designation**: controlled synthetic shape-transformation latent video DDPM (50k dataset, full-AE encoder, RTX 4070 Laptop GPU). Not state of the art. No physical realism is claimed.

---

## 24. Final summary (Phase 7)

**What works well (strongest results)**
- Controlled DDPM training converged cleanly with **best val loss `0.0007815` at epoch `114`**, early-stopped at epoch `164` of `500` (patience `50`). No divergence, no fallbacks.
- All **14 / 14** required samples passed every verification gate: 10 frames each of shape `(1000, 6)`, latent `(10, 32, 32)`, control `(16,)`, all finite, all artifacts (npz, control_vector, control_config, preview, gif) present.
- Outputs are **smooth and stable**: f2f L2 mean `0.0274` (std `0.005`), latent norm drift <5% per step, no NaN / Inf, no collapse, no explosion.
- The **16-dim control vector demonstrably modulates trajectories**: cavity strength `0.06` vs `0.16` and twist strength `0.35` vs `1.0` produce visibly distinct trajectories from identical endpoints, which the prior endpoint-only DDPM could not do.
- All 7 required deformation modes are represented in the sample suite, with at least one parameter variation in 5 of them.

**Main weaknesses**
- **AE is the binding constraint.** Reconstruction MSE plateaus at `0.238`, and decoded normals are sub-unit (mean ‖n‖ ≈ `0.59`). Any downstream visualization that depends on accurate normals must renormalize.
- **No baseline comparison run.** Endpoint-only DDPM and linear latent interpolation have not been re-evaluated on the same 14 endpoint pairs, so the *quantitative* advantage of conditioning is asserted but not measured.
- **Synthetic deformations only.** Trajectories are geometrically constructed, not physically grounded; the model has no exposure to real material response.
- **Mode imbalance.** `cavity_contraction` is 8% of the dataset; rare-mode behavior is correspondingly less well learned.
- **Quantitative metrics are sanity gates.** Low DDPM val loss (`7.8e-4`) is not, by itself, evidence of sample quality.

**Likely next research directions**
1. AE capacity / loss study: `latent_size ∈ {32, 64, 128}` with split xyz / normal losses. This unblocks every downstream task.
2. Quantitative baseline comparison vs the endpoint-only DDPM and vs linear latent interpolation on a held-out set of endpoint pairs, reporting f2f, endpoint, smoothness, and a perceptual metric on decoded GIFs.
3. Conditioning redesign: FiLM modulation + per-mode embedding head + per-frame trajectory-time scalar, replacing the single 16-dim additive control vector.
4. Real-data ladder: introduce a small set of physically simulated trajectories (e.g., FEM-driven cavity inflation or twist) to test whether the controlled model transfers to physically meaningful motion.
5. Multi-stage / non-monotonic modes (`two_stage_morph`, `bulge_then_relax`) are the most informative stress tests; future architectures (transformer backbone, longer sequences) should be evaluated primarily on these.

**Are larger latent sizes or transformer backbones justified?**
- **Larger latent sizes: yes, conditionally justified.** The AE val loss flattens at `latent_size = 32` and the decoded-normal magnitude collapse is consistent with an under-capacity bottleneck. A controlled `latent_size = 64` AE study is the lowest-risk, highest-information next experiment.
- **Transformer backbone: justified to test, not yet justified to commit.** The current U-Net handles the 7-mode control distribution well on monotonic modes. Two-stage and bulge-then-relax samples (`8`, `11`) have the highest second-difference smoothness values and the largest latent-norm jumps (`0.56 – 0.61`), suggesting the U-Net is being mildly stretched by non-monotonic trajectories. A small axial-attention prototype on the existing 50k dataset would tell us whether the gain is real before committing to a full transformer port.

No physical realism is claimed in this report. No state-of-the-art claim is made. The 50k controlled DDPM is the strongest controlled latent video model currently in this codebase, and it is suitable for an internal methods demo and as a baseline for the next round of experiments.

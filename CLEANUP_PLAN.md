# CLEANUP_PLAN.md

Repo: `Latent_3D_Video/` (copied cleanup repo).
Goal: minimal repo for the **PC_AE_Video_DDPM** pipeline driven by
`DDPM_Video__PCAE.py` and `data/normalized_rotated_point_clouds6.npy`.

## Dependency summary

`DDPM_Video__PCAE.py` is self-contained. Only third-party imports are
`numpy`, `matplotlib`, `torch` (+ stdlib). It does **not** import any module
under `models/`, `diffusion/`, `utils/`, `metrics/`, `data/`, `scripts/`, or
the sibling top-level scripts (`train_video_ddpm.py`, `sample_video_ddpm.py`,
`DDPM_PCAE (1).py`, `evaluate_pilot_samples.py`, `make_pointcloud_video_previews.py`).

Files it reads/writes at runtime are all CLI-supplied paths. Defaults:

- input data: `data/normalized_rotated_point_clouds6.npy`
- AE ckpt dir: artifact names `ae_best.pth`, `ae_final.pth`, `ae_latest.pth`,
  `ae_meta.json`, `ae_epoch_losses.txt`, `epoch_viz/ae_epoch_XXXX.png`
- video DDPM ckpt dir: `video_ddpm_{best,final,latest}.pt`,
  `video_ddpm_meta.json`, `video_ddpm_epoch_losses.txt`
- controlled DDPM ckpt dir: `controlled_video_ddpm_{best,final,latest}.pt`,
  `controlled_video_ddpm_meta.json`, `controlled_video_ddpm_epoch_losses.txt`
- encoded features: `<dir>/encoded_features.npy`
- latent videos: `<dir>/*.npy`, `<dir>/*.meta.json`,
  `<dir>/*.labels.npy` (for controlled)
- samples: `<sample_dir>/sample_XXXX/{frame_XXX.npy, latent_video.npz,
  preview.png, animation.gif, control_vector.npy, control_config.json}`
- audit: `--output_report` markdown path

## ⚠️ Active training warning

`checkpoints/ae_video_pcae_full_32/` was modified at 14:13 (current run).
This whole directory and the data file it depends on are not touched.

## A. KEEP_REQUIRED

Files required to run/reproduce the PC_AE_Video_DDPM pipeline.

- `DDPM_Video__PCAE.py`
- `data/normalized_rotated_point_clouds6.npy`
- `LICENSE`
- `README.md` *(will be replaced by a new accurate README)*
- `checkpoints/ae_video_pcae_pilot_32/` *(active pilot AE — has best, final, latest, meta, losses, epoch_viz)*
- `checkpoints/ae_video_pcae_full_32/` *(ACTIVELY TRAINING; do not touch)*
- `checkpoints/video_ddpm_pcae_pilot_32_10k/` *(pilot endpoint-DDPM)*
- `checkpoints/controlled_video_ddpm_smoke_32/` *(smoke control-DDPM)*
- `outputs/latent_videos_pilot_32/encoded_features.npy`
- `outputs/latent_videos_pilot_32/latent_interpolation_videos_10k.npy`
- `outputs/latent_videos_pilot_32/latent_interpolation_videos_10k.meta.json`
- `outputs/video_samples_pilot_32_10k/` *(decoded pilot samples + eval json)*
- `outputs/controlled_latent_videos_32/controlled_smoke.npy`
- `outputs/controlled_latent_videos_32/controlled_smoke_controls.npy`
- `outputs/controlled_latent_videos_32/controlled_smoke.labels.npy`
- `outputs/controlled_latent_videos_32/controlled_smoke.meta.json`
- `outputs/controlled_samples_smoke_32/` *(decoded smoke samples)*

## B. KEEP_USEFUL

Helpful for documentation, monitoring, auditing. Not strictly required to run
the CLI, but directly tied to this pipeline.

- `evaluate_pilot_samples.py` *(reads `outputs/video_samples_pilot_32_10k/`)*
- `make_pointcloud_video_previews.py` *(rebuilds previews/gifs from same dir)*
- `logs/ae_full_32.log`
- `logs/ae_resume.log`
- `logs/build_videos.log`
- `logs/encode.log`
- `logs/video_ddpm.log`
- `logs/video_ddpm_resume.log`
- `logs/video_ddpm_resume_to40.log`
- `.claude/` *(IDE/agent config — empty but harmless)*

## C. ARCHIVE_CANDIDATE

Unrelated to PC_AE_Video_DDPM but not safe to delete outright. Move to
`cleanup_archive/<original-path>`.

### Old Kinetics video DDPM stack (not imported by `DDPM_Video__PCAE.py`)

- `train_video_ddpm.py`
- `sample_video_ddpm.py`
- `models/` *(attention.py, conditioning_encoder.py, diffusion_schedule.py, positional_encoding.py, resblocks.py, spatiotemporal_attention.py, temporal_attention.py, temporal_modules.py, video_unet3d.py)*
- `diffusion/` *(schedule.py)*
- `utils/` *(config.py, diagnostics.py, distributed.py, ema.py, io.py, logger.py)*
- `metrics/` *(fvd.py)*
- `configs/default.yaml`
- `runs/` *(tensorboard event files for Kinetics runs)*
- `scripts/setup_kinetics_downloader.sh`
- `scripts/smoke_subset.sh`
- `kinetics400_val_list_videos.txt` *(root copy)*
- `data/kinetics_video_dataset.py`
- `data/kinetics400_val_list_videos.txt`

### Old PointCloudAE / DDPM_PCAE artifacts (superseded by `DDPM_Video__PCAE.py`)

- `DDPM_PCAE (1).py`
- `data/PointCloud_AE.py`
- `data/PointCloudAE_aug.py`
- `data/NORMAL_MOD.ipynb`
- `data/find_bad_videos.py`

### Old data not referenced by current pipeline

- `data/2channel_experiment/` *(2channel_rotated_point_clouds6.npy)*
- `data/just points/` *(combined / node2vec / processed / scale_coefficients)*
- `data/normals_resampled/` *(7 variant normals files, 3.8 GB)*
- `data/rotated_4_times/`
- `data/rotated before normalize and centering/`
- `data/unaugmented 6/`
- `data/unused data/`
- `data/combined_pc_1000_normals.npy`
- `data/encoded_features_Justnormal.npy`
- `data/metrics_array.npy`
- `data/normalized_scale_coefficients6.npy`
- `data/obb_vectors.npy`
- `data/PointCloud_encoded_features_10k_03.npy`
- `data/processed_point_clouds.npy`

### Older smoke checkpoints/outputs superseded by pilot_32 / smoke_32

- `checkpoints/ae_video_pcae_real_smoke/`
- `checkpoints/video_ddpm_pcae_real_smoke/`
- `outputs/latent_videos_real_smoke/`
- `outputs/video_samples_real_smoke/`

## D. DELETE_CANDIDATE (still moved to archive first per safety rule)

Unambiguous junk that is regenerable.

- `__pycache__/` *(top-level)*
- `diffusion/__pycache__/`
- `metrics/__pycache__/`
- `models/__pycache__/`
- `utils/__pycache__/`
- `data/.ipynb_checkpoints/` *(empty)*
- `outputs/video_samples_pilot_32_10k_partial/` *(empty stub dir)*

## Validation plan

1. `python3 -m py_compile DDPM_Video__PCAE.py`
2. `python3 DDPM_Video__PCAE.py --help`
3. `python3 DDPM_Video__PCAE.py <subcmd> --help` for every subcommand
4. Confirm `data/normalized_rotated_point_clouds6.npy` shape is `(N, 1000, 6)`
5. Confirm smoke controlled artifacts still load

## Manual-review questions

- Should `cleanup_archive/` itself be deleted after review? (User explicit
  approval required.)
- Should the `epoch_viz/` images in checkpoints be slimmed (kept all)?
- Should we add a `requirements.txt`? *(not present today; not in scope)*

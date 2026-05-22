from pathlib import Path
import json
import numpy as np

ROOT = Path("outputs/video_samples_pilot_32_10k")
OUT = ROOT / "pilot_sample_eval.json"

summary = {}

for sample_dir in sorted(ROOT.glob("sample_*")):
    frames = [np.load(p).astype(np.float32) for p in sorted(sample_dir.glob("frame_*.npy"))]
    if not frames:
        continue

    arr = np.stack(frames, axis=0)  # [T, P, 6]
    xyz = arr[:, :, :3]
    normals = arr[:, :, 3:6]

    frame_l2 = []
    for t in range(len(frames) - 1):
        frame_l2.append(float(np.linalg.norm(xyz[t + 1] - xyz[t], axis=-1).mean()))

    endpoint_l2 = float(np.linalg.norm(xyz[-1] - xyz[0], axis=-1).mean())

    normal_norms = np.linalg.norm(normals, axis=-1)

    sample_summary = {
        "frame_count": int(arr.shape[0]),
        "frame_shape": list(arr.shape[1:]),
        "finite": bool(np.isfinite(arr).all()),
        "xyz_min": float(xyz.min()),
        "xyz_max": float(xyz.max()),
        "normal_min": float(normals.min()),
        "normal_max": float(normals.max()),
        "normal_norm_mean": float(normal_norms.mean()),
        "normal_norm_std": float(normal_norms.std()),
        "frame_to_frame_mean_l2": frame_l2,
        "frame_to_frame_mean_l2_mean": float(np.mean(frame_l2)),
        "frame_to_frame_mean_l2_std": float(np.std(frame_l2)),
        "endpoint_mean_l2": endpoint_l2,
        "latent_video_exists": bool((sample_dir / "latent_video.npz").exists()),
        "preview_exists": bool((sample_dir / "preview.png").exists()),
        "animation_gif_exists": bool((sample_dir / "animation.gif").exists()),
        "animation_mp4_exists": bool((sample_dir / "animation.mp4").exists()),
    }
    summary[sample_dir.name] = sample_summary

OUT.write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
print("\nsaved", OUT)

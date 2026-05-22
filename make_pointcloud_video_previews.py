from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter

ROOT = Path("outputs/video_samples_pilot_32_10k")

def set_equal_3d(ax, pts):
    xyz = pts[:, :3]
    mins = xyz.min(axis=0)
    maxs = xyz.max(axis=0)
    center = 0.5 * (mins + maxs)
    radius = 0.5 * np.max(maxs - mins)
    radius = max(float(radius), 1e-6)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)

for sample_dir in sorted(ROOT.glob("sample_*")):
    frames = [np.load(p) for p in sorted(sample_dir.glob("frame_*.npy"))]
    if not frames:
        continue

    all_pts = np.concatenate(frames, axis=0)

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")

    def update(i):
        ax.clear()
        pts = frames[i]
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=2)
        ax.set_title(f"{sample_dir.name} frame {i:02d}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        set_equal_3d(ax, all_pts)
        ax.view_init(elev=25, azim=35)
        return []

    anim = FuncAnimation(fig, update, frames=len(frames), interval=350, blit=False)

    gif_path = sample_dir / "animation.gif"
    anim.save(gif_path, writer=PillowWriter(fps=3))
    print("saved", gif_path)

    try:
        mp4_path = sample_dir / "animation.mp4"
        anim.save(mp4_path, writer=FFMpegWriter(fps=3))
        print("saved", mp4_path)
    except Exception as e:
        print("mp4 skipped for", sample_dir, "reason:", e)

    plt.close(fig)

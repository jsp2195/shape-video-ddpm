#!/usr/bin/env python3
"""
End-to-end PointCloudAE + endpoint-conditioned Latent Video DDPM pipeline.

Same PointCloudAE / dataset / training utilities as DDPM_PCAE.py. The 2D
latent image DDPM is replaced by a 3D U-Net that operates on latent video
volumes [B, C, T_middle, H, W], conditioned on first and last AE-latent
frames. Synthetic videos are built by linear interpolation between random
latent endpoints, so no MP4 / RGB / BAIR / Kinetics data is needed.

CLI commands:
    train_ae            – train PointCloudAE on point clouds
    encode_dataset      – encode all point clouds into AE latent grids
    build_latent_videos – build a synthetic latent interpolation video dataset
    train_video_ddpm    – train endpoint-conditioned latent video DDPM
    sample_video        – sample a latent video and decode each frame
    run_all             – run the full pipeline

Example commands:

python3 DDPM_Video__PCAE.py train_ae \\
  --pointcloud_path data/normalized_rotated_point_clouds6.npy \\
  --ae_ckpt_dir checkpoints/ae_video_pcae \\
  --latent_size 32 \\
  --ae_batch_size 16 \\
  --ae_epochs 300 \\
  --patience 30 \\
  --ae_lr 1e-4

python3 DDPM_Video__PCAE.py encode_dataset \\
  --pointcloud_path data/normalized_rotated_point_clouds6.npy \\
  --ae_ckpt_dir checkpoints/ae_video_pcae \\
  --encoded_features outputs/latent_videos/encoded_features.npy \\
  --encode_batch_size 64

python3 DDPM_Video__PCAE.py build_latent_videos \\
  --encoded_features outputs/latent_videos/encoded_features.npy \\
  --latent_video_path outputs/latent_videos/latent_interpolation_videos.npy \\
  --num_pairs 50000 \\
  --num_frames 10 \\
  --interpolation linear

python3 DDPM_Video__PCAE.py train_video_ddpm \\
  --latent_video_path outputs/latent_videos/latent_interpolation_videos.npy \\
  --video_ddpm_ckpt_dir checkpoints/video_ddpm_pcae \\
  --batch_size 32 \\
  --epochs 300 \\
  --patience 30 \\
  --lr 1e-4 \\
  --timesteps 1000 \\
  --beta_schedule cosine

python3 DDPM_Video__PCAE.py sample_video \\
  --encoded_features outputs/latent_videos/encoded_features.npy \\
  --ae_ckpt_dir checkpoints/ae_video_pcae \\
  --video_ddpm_ckpt_dir checkpoints/video_ddpm_pcae \\
  --pointcloud_path data/normalized_rotated_point_clouds6.npy \\
  --start_index 0 \\
  --end_index 10 \\
  --num_frames 10 \\
  --sample_dir outputs/video_samples

python3 DDPM_Video__PCAE.py run_all \\
  --pointcloud_path data/normalized_rotated_point_clouds6.npy \\
  --ae_ckpt_dir checkpoints/ae_video_pcae \\
  --encoded_features outputs/latent_videos/encoded_features.npy \\
  --latent_video_path outputs/latent_videos/latent_interpolation_videos.npy \\
  --video_ddpm_ckpt_dir checkpoints/video_ddpm_pcae \\
  --sample_dir outputs/video_samples \\
  --latent_size 32 \\
  --num_pairs 50000 \\
  --num_frames 10
"""
import argparse
import json
import logging
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as tnn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# -----------------------------
# Defaults
# -----------------------------
DEFAULT_POINTCLOUDS = "data/normalized_rotated_point_clouds6.npy"
DEFAULT_AE_CKPT_DIR = "checkpoints/ae_video_pcae"
DEFAULT_ENCODED_FEATURES = "outputs/latent_videos/encoded_features.npy"
DEFAULT_LATENT_VIDEO_PATH = "outputs/latent_videos/latent_interpolation_videos.npy"
DEFAULT_VIDEO_DDPM_CKPT_DIR = "checkpoints/video_ddpm_pcae"
DEFAULT_VIDEO_SAMPLE_DIR = "outputs/video_samples"

DEFAULT_NUM_FRAMES = 10
DEFAULT_ENDPOINT_CONTEXT = 1
DEFAULT_BASE_CHANNELS = 64
DEFAULT_TIMESTEPS = 1000
DEFAULT_BETA_START = 1e-4
DEFAULT_BETA_END = 0.02


# -----------------------------
# Utility functions
# -----------------------------
def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    logging.basicConfig(level=level, format="[%(asctime)s] %(levelname)s: %(message)s")
    root.setLevel(level)
    for noisy in ("matplotlib", "PIL", "matplotlib.font_manager", "matplotlib.pyplot"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_file(path: str) -> None:
    if not Path(path).exists():
        raise FileNotFoundError(f"Required file not found: {path}")


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def save_json(data: Dict, path: str) -> None:
    ensure_dir(str(Path(path).parent))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def append_epoch_loss_txt(path: str, epoch: int, train_loss: float, val_loss: float) -> None:
    ensure_dir(str(Path(path).parent))
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"{epoch}\t{train_loss:.8f}\t{val_loss:.8f}\n")


# -----------------------------
# Point-cloud dataset utilities (mirrors DDPM_PCAE.py)
# -----------------------------
class PointCloudDataset(Dataset):
    def __init__(self, arr: np.ndarray):
        if arr.ndim != 3 or arr.shape[-1] != 6:
            raise ValueError(f"Point cloud array must have shape (N,P,6), got {arr.shape}")
        self.data = torch.from_numpy(arr.astype(np.float32))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def get_pointcloud_loaders(
    path: str,
    batch_size: int,
    val_split: float,
    seed: int,
    max_train_samples: Optional[int] = None,
    max_val_samples: Optional[int] = None,
):
    ensure_file(path)
    arr = np.load(path).astype(np.float32)

    n_total = len(arr)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n_total)
    n_val_full = int(n_total * val_split)
    n_train_full = n_total - n_val_full

    train_idx_full = idx[:n_train_full]
    val_idx_full = idx[n_train_full:]
    train_cap = len(train_idx_full) if max_train_samples is None else min(len(train_idx_full), max_train_samples)
    val_cap = len(val_idx_full) if max_val_samples is None else min(len(val_idx_full), max_val_samples)

    train_ds = PointCloudDataset(arr[train_idx_full[:train_cap]])
    val_ds = PointCloudDataset(arr[val_idx_full[:val_cap]])

    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=pin)
    return train_loader, val_loader, arr.shape[1]


# -----------------------------
# Point-cloud AE (mirrors DDPM_PCAE.py)
# -----------------------------
class SelfAttention(tnn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.query_conv = tnn.Conv1d(in_channels, max(in_channels // 8, 1), 1)
        self.key_conv = tnn.Conv1d(in_channels, max(in_channels // 8, 1), 1)
        self.value_conv = tnn.Conv1d(in_channels, in_channels, 1)
        self.softmax = tnn.Softmax(dim=-1)

    def forward(self, x):
        q = self.query_conv(x)
        k = self.key_conv(x)
        v = self.value_conv(x)
        d_k = q.shape[1]
        attn = self.softmax(torch.bmm(q.permute(0, 2, 1), k) / (d_k ** 0.5))
        out = torch.bmm(v, attn.permute(0, 2, 1))
        return out + x


class FeatureFusion(tnn.Module):
    def __init__(self, in_xyz, in_normal, out_channels):
        super().__init__()
        self.fusion_conv = tnn.Conv1d(in_xyz + in_normal, out_channels, 1)
        self.bn = tnn.BatchNorm1d(out_channels)

    def forward(self, xyz_features, normal_features):
        combined = torch.cat((xyz_features, normal_features), dim=1)
        return F.relu(self.bn(self.fusion_conv(combined)))


class PointCloudAE(tnn.Module):
    def __init__(self, point_size: int, latent_size: int):
        super().__init__()
        self.point_size = point_size
        self.latent_size = latent_size
        feature_size = latent_size ** 2

        self.conv1_xyz = tnn.Conv1d(3, 64, 1)
        self.conv2_xyz = tnn.Conv1d(64, 128, 1)
        self.conv3_xyz = tnn.Conv1d(128, feature_size // 2, 1)
        self.bn1_xyz = tnn.BatchNorm1d(64)
        self.bn2_xyz = tnn.BatchNorm1d(128)
        self.bn3_xyz = tnn.BatchNorm1d(feature_size // 2)

        self.conv1_normal = tnn.Conv1d(3, 32, 1)
        self.conv2_normal = tnn.Conv1d(32, 64, 1)
        self.conv3_normal = tnn.Conv1d(64, feature_size // 2, 1)
        self.bn1_normal = tnn.BatchNorm1d(32)
        self.bn2_normal = tnn.BatchNorm1d(64)
        self.bn3_normal = tnn.BatchNorm1d(feature_size // 2)

        self.feature_fusion = FeatureFusion(feature_size // 2, feature_size // 2, feature_size)
        self.self_attention = SelfAttention(feature_size)

        self.fc1 = tnn.Linear(feature_size, 1024)
        self.fc2 = tnn.Linear(1024, 2048)
        self.fc3 = tnn.Linear(2048, point_size * 6)
        self.fc_res1 = tnn.Linear(1024, 2048)
        self.fc_res2 = tnn.Linear(2048, point_size * 6)
        self.act = tnn.LeakyReLU(negative_slope=0.1)

    def encoder(self, x):
        xyz, normals = torch.split(x, 3, dim=1)
        xyz = self.act(self.bn1_xyz(self.conv1_xyz(xyz)))
        xyz = self.act(self.bn2_xyz(self.conv2_xyz(xyz)))
        xyz = self.act(self.bn3_xyz(self.conv3_xyz(xyz)))

        normals = self.act(self.bn1_normal(self.conv1_normal(normals)))
        normals = self.act(self.bn2_normal(self.conv2_normal(normals)))
        normals = self.act(self.bn3_normal(self.conv3_normal(normals)))

        fused = self.feature_fusion(xyz, normals)
        x = self.self_attention(fused)
        x = F.adaptive_max_pool1d(x, 1)
        return x.view(-1, self.latent_size, self.latent_size)

    def decoder(self, x):
        x = x.view(-1, self.latent_size ** 2)
        x = self.act(self.fc1(x))
        res1 = x
        x = self.act(self.fc2(x))
        x = x + self.fc_res1(res1)
        res2 = x
        x = self.fc3(x)
        x = x + self.fc_res2(res2)
        return x.view(-1, self.point_size, 6)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


# -----------------------------
# AE train / encode / load
# -----------------------------
def _chamfer_normal_loss(recon, target, lambda_p=10.0):
    coords_out = recon[:, :, :3]
    coords_tgt = target[:, :, :3]

    diff = coords_out.unsqueeze(2) - coords_tgt.unsqueeze(1)
    dist = (diff ** 2).sum(-1)

    min_dist_x = torch.min(dist, dim=2)[0]
    min_dist_y = torch.min(dist, dim=1)[0]
    chamfer = torch.mean(min_dist_x, dim=1) + torch.mean(min_dist_y, dim=1)
    chamfer = chamfer.mean()

    if recon.shape[-1] >= 6 and target.shape[-1] >= 6:
        normals_out = F.normalize(recon[:, :, 3:6], dim=-1)
        normals_tgt = F.normalize(target[:, :, 3:6], dim=-1)

        idx_x = torch.argmin(dist, dim=2)
        idx_y = torch.argmin(dist, dim=1)

        nearest_y_n = torch.gather(normals_tgt, 1, idx_x.unsqueeze(-1).expand(-1, -1, 3))
        nearest_x_n = torch.gather(normals_out, 1, idx_y.unsqueeze(-1).expand(-1, -1, 3))

        cos_x = (normals_out * nearest_y_n).sum(-1)
        cos_y = (normals_tgt * nearest_x_n).sum(-1)

        nloss = (1 - torch.abs(cos_x)).mean(1) + (1 - torch.abs(cos_y)).mean(1)
        nloss = nloss.mean()
    else:
        nloss = torch.tensor(0.0, device=recon.device)

    return lambda_p * chamfer + nloss


def train_epoch_ae(model, loader, optimizer, device):
    model.train()
    losses = []
    for batch in loader:
        batch = batch.to(device)
        inp = batch.permute(0, 2, 1)
        recon = model(inp)
        loss = _chamfer_normal_loss(recon, batch)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return float(np.mean(losses)) if losses else float("inf")


def validate_epoch_ae(model, loader, device):
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            inp = batch.permute(0, 2, 1)
            recon = model(inp)
            loss = _chamfer_normal_loss(recon, batch)
            losses.append(loss.item())
    return float(np.mean(losses)) if losses else float("inf")


def save_ae_epoch_visual(model, fixed_batch, device, out_path: str, fig_count: int = 4) -> None:
    was_train = model.training
    model.eval()
    with torch.no_grad():
        batch = fixed_batch.to(device)
        inp = batch.permute(0, 2, 1).contiguous()
        latent = model.encoder(inp)
        recon = model(inp)

    gt_np = batch.detach().cpu().numpy()
    latent_np = latent.detach().cpu().numpy()
    recon_np = recon.detach().cpu().numpy()

    n = min(fig_count, gt_np.shape[0])
    if n <= 0:
        return

    fig = plt.figure(figsize=(3.2 * n, 8.5))
    for i in range(n):
        ax1 = fig.add_subplot(3, n, i + 1, projection="3d")
        ax1.scatter(gt_np[i, :, 0], gt_np[i, :, 1], gt_np[i, :, 2], s=1, c="#2F6DB3")
        ax1.set_title(f"GT {i}")
        ax1.axis("off")

        ax2 = fig.add_subplot(3, n, n + i + 1)
        im = ax2.imshow(latent_np[i], cmap="coolwarm")
        ax2.set_title(f"Latent {i}")
        ax2.axis("off")
        fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)

        ax3 = fig.add_subplot(3, n, 2 * n + i + 1, projection="3d")
        ax3.scatter(recon_np[i, :, 0], recon_np[i, :, 1], recon_np[i, :, 2], s=1, c="#E68613")
        ax3.set_title(f"Recon {i}")
        ax3.axis("off")

    ensure_dir(str(Path(out_path).parent))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    if was_train:
        model.train()


def train_ae(args) -> None:
    max_train_samples = getattr(args, "max_train_samples", None)
    max_val_samples = getattr(args, "max_val_samples", None)
    ensure_dir(args.ae_ckpt_dir)
    viz_dir = os.path.join(args.ae_ckpt_dir, "epoch_viz")
    ensure_dir(viz_dir)

    train_loader, val_loader, point_size = get_pointcloud_loaders(
        args.pointcloud_path,
        args.ae_batch_size,
        args.val_split,
        args.seed,
        max_train_samples=max_train_samples,
        max_val_samples=max_val_samples,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = PointCloudAE(point_size=point_size, latent_size=args.latent_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.ae_lr)
    fixed_val_batch = next(iter(val_loader), None)

    start_epoch = 1
    best_val = float("inf")
    latest_path = os.path.join(args.ae_ckpt_dir, "ae_latest.pth")
    loss_txt_path = os.path.join(args.ae_ckpt_dir, "ae_epoch_losses.txt")
    resumed = os.path.exists(latest_path)

    if not resumed:
        with open(loss_txt_path, "w", encoding="utf-8") as f:
            f.write("epoch\ttrain_loss\tval_loss\n")
    elif not os.path.exists(loss_txt_path):
        with open(loss_txt_path, "w", encoding="utf-8") as f:
            f.write("epoch\ttrain_loss\tval_loss\n")

    if resumed:
        state = torch.load(latest_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_epoch = state["epoch"] + 1
        best_val = state.get("best_val", best_val)
        logging.info("Restored AE from epoch %d", state["epoch"])

    no_improve = 0
    for epoch in range(start_epoch, args.ae_epochs + 1):
        tr = train_epoch_ae(model, train_loader, optimizer, device)
        va = validate_epoch_ae(model, val_loader, device)
        logging.info("AE epoch %d/%d | train=%.6f val=%.6f", epoch, args.ae_epochs, tr, va)
        append_epoch_loss_txt(loss_txt_path, epoch, tr, va)

        if fixed_val_batch is not None:
            save_ae_epoch_visual(
                model,
                fixed_val_batch,
                device,
                os.path.join(viz_dir, f"ae_epoch_{epoch:04d}.png"),
            )

        torch.save(
            {"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "best_val": best_val},
            latest_path,
        )

        if va < best_val:
            best_val = va
            no_improve = 0
            torch.save(model.state_dict(), os.path.join(args.ae_ckpt_dir, "ae_best.pth"))
        else:
            no_improve += 1
            if no_improve >= args.patience:
                logging.info("AE early stopping at epoch %d", epoch)
                break

    torch.save(model.state_dict(), os.path.join(args.ae_ckpt_dir, "ae_final.pth"))
    save_json(
        {"point_size": point_size, "latent_size": args.latent_size},
        os.path.join(args.ae_ckpt_dir, "ae_meta.json"),
    )
    logging.info("AE training complete")


def load_ae_for_decode(ae_ckpt_dir: str, prefer_best: bool = True, device: Optional[torch.device] = None) -> PointCloudAE:
    meta = load_json(os.path.join(ae_ckpt_dir, "ae_meta.json"))
    point_size = int(meta["point_size"])
    latent_size = int(meta["latent_size"])
    model = PointCloudAE(point_size=point_size, latent_size=latent_size)

    weights_path = os.path.join(ae_ckpt_dir, "ae_best.pth" if prefer_best else "ae_final.pth")
    if not os.path.exists(weights_path):
        weights_path = os.path.join(ae_ckpt_dir, "ae_final.pth")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"No AE checkpoint found in {ae_ckpt_dir}")
    model.load_state_dict(torch.load(weights_path, map_location=device or "cpu", weights_only=True))
    model.eval()
    if device is not None:
        model.to(device)
    return model


def encode_dataset(args) -> None:
    ensure_file(args.pointcloud_path)
    ensure_dir(str(Path(args.encoded_features).parent))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ae = load_ae_for_decode(args.ae_ckpt_dir, prefer_best=True, device=device)
    ae.eval()

    arr = np.load(args.pointcloud_path).astype(np.float32)
    if arr.ndim != 3 or arr.shape[-1] != 6:
        raise ValueError(f"Point cloud array must have shape (N,P,6), got {arr.shape}")
    if arr.shape[1] != ae.point_size:
        raise ValueError(
            f"Point count mismatch: dataset has {arr.shape[1]} points per cloud, "
            f"but AE expects {ae.point_size}"
        )

    batch_size = int(args.encode_batch_size)
    latents = []

    with torch.no_grad():
        for start in range(0, len(arr), batch_size):
            end = min(start + batch_size, len(arr))
            batch = torch.from_numpy(arr[start:end]).to(device)
            batch = batch.permute(0, 2, 1).contiguous()
            z = ae.encoder(batch)
            latents.append(z.cpu().numpy().astype(np.float32))

    encoded = np.concatenate(latents, axis=0)
    np.save(args.encoded_features, encoded)

    logging.info("Encoded dataset saved to %s", args.encoded_features)
    logging.info("Encoded shape: %s", encoded.shape)
    logging.info("Encoded min/max: %.6f / %.6f", float(encoded.min()), float(encoded.max()))


# -----------------------------
# Latent video building
# -----------------------------
def build_latent_interpolation_videos(args) -> None:
    ensure_file(args.encoded_features)
    encoded = np.load(args.encoded_features).astype(np.float32)
    if encoded.ndim != 3:
        raise ValueError(f"Expected encoded_features shape (N, H, W), got {encoded.shape}")
    n_total, H, W = encoded.shape
    if H != W:
        raise ValueError(f"Expected square latent grids, got ({H}, {W})")
    if n_total < 2:
        raise ValueError(f"Need at least 2 encoded latents to build pairs, got {n_total}")

    num_pairs = int(args.num_pairs)
    num_frames = int(args.num_frames)
    if num_frames < 2:
        raise ValueError(f"num_frames must be >= 2, got {num_frames}")
    if args.interpolation != "linear":
        raise ValueError(f"Unsupported interpolation mode: {args.interpolation}")

    rng = np.random.default_rng(args.seed)
    idx_a = rng.integers(0, n_total, size=num_pairs)
    idx_b = rng.integers(0, n_total, size=num_pairs)
    same = idx_a == idx_b
    # Resample to avoid identical endpoints; bounded loop since n_total >= 2.
    for _ in range(64):
        if not np.any(same):
            break
        idx_b[same] = rng.integers(0, n_total, size=int(same.sum()))
        same = idx_a == idx_b
    # Anything still identical, manually shift by 1 mod n.
    if np.any(same):
        idx_b[same] = (idx_b[same] + 1) % n_total

    z_a = encoded[idx_a]  # [num_pairs, H, W]
    z_b = encoded[idx_b]  # [num_pairs, H, W]
    alphas = np.linspace(0.0, 1.0, num_frames, dtype=np.float32)  # [T]

    # videos[i, t] = (1 - a_t) * z_a[i] + a_t * z_b[i]
    a = alphas[None, :, None, None]
    videos = ((1.0 - a) * z_a[:, None, :, :] + a * z_b[:, None, :, :]).astype(np.float32)

    if videos.shape != (num_pairs, num_frames, H, W):
        raise RuntimeError(
            f"Constructed video shape {videos.shape} does not match expected "
            f"({num_pairs}, {num_frames}, {H}, {W})"
        )

    latent_max = float(max(np.max(np.abs(videos)), 1e-6))

    ensure_dir(str(Path(args.latent_video_path).parent))
    np.save(args.latent_video_path, videos)

    meta = {
        "latent_max": latent_max,
        "latent_size": int(H),
        "num_frames": int(num_frames),
        "num_pairs": int(num_pairs),
        "interpolation": str(args.interpolation),
        "seed": int(args.seed),
        "video_shape": list(videos.shape),
    }
    meta_path = str(Path(args.latent_video_path).with_suffix(".meta.json"))
    save_json(meta, meta_path)

    logging.info(
        "Saved %d latent interpolation videos shape=%s to %s",
        num_pairs, videos.shape, args.latent_video_path,
    )
    logging.info("Latent max (videos): %.6f", latent_max)


# -----------------------------
# Latent video dataset
# -----------------------------
class LatentVideoDataset(Dataset):
    """Returns (middle, start, end) tensors per item.

    middle: [1, T_middle, H, W]
    start:  [1, endpoint_context, H, W]
    end:    [1, endpoint_context, H, W]
    """

    def __init__(self, videos: np.ndarray, latent_max: float, endpoint_context: int = 1):
        if videos.ndim != 4:
            raise ValueError(f"videos must be 4D (N, T, H, W), got {videos.shape}")
        T = videos.shape[1]
        ec = int(endpoint_context)
        if ec < 1 or 2 * ec >= T:
            raise ValueError(f"endpoint_context={ec} invalid for T={T}")
        self.videos = videos.astype(np.float32)
        self.latent_max = float(latent_max)
        self.endpoint_context = ec
        self.T = T

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        v = self.videos[idx]  # [T, H, W]
        T = self.T
        ec = self.endpoint_context
        v_norm = np.clip(v / self.latent_max, -1.0, 1.0).astype(np.float32)

        start = v_norm[0:ec]            # [ec, H, W]
        end = v_norm[T - ec:T]          # [ec, H, W]
        middle = v_norm[ec:T - ec]      # [T - 2*ec, H, W]

        start_t = torch.from_numpy(start).unsqueeze(0)    # [1, ec, H, W]
        end_t = torch.from_numpy(end).unsqueeze(0)        # [1, ec, H, W]
        middle_t = torch.from_numpy(middle).unsqueeze(0)  # [1, T-2*ec, H, W]
        return middle_t, start_t, end_t


# -----------------------------
# Video diffusion model components
# -----------------------------
class SinusoidalPosEmb(tnn.Module):
    def __init__(self, dim, max_positions=10000):
        super().__init__()
        self.dim = dim
        self.max_positions = max_positions

    def forward(self, x):
        # x: [B] float32 of diffusion timesteps
        x = x.to(torch.float32)
        half_dim = self.dim // 2
        inv_freq_scale = math.log(self.max_positions) / max(half_dim - 1, 1)
        positions = torch.arange(half_dim, device=x.device, dtype=x.dtype)
        emb = torch.exp(positions * -inv_freq_scale)
        emb = x[:, None] * emb[None, :]
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)


class ResBlock3D(tnn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int, groups: int = 8):
        super().__init__()
        g_out = min(groups, out_ch)
        self.conv1 = tnn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1)
        self.norm1 = tnn.GroupNorm(g_out, out_ch)
        self.conv2 = tnn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1)
        self.norm2 = tnn.GroupNorm(g_out, out_ch)
        self.time_mlp = tnn.Linear(time_dim, out_ch)
        self.act = tnn.SiLU()
        self.res_conv = tnn.Conv3d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else tnn.Identity()

    def forward(self, x, t_emb):
        h = self.act(self.norm1(self.conv1(x)))
        t_proj = self.time_mlp(self.act(t_emb))[:, :, None, None, None]
        h = h + t_proj
        h = self.act(self.norm2(self.conv2(h)))
        return h + self.res_conv(x)


class Downsample3D(tnn.Module):
    def __init__(self, ch: int):
        super().__init__()
        # Halve spatial dims, preserve T.
        self.conv = tnn.Conv3d(ch, ch, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1))

    def forward(self, x):
        return self.conv(x)


class Upsample3D(tnn.Module):
    def __init__(self, ch: int):
        super().__init__()
        # Double spatial dims, preserve T.
        self.conv = tnn.ConvTranspose3d(ch, ch, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1))

    def forward(self, x):
        return self.conv(x)


class SimpleEndpointConditionedVideoUNet(tnn.Module):
    """Compact endpoint-conditioned 3D U-Net.

    Forward signature:
        forward(x_noisy, t, z_start, z_end)
    where
        x_noisy: [B, 1, T_middle, H, W]
        t      : [B] (long) diffusion timesteps
        z_start: [B, 1, T_ec, H, W]
        z_end  : [B, 1, T_ec, H, W]
    The endpoint context frames are temporally averaged to a single [B, 1, 1, H, W]
    map (cheap and shape-agnostic), then broadcast across T_middle and concatenated
    as extra channels alongside x_noisy → 3-channel 3D U-Net input.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        base_ch: int = 64,
        dim_mults: Sequence[int] = (1, 2, 4),
        groups: int = 8,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_ch = base_ch
        self.dim_mults = tuple(dim_mults)

        time_dim = base_ch * 4
        self.time_dim = time_dim
        self.time_mlp = tnn.Sequential(
            SinusoidalPosEmb(base_ch),
            tnn.Linear(base_ch, time_dim),
            tnn.SiLU(),
            tnn.Linear(time_dim, time_dim),
        )

        # chs[0] = base channels after init_conv, then one entry per level.
        chs = [base_ch] + [base_ch * m for m in dim_mults]
        self.chs = chs
        self.init_conv = tnn.Conv3d(in_channels, base_ch, kernel_size=3, padding=1)

        self.downs = tnn.ModuleList()
        for i in range(len(dim_mults)):
            in_ch = chs[i]
            out_ch = chs[i + 1]
            is_last = i == len(dim_mults) - 1
            self.downs.append(tnn.ModuleList([
                ResBlock3D(in_ch, out_ch, time_dim, groups=groups),
                ResBlock3D(out_ch, out_ch, time_dim, groups=groups),
                Downsample3D(out_ch) if not is_last else tnn.Identity(),
            ]))

        mid_ch = chs[-1]
        self.mid1 = ResBlock3D(mid_ch, mid_ch, time_dim, groups=groups)
        self.mid2 = ResBlock3D(mid_ch, mid_ch, time_dim, groups=groups)

        self.ups = tnn.ModuleList()
        for i in range(len(dim_mults) - 1, -1, -1):
            in_ch = chs[i + 1]
            out_ch = chs[i + 1] if i == len(dim_mults) - 1 else chs[i + 1]
            # We collapse to chs[i] on the resblock output for next-level concat.
            out_ch = chs[i] if i > 0 else chs[0]
            skip_ch = chs[i + 1]
            is_first_up = i == len(dim_mults) - 1
            is_last_up = i == 0
            self.ups.append(tnn.ModuleList([
                ResBlock3D(in_ch + skip_ch, out_ch, time_dim, groups=groups),
                ResBlock3D(out_ch, out_ch, time_dim, groups=groups),
                Upsample3D(out_ch) if not is_last_up else tnn.Identity(),
            ]))
            _ = is_first_up  # placeholder; kept for clarity

        self.final_norm = tnn.GroupNorm(min(groups, chs[0]), chs[0])
        self.final_act = tnn.SiLU()
        self.final_conv = tnn.Conv3d(chs[0], out_channels, kernel_size=1)

    @staticmethod
    def _endpoint_to_single_frame(z: torch.Tensor) -> torch.Tensor:
        # z: [B, 1, T_ec, H, W] -> [B, 1, 1, H, W] via mean over T_ec.
        if z.dim() != 5:
            raise ValueError(f"endpoint tensor must be 5D, got shape {tuple(z.shape)}")
        return z.mean(dim=2, keepdim=True)

    def forward(self, x_noisy, t, z_start, z_end):
        if x_noisy.dim() != 5:
            raise ValueError(f"x_noisy must be 5D [B,C,T,H,W], got {tuple(x_noisy.shape)}")
        B, C_in, T, H, W = x_noisy.shape
        if C_in != 1:
            raise ValueError(f"x_noisy must have C=1, got C={C_in}")

        z_start_1 = self._endpoint_to_single_frame(z_start)  # [B, 1, 1, H, W]
        z_end_1 = self._endpoint_to_single_frame(z_end)
        if z_start_1.shape[-2:] != (H, W) or z_end_1.shape[-2:] != (H, W):
            raise ValueError(
                f"endpoint spatial dims must match x_noisy; got "
                f"start={tuple(z_start_1.shape)}, end={tuple(z_end_1.shape)}, x={tuple(x_noisy.shape)}"
            )

        start_rep = z_start_1.expand(-1, -1, T, -1, -1)
        end_rep = z_end_1.expand(-1, -1, T, -1, -1)
        x = torch.cat([x_noisy, start_rep, end_rep], dim=1)  # [B, 3, T, H, W]

        t_emb = self.time_mlp(t)

        x = self.init_conv(x)
        skips = []
        for resblock1, resblock2, downsample in self.downs:
            x = resblock1(x, t_emb)
            x = resblock2(x, t_emb)
            skips.append(x)
            x = downsample(x)

        x = self.mid1(x, t_emb)
        x = self.mid2(x, t_emb)

        for resblock1, resblock2, upsample in self.ups:
            skip = skips.pop()
            x = torch.cat([x, skip], dim=1)
            x = resblock1(x, t_emb)
            x = resblock2(x, t_emb)
            x = upsample(x)

        x = self.final_act(self.final_norm(x))
        return self.final_conv(x)


# -----------------------------
# Video diffusion schedule and trainer
# -----------------------------
class VideoDiffusionSchedule:
    def __init__(
        self,
        timesteps: int = DEFAULT_TIMESTEPS,
        beta_schedule: str = "cosine",
        beta_start: float = DEFAULT_BETA_START,
        beta_end: float = DEFAULT_BETA_END,
        device: Optional[torch.device] = None,
    ):
        self.timesteps = int(timesteps)
        self.beta_schedule = beta_schedule
        self.device = device or torch.device("cpu")

        if beta_schedule == "linear":
            betas = torch.linspace(beta_start, beta_end, self.timesteps, dtype=torch.float32)
        elif beta_schedule == "cosine":
            betas = self._cosine_betas(self.timesteps).to(torch.float32)
        else:
            raise ValueError(f"Unknown beta_schedule: {beta_schedule}")

        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)

        self.betas = betas.to(self.device)
        self.alphas = alphas.to(self.device)
        self.alpha_bars = alpha_bars.to(self.device)
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - self.alpha_bars)

    @staticmethod
    def _cosine_betas(timesteps: int, s: float = 0.008) -> torch.Tensor:
        steps = timesteps + 1
        t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
        f_t = torch.cos((t + s) / (1 + s) * math.pi / 2.0) ** 2
        alpha_bars = f_t / f_t[0]
        betas = 1.0 - alpha_bars[1:] / alpha_bars[:-1]
        return torch.clamp(betas, 0.0, 0.999)


class LatentVideoDDPMTrainer:
    def __init__(self, model: SimpleEndpointConditionedVideoUNet, schedule: VideoDiffusionSchedule):
        self.model = model
        self.schedule = schedule
        self.device = next(model.parameters()).device

    def _gather(self, vec: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        out = vec.to(self.device)[t]
        return out[:, None, None, None, None]

    def forward_noise(self, x0: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if x0.dim() != 5:
            raise ValueError(f"x0 must be 5D [B,C,T,H,W], got {tuple(x0.shape)}")
        noise = torch.randn_like(x0)
        sa = self._gather(self.schedule.sqrt_alpha_bars, t).to(x0.dtype)
        osa = self._gather(self.schedule.sqrt_one_minus_alpha_bars, t).to(x0.dtype)
        return sa * x0 + osa * noise, noise

    def train_step(self, opt, middle_x0, z_start, z_end):
        B = middle_x0.shape[0]
        t = torch.randint(0, self.schedule.timesteps, (B,), device=middle_x0.device, dtype=torch.long)
        xt, noise = self.forward_noise(middle_x0, t)
        opt.zero_grad(set_to_none=True)
        pred = self.model(xt, t, z_start, z_end)
        loss = F.mse_loss(pred, noise)
        loss.backward()
        opt.step()
        return loss

    def val_step(self, middle_x0, z_start, z_end):
        B = middle_x0.shape[0]
        t = torch.randint(0, self.schedule.timesteps, (B,), device=middle_x0.device, dtype=torch.long)
        xt, noise = self.forward_noise(middle_x0, t)
        pred = self.model(xt, t, z_start, z_end)
        return F.mse_loss(pred, noise)

    def reverse_step(self, x_t, pred_noise, t, add_noise: bool = True):
        a_t = self._gather(self.schedule.alphas, t).to(x_t.dtype)
        ab_t = self._gather(self.schedule.alpha_bars, t).to(x_t.dtype)
        b_t = self._gather(self.schedule.betas, t).to(x_t.dtype)
        eps_coef = (1.0 - a_t) / torch.sqrt(1.0 - ab_t)
        mean = (1.0 / torch.sqrt(a_t)) * (x_t - eps_coef * pred_noise)
        if not add_noise:
            return mean
        noise = torch.randn_like(x_t)
        return mean + torch.sqrt(b_t) * noise

    @torch.no_grad()
    def sample(self, z_start: torch.Tensor, z_end: torch.Tensor, t_middle: int, latent_size: int) -> torch.Tensor:
        if z_start.dim() != 5 or z_end.dim() != 5:
            raise ValueError("z_start and z_end must be 5D [B,1,T_ec,H,W]")
        B = z_start.shape[0]
        x = torch.randn((B, 1, t_middle, latent_size, latent_size), device=self.device, dtype=torch.float32)
        for i in range(self.schedule.timesteps - 1, -1, -1):
            t = torch.full((B,), i, dtype=torch.long, device=self.device)
            pred = self.model(x, t, z_start, z_end)
            x = self.reverse_step(x, pred, t, add_noise=(i > 0))
        return x


# -----------------------------
# Video DDPM training
# -----------------------------
def _load_or_compute_latent_max(latent_video_path: str, videos: np.ndarray) -> float:
    meta_path = str(Path(latent_video_path).with_suffix(".meta.json"))
    if Path(meta_path).exists():
        meta = load_json(meta_path)
        if "latent_max" in meta:
            return float(meta["latent_max"])
    return float(max(np.max(np.abs(videos)), 1e-6))


def train_video_ddpm(args) -> None:
    ensure_file(args.latent_video_path)
    ensure_dir(args.video_ddpm_ckpt_dir)

    videos = np.load(args.latent_video_path)
    if videos.ndim != 4:
        raise ValueError(f"Expected latent video shape (N, T, H, W), got {videos.shape}")
    N, T, H, W = videos.shape
    if H != W:
        raise ValueError(f"Expected square spatial dims, got ({H}, {W})")

    endpoint_context = int(getattr(args, "endpoint_context", DEFAULT_ENDPOINT_CONTEXT))
    t_middle = T - 2 * endpoint_context
    if t_middle <= 0:
        raise ValueError(f"endpoint_context={endpoint_context} too large for T={T}")

    latent_max = _load_or_compute_latent_max(args.latent_video_path, videos)

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(N)
    n_val = max(int(round(N * args.val_split)), 1 if N >= 2 else 0)
    if n_val >= N:
        n_val = max(N - 1, 0)
    n_train = N - n_val
    train_idx = perm[:n_train]
    val_idx = perm[n_train:]

    train_ds = LatentVideoDataset(videos[train_idx], latent_max, endpoint_context=endpoint_context)
    has_val = len(val_idx) > 0
    val_ds = LatentVideoDataset(videos[val_idx], latent_max, endpoint_context=endpoint_context) if has_val else None

    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=pin)
    val_loader = (
        DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=pin)
        if has_val else None
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleEndpointConditionedVideoUNet(
        in_channels=3, out_channels=1, base_ch=int(args.base_channels),
    ).to(device)
    schedule = VideoDiffusionSchedule(
        timesteps=int(args.timesteps),
        beta_schedule=str(args.beta_schedule),
        beta_start=float(args.beta_start),
        beta_end=float(args.beta_end),
        device=device,
    )
    trainer = LatentVideoDDPMTrainer(model, schedule)
    opt = torch.optim.Adam(model.parameters(), lr=float(args.lr))

    latest_path = os.path.join(args.video_ddpm_ckpt_dir, "video_ddpm_latest.pt")
    best_path = os.path.join(args.video_ddpm_ckpt_dir, "video_ddpm_best.pt")
    final_path = os.path.join(args.video_ddpm_ckpt_dir, "video_ddpm_final.pt")
    loss_txt_path = os.path.join(args.video_ddpm_ckpt_dir, "video_ddpm_epoch_losses.txt")

    start_epoch = 1
    best_val = float("inf")
    resumed = os.path.exists(latest_path)

    if not resumed:
        with open(loss_txt_path, "w", encoding="utf-8") as f:
            f.write("epoch\ttrain_loss\tval_loss\n")
    elif not os.path.exists(loss_txt_path):
        with open(loss_txt_path, "w", encoding="utf-8") as f:
            f.write("epoch\ttrain_loss\tval_loss\n")

    if resumed:
        state = torch.load(latest_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        start_epoch = int(state.get("epoch", 0)) + 1
        best_val = float(state.get("best_val", best_val))
        logging.info("Restored video DDPM checkpoint: %s (epoch %d)", latest_path, state.get("epoch", 0))

    no_improve = 0
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        tr_losses = []
        for middle, z_start, z_end in train_loader:
            middle = middle.to(device)
            z_start = z_start.to(device)
            z_end = z_end.to(device)
            loss = trainer.train_step(opt, middle, z_start, z_end)
            tr_losses.append(float(loss.item()))

        va_losses = []
        if has_val:
            model.eval()
            with torch.no_grad():
                for middle, z_start, z_end in val_loader:
                    middle = middle.to(device)
                    z_start = z_start.to(device)
                    z_end = z_end.to(device)
                    va_losses.append(float(trainer.val_step(middle, z_start, z_end).item()))

        tr = float(np.mean(tr_losses)) if tr_losses else float("inf")
        va = float(np.mean(va_losses)) if va_losses else tr
        logging.info("Video DDPM epoch %d/%d | train=%.6f val=%.6f", epoch, args.epochs, tr, va)
        append_epoch_loss_txt(loss_txt_path, epoch, tr, va)

        torch.save(
            {"epoch": epoch, "model": model.state_dict(), "optimizer": opt.state_dict(), "best_val": best_val},
            latest_path,
        )

        if va < best_val:
            best_val = va
            no_improve = 0
            torch.save(model.state_dict(), best_path)
        else:
            no_improve += 1
            if no_improve >= args.patience:
                logging.info("Video DDPM early stopping at epoch %d", epoch)
                break

    torch.save(model.state_dict(), final_path)

    save_json(
        {
            "latent_size": int(H),
            "num_frames": int(T),
            "endpoint_context": int(endpoint_context),
            "t_middle": int(t_middle),
            "latent_max": float(latent_max),
            "timesteps": int(args.timesteps),
            "beta_schedule": str(args.beta_schedule),
            "beta_start": float(args.beta_start),
            "beta_end": float(args.beta_end),
            "base_channels": int(args.base_channels),
            "in_channels": 3,
            "out_channels": 1,
            "latent_video_path": str(args.latent_video_path),
        },
        os.path.join(args.video_ddpm_ckpt_dir, "video_ddpm_meta.json"),
    )
    logging.info("Video DDPM training complete")


# -----------------------------
# Sampling and preview
# -----------------------------
def _save_video_preview_png(decoded_frames: List[np.ndarray], out_path: Path, point_color: str = "#2F6DB3") -> None:
    T = len(decoded_frames)
    if T == 0:
        return
    cols = min(T, 5)
    rows = math.ceil(T / cols)
    fig = plt.figure(figsize=(3.2 * cols, 3.2 * rows))
    for i, pc in enumerate(decoded_frames):
        ax = fig.add_subplot(rows, cols, i + 1, projection="3d")
        pts = pc[:, :3]
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=1, c=point_color)
        ax.set_title(f"Frame {i}")
        ax.axis("off")
    plt.tight_layout()
    ensure_dir(str(out_path.parent))
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _load_video_ddpm_for_sampling(video_ddpm_ckpt_dir: str, device: torch.device) -> Tuple[
    SimpleEndpointConditionedVideoUNet, LatentVideoDDPMTrainer, Dict
]:
    meta_path = os.path.join(video_ddpm_ckpt_dir, "video_ddpm_meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Missing video_ddpm_meta.json in {video_ddpm_ckpt_dir}")
    meta = load_json(meta_path)
    base_channels = int(meta.get("base_channels", DEFAULT_BASE_CHANNELS))

    model = SimpleEndpointConditionedVideoUNet(
        in_channels=int(meta.get("in_channels", 3)),
        out_channels=int(meta.get("out_channels", 1)),
        base_ch=base_channels,
    ).to(device)

    weights_path = os.path.join(video_ddpm_ckpt_dir, "video_ddpm_best.pt")
    if not Path(weights_path).exists():
        weights_path = os.path.join(video_ddpm_ckpt_dir, "video_ddpm_final.pt")
    if not Path(weights_path).exists():
        raise FileNotFoundError(f"No video DDPM weights in {video_ddpm_ckpt_dir}")
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.eval()

    schedule = VideoDiffusionSchedule(
        timesteps=int(meta["timesteps"]),
        beta_schedule=str(meta["beta_schedule"]),
        beta_start=float(meta["beta_start"]),
        beta_end=float(meta["beta_end"]),
        device=device,
    )
    trainer = LatentVideoDDPMTrainer(model, schedule)
    return model, trainer, meta


def _resolve_endpoints(args, latent_size: int, latent_max: float) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """Return (z_start, z_end, start_index, end_index) as float32 [H, W] arrays."""
    rng = np.random.default_rng(int(args.seed) + 100)
    start_idx = int(args.start_index)
    end_idx = int(args.end_index)

    if Path(args.encoded_features).exists():
        encoded = np.load(args.encoded_features).astype(np.float32)
        if encoded.ndim != 3:
            raise ValueError(f"encoded_features must be 3D (N,H,W), got {encoded.shape}")
        if encoded.shape[-1] != latent_size or encoded.shape[-2] != latent_size:
            raise ValueError(
                f"encoded_features spatial {encoded.shape[-2:]} does not match latent_size={latent_size}"
            )

        if start_idx < 0 or end_idx < 0:
            chosen = rng.choice(len(encoded), size=2, replace=(len(encoded) < 2))
            if start_idx < 0:
                start_idx = int(chosen[0])
            if end_idx < 0:
                end_idx = int(chosen[1])

        if start_idx >= len(encoded) or end_idx >= len(encoded) or start_idx < 0 or end_idx < 0:
            raise IndexError(
                f"start_index={start_idx} or end_index={end_idx} out of bounds for encoded length {len(encoded)}"
            )
        return encoded[start_idx], encoded[end_idx], start_idx, end_idx

    # No encoded features. Random gaussian endpoints are sanity-check only and
    # must be opted into explicitly via --allow_random_endpoints.
    if not bool(getattr(args, "allow_random_endpoints", False)):
        raise FileNotFoundError(
            f"encoded_features not found at {args.encoded_features}. "
            "Pass --allow_random_endpoints to sample with random Gaussian endpoints "
            "(sanity check only — not a meaningful generation)."
        )
    logging.warning(
        "encoded_features not found; sampling random gaussian endpoints scaled to latent_max=%.4f "
        "(sanity-check mode, --allow_random_endpoints was set)",
        latent_max,
    )
    z_a = rng.standard_normal((latent_size, latent_size)).astype(np.float32) * latent_max
    z_b = rng.standard_normal((latent_size, latent_size)).astype(np.float32) * latent_max
    return z_a, z_b, -1, -1


def sample_video(args) -> None:
    ensure_dir(args.sample_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, trainer, meta = _load_video_ddpm_for_sampling(args.video_ddpm_ckpt_dir, device)
    latent_size = int(meta["latent_size"])
    endpoint_context = int(meta.get("endpoint_context", DEFAULT_ENDPOINT_CONTEXT))
    latent_max = float(meta["latent_max"])

    num_frames = int(args.num_frames) if int(args.num_frames) > 0 else int(meta["num_frames"])
    t_middle = num_frames - 2 * endpoint_context
    if t_middle <= 0:
        raise ValueError(
            f"num_frames={num_frames} too small for endpoint_context={endpoint_context}"
        )

    z_start_np, z_end_np, start_idx, end_idx = _resolve_endpoints(args, latent_size, latent_max)

    z_start_n = np.clip(z_start_np / latent_max, -1.0, 1.0).astype(np.float32)
    z_end_n = np.clip(z_end_np / latent_max, -1.0, 1.0).astype(np.float32)

    z_start_t = torch.from_numpy(z_start_n).to(device).view(1, 1, 1, latent_size, latent_size)
    z_end_t = torch.from_numpy(z_end_n).to(device).view(1, 1, 1, latent_size, latent_size)

    with torch.no_grad():
        middle = trainer.sample(z_start_t, z_end_t, t_middle=t_middle, latent_size=latent_size)
    middle_np = middle.cpu().numpy()[0, 0]            # [t_middle, H, W] normalized
    middle_np = (middle_np * latent_max).astype(np.float32)

    # Assemble full latent video with exact endpoint frames.
    full_video = np.concatenate(
        [z_start_np[None, ...], middle_np, z_end_np[None, ...]],
        axis=0,
    ).astype(np.float32)
    if full_video.shape != (num_frames, latent_size, latent_size):
        raise RuntimeError(
            f"Assembled video shape {full_video.shape} != expected "
            f"({num_frames},{latent_size},{latent_size})"
        )

    ae = load_ae_for_decode(args.ae_ckpt_dir, prefer_best=True, device=device)

    sample_id = int(args.sample_id) if int(args.sample_id) >= 0 else 0
    sample_dir = Path(args.sample_dir) / f"sample_{sample_id:04d}"
    sample_dir.mkdir(parents=True, exist_ok=True)

    decoded_frames: List[np.ndarray] = []
    with torch.no_grad():
        for f_i in range(full_video.shape[0]):
            latent = full_video[f_i]
            z = torch.from_numpy(latent[None, ...]).to(device)
            decoded = ae.decoder(z).squeeze(0).cpu().numpy().astype(np.float32)
            decoded_frames.append(decoded)
            np.save(sample_dir / f"frame_{f_i:03d}.npy", decoded)

    np.savez(
        sample_dir / "latent_video.npz",
        latent_video=full_video,
        z_start=z_start_np,
        z_end=z_end_np,
        start_index=int(start_idx),
        end_index=int(end_idx),
        num_frames=int(num_frames),
        endpoint_context=int(endpoint_context),
        latent_max=float(latent_max),
    )

    _save_video_preview_png(decoded_frames, sample_dir / "preview.png")
    logging.info("Saved video sample to %s (frames=%d)", sample_dir, len(decoded_frames))


# =============================================================
# Controlled synthetic shape-transformation extensions
#   - deformation utilities
#   - build_controlled_shape_videos
#   - ControlConditionedVideoUNet / dataset / trainer
#   - train_controlled_video_ddpm
#   - sample_controlled_video
#   - audit_controlled_videos
# =============================================================

DEFAULT_CONTROL_DIM = 16
DEFAULT_CONTROLLED_CLIP = 1.25
DEFAULT_CONTROLLED_NUM_SEQUENCES = 50000

DEFORMATION_MODE_TO_ID = {
    "cavity_expansion": 0,
    "cavity_contraction": 1,
    "anisotropic_expansion": 2,
    "twist": 3,
    "hybrid_cavity_twist": 4,
    "bulge_then_relax": 5,
    "two_stage_morph": 6,
}
ALL_DEFORMATION_MODES = list(DEFORMATION_MODE_TO_ID.keys())


def _smoothstep(t: np.ndarray) -> np.ndarray:
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _normalize_rows(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return (v / np.maximum(n, eps)).astype(np.float32)


def _sample_unit_vector(rng: np.random.Generator) -> np.ndarray:
    for _ in range(16):
        v = rng.standard_normal(3).astype(np.float32)
        n = float(np.linalg.norm(v))
        if n > 1e-6:
            return (v / n).astype(np.float32)
    return np.array([0.0, 0.0, 1.0], dtype=np.float32)


def _alpha_monotonic(T: int) -> np.ndarray:
    if T < 2:
        return np.zeros(T, dtype=np.float32)
    return _smoothstep(np.linspace(0.0, 1.0, T, dtype=np.float32)).astype(np.float32)


def _alpha_bulge(T: int) -> np.ndarray:
    if T < 2:
        return np.zeros(T, dtype=np.float32)
    nt = np.linspace(0.0, 1.0, T, dtype=np.float32)
    return np.sin(np.pi * nt).astype(np.float32)


def _radial_shell(xyz: np.ndarray, center: np.ndarray, cavity_radius: float, shell_width: float):
    rel = xyz - center[None, :]
    r = np.linalg.norm(rel, axis=-1).astype(np.float32)
    direction = _normalize_rows(rel)
    sigma2 = max(2.0 * float(shell_width) * float(shell_width), 1e-8)
    mask = np.exp(-((r - float(cavity_radius)) ** 2) / sigma2).astype(np.float32)
    return mask, direction, r


def _apply_cavity(
    xyz: np.ndarray, normals: np.ndarray, alphas: np.ndarray,
    center: np.ndarray, cavity_radius: float, shell_width: float, max_strength: float,
    clip_value: float,
) -> Tuple[np.ndarray, np.ndarray]:
    mask, direction, _ = _radial_shell(xyz, center, cavity_radius, shell_width)
    T = len(alphas)
    P = xyz.shape[0]
    xyz_f = np.zeros((T, P, 3), dtype=np.float32)
    nrm_f = np.zeros((T, P, 3), dtype=np.float32)
    disp = (float(max_strength) * mask)[:, None] * direction
    for ti, a in enumerate(alphas):
        a = float(a)
        xyz_t = xyz + a * disp
        np.clip(xyz_t, -clip_value, clip_value, out=xyz_t)
        w = 0.25 * a * mask
        n_t = (1.0 - w)[:, None] * normals + w[:, None] * direction
        n_t = _normalize_rows(n_t)
        xyz_f[ti] = xyz_t
        nrm_f[ti] = n_t
    return xyz_f, nrm_f


def _apply_anisotropic(
    xyz: np.ndarray, normals: np.ndarray, alphas: np.ndarray,
    center: np.ndarray, cavity_radius: float, shell_width: float, max_strength: float,
    axis: np.ndarray, clip_value: float,
) -> Tuple[np.ndarray, np.ndarray]:
    mask, direction, _ = _radial_shell(xyz, center, cavity_radius, shell_width)
    aniso = np.abs(direction @ axis).astype(np.float32)
    T = len(alphas)
    P = xyz.shape[0]
    xyz_f = np.zeros((T, P, 3), dtype=np.float32)
    nrm_f = np.zeros((T, P, 3), dtype=np.float32)
    disp = (float(max_strength) * mask * aniso)[:, None] * direction
    for ti, a in enumerate(alphas):
        a = float(a)
        xyz_t = xyz + a * disp
        np.clip(xyz_t, -clip_value, clip_value, out=xyz_t)
        w = 0.25 * a * mask * aniso
        n_t = (1.0 - w)[:, None] * normals + w[:, None] * direction
        n_t = _normalize_rows(n_t)
        xyz_f[ti] = xyz_t
        nrm_f[ti] = n_t
    return xyz_f, nrm_f


def _rotate_vectors_axis(v: np.ndarray, axis: np.ndarray, angles: np.ndarray) -> np.ndarray:
    cosA = np.cos(angles).astype(np.float32)
    sinA = np.sin(angles).astype(np.float32)
    k = axis.astype(np.float32)
    k_b = np.broadcast_to(k, v.shape)
    kv_cross = np.cross(k_b, v).astype(np.float32)
    k_dot_v = (v * k[None, :]).sum(axis=-1, keepdims=True).astype(np.float32)
    v_rot = (
        v * cosA[:, None]
        + kv_cross * sinA[:, None]
        + (k[None, :] * k_dot_v) * (1.0 - cosA[:, None])
    )
    return v_rot.astype(np.float32)


def _apply_twist(
    xyz: np.ndarray, normals: np.ndarray, alphas: np.ndarray,
    center: np.ndarray, axis: np.ndarray, max_angle: float, clip_value: float,
) -> Tuple[np.ndarray, np.ndarray]:
    axis = axis.astype(np.float32)
    n_axis = float(np.linalg.norm(axis))
    if n_axis < 1e-6:
        axis = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    else:
        axis = (axis / n_axis).astype(np.float32)
    rel = (xyz - center[None, :]).astype(np.float32)
    axial = (rel @ axis).astype(np.float32)
    a_min = float(axial.min())
    a_max = float(axial.max())
    rng_axial = max(a_max - a_min, 1e-6)
    u = ((axial - a_min) / rng_axial).astype(np.float32)
    u_signed = ((u - 0.5) * 2.0).astype(np.float32)
    T = len(alphas)
    P = xyz.shape[0]
    xyz_f = np.zeros((T, P, 3), dtype=np.float32)
    nrm_f = np.zeros((T, P, 3), dtype=np.float32)
    for ti, a in enumerate(alphas):
        angles = (float(a) * float(max_angle) * u_signed).astype(np.float32)
        v_rot = _rotate_vectors_axis(rel, axis, angles)
        xyz_t = v_rot + center[None, :]
        np.clip(xyz_t, -clip_value, clip_value, out=xyz_t)
        n_rot = _rotate_vectors_axis(normals.astype(np.float32), axis, angles)
        n_rot = _normalize_rows(n_rot)
        xyz_f[ti] = xyz_t
        nrm_f[ti] = n_rot
    return xyz_f, nrm_f


def _sample_params_for_mode(mode: str, xyz: np.ndarray, rng: np.random.Generator) -> Dict:
    centroid = xyz.mean(axis=0).astype(np.float32)
    center = (centroid + rng.uniform(-0.05, 0.05, size=3).astype(np.float32)).astype(np.float32)
    axis = _sample_unit_vector(rng)
    params = {
        "center": center,
        "axis": axis,
        "cavity_radius": 0.0,
        "shell_width": 0.0,
        "max_strength": 0.0,
        "twist_strength": 0.0,
        "anisotropy_strength": 0.0,
        "phase_or_relaxation_strength": 0.0,
        "composite_strength": 0.0,
    }
    if mode == "cavity_expansion":
        params["cavity_radius"] = float(rng.uniform(0.15, 0.55))
        params["shell_width"] = float(rng.uniform(0.08, 0.25))
        params["max_strength"] = float(rng.uniform(0.03, 0.18))
    elif mode == "cavity_contraction":
        params["cavity_radius"] = float(rng.uniform(0.15, 0.55))
        params["shell_width"] = float(rng.uniform(0.08, 0.25))
        params["max_strength"] = float(rng.uniform(-0.16, -0.03))
    elif mode == "anisotropic_expansion":
        params["cavity_radius"] = float(rng.uniform(0.15, 0.55))
        params["shell_width"] = float(rng.uniform(0.08, 0.25))
        params["max_strength"] = float(rng.uniform(0.04, 0.20))
        params["anisotropy_strength"] = params["max_strength"]
    elif mode == "twist":
        params["twist_strength"] = float(rng.uniform(0.2, 1.2))
    elif mode == "hybrid_cavity_twist":
        params["cavity_radius"] = float(rng.uniform(0.15, 0.55))
        params["shell_width"] = float(rng.uniform(0.08, 0.25))
        params["max_strength"] = float(rng.uniform(0.02, 0.10))
        params["twist_strength"] = float(rng.uniform(0.15, 0.8))
    elif mode == "bulge_then_relax":
        params["cavity_radius"] = float(rng.uniform(0.15, 0.55))
        params["shell_width"] = float(rng.uniform(0.08, 0.25))
        params["max_strength"] = float(rng.uniform(0.04, 0.16))
        params["phase_or_relaxation_strength"] = 1.0
    elif mode == "two_stage_morph":
        params["cavity_radius"] = float(rng.uniform(0.15, 0.55))
        params["shell_width"] = float(rng.uniform(0.08, 0.25))
        params["max_strength"] = float(rng.uniform(0.04, 0.14))
        params["twist_strength"] = float(rng.uniform(0.3, 0.9))
        params["anisotropy_strength"] = float(rng.uniform(0.05, 0.16))
        params["composite_strength"] = float(
            (params["max_strength"] + params["twist_strength"] + params["anisotropy_strength"]) / 3.0
        )
    else:
        raise ValueError(f"Unknown deformation mode: {mode}")
    return params


def _build_control_vector(mode: str, params: Dict, num_frames: int, latent_size: int) -> np.ndarray:
    mode_id = DEFORMATION_MODE_TO_ID[mode]
    n_modes = max(len(DEFORMATION_MODE_TO_ID) - 1, 1)
    c = np.zeros(DEFAULT_CONTROL_DIM, dtype=np.float32)
    c[0] = float(mode_id) / float(n_modes)
    c[1] = float(params.get("cavity_radius", 0.0))
    c[2] = float(params.get("shell_width", 0.0))
    c[3] = float(params.get("max_strength", 0.0))
    c[4] = float(params.get("twist_strength", 0.0))
    c[5] = float(params.get("anisotropy_strength", 0.0))
    center = np.asarray(params.get("center", np.zeros(3, dtype=np.float32)), dtype=np.float32).reshape(-1)
    c[6] = float(center[0]) if center.size > 0 else 0.0
    c[7] = float(center[1]) if center.size > 1 else 0.0
    c[8] = float(center[2]) if center.size > 2 else 0.0
    axis = np.asarray(params.get("axis", np.zeros(3, dtype=np.float32)), dtype=np.float32).reshape(-1)
    c[9] = float(axis[0]) if axis.size > 0 else 0.0
    c[10] = float(axis[1]) if axis.size > 1 else 0.0
    c[11] = float(axis[2]) if axis.size > 2 else 0.0
    c[12] = float(num_frames) / 100.0
    c[13] = float(latent_size) / 128.0
    c[14] = float(params.get("phase_or_relaxation_strength", 0.0))
    c[15] = float(params.get("composite_strength", 0.0))
    return c


def _generate_transformation_frames(
    pc: np.ndarray, mode: str, params: Dict, num_frames: int, clip_value: float,
) -> np.ndarray:
    xyz = pc[:, :3].astype(np.float32).copy()
    normals = _normalize_rows(pc[:, 3:6].astype(np.float32).copy())

    if mode == "bulge_then_relax":
        alphas = _alpha_bulge(num_frames)
    else:
        alphas = _alpha_monotonic(num_frames)

    if mode in ("cavity_expansion", "cavity_contraction", "bulge_then_relax"):
        xyz_f, nrm_f = _apply_cavity(
            xyz, normals, alphas,
            params["center"], params["cavity_radius"], params["shell_width"],
            params["max_strength"], clip_value,
        )
    elif mode == "anisotropic_expansion":
        xyz_f, nrm_f = _apply_anisotropic(
            xyz, normals, alphas,
            params["center"], params["cavity_radius"], params["shell_width"],
            params["max_strength"], params["axis"], clip_value,
        )
    elif mode == "twist":
        xyz_f, nrm_f = _apply_twist(
            xyz, normals, alphas,
            params["center"], params["axis"], params["twist_strength"], clip_value,
        )
    elif mode == "hybrid_cavity_twist":
        xyz_c, nrm_c = _apply_cavity(
            xyz, normals, alphas,
            params["center"], params["cavity_radius"], params["shell_width"],
            params["max_strength"], clip_value,
        )
        xyz_f = np.zeros_like(xyz_c)
        nrm_f = np.zeros_like(nrm_c)
        for ti, a in enumerate(alphas):
            xyzi = xyz_c[ti]
            ni = nrm_c[ti]
            tw_xyz, tw_n = _apply_twist(
                xyzi, ni, np.array([float(a)], dtype=np.float32),
                params["center"], params["axis"], params["twist_strength"], clip_value,
            )
            xyz_f[ti] = tw_xyz[0]
            nrm_f[ti] = tw_n[0]
    elif mode == "two_stage_morph":
        T = num_frames
        T1 = max(1, T // 2)
        # Stage 1: cavity ramp from 0 to 1 over frames 0..T1, then hold
        alphas1 = np.zeros(T, dtype=np.float32)
        if T1 + 1 >= 2:
            alphas1[: T1 + 1] = _alpha_monotonic(T1 + 1)
        alphas1[T1 + 1:] = 1.0
        xyz_c, nrm_c = _apply_cavity(
            xyz, normals, alphas1,
            params["center"], params["cavity_radius"], params["shell_width"],
            params["max_strength"], clip_value,
        )
        # Stage 2: twist ramp from 0 across frames T1..T-1
        T2 = T - T1
        twist_alphas = np.zeros(T, dtype=np.float32)
        if T2 >= 2:
            twist_alphas[T1:] = _alpha_monotonic(T2)
        xyz_f = np.zeros_like(xyz_c)
        nrm_f = np.zeros_like(nrm_c)
        for ti, a in enumerate(twist_alphas):
            xyzi = xyz_c[ti]
            ni = nrm_c[ti]
            if float(a) <= 1e-7:
                xyz_f[ti] = xyzi
                nrm_f[ti] = ni
            else:
                tw_xyz, tw_n = _apply_twist(
                    xyzi, ni, np.array([float(a)], dtype=np.float32),
                    params["center"], params["axis"], params["twist_strength"], clip_value,
                )
                xyz_f[ti] = tw_xyz[0]
                nrm_f[ti] = tw_n[0]
    else:
        raise ValueError(f"Unknown deformation mode: {mode}")

    # Enforce frame 0 equals original
    xyz_f[0] = xyz
    nrm_f[0] = normals
    return np.concatenate([xyz_f, nrm_f], axis=-1).astype(np.float32)


def build_controlled_shape_videos(args) -> None:
    ensure_file(args.pointcloud_path)
    ensure_dir(str(Path(args.controlled_video_path).parent))
    ensure_dir(str(Path(args.control_path).parent))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ae = load_ae_for_decode(
        args.ae_ckpt_dir,
        prefer_best=bool(getattr(args, "use_best_ae", False)),
        device=device,
    )
    ae.eval()

    pcs = np.load(args.pointcloud_path, mmap_mode="r")
    if pcs.ndim != 3 or pcs.shape[-1] != 6:
        raise ValueError(f"Point cloud must have shape (N,P,6), got {pcs.shape}")
    n_total, P, _ = pcs.shape
    if ae.point_size != P:
        raise ValueError(f"AE point_size={ae.point_size} != dataset P={P}")

    modes = [m.strip() for m in str(args.deformation_modes).split(",") if m.strip()]
    probs_str = [p.strip() for p in str(args.deformation_mode_probs).split(",") if p.strip()]
    probs = np.array([float(p) for p in probs_str], dtype=np.float64)
    if len(modes) != len(probs):
        raise ValueError(
            f"deformation_modes ({len(modes)}) and deformation_mode_probs ({len(probs)}) must match"
        )
    for m in modes:
        if m not in DEFORMATION_MODE_TO_ID:
            raise ValueError(f"Unknown deformation mode: {m}")
    p_sum = float(probs.sum())
    if p_sum <= 0.0:
        raise ValueError("deformation_mode_probs must sum to a positive value")
    probs = (probs / p_sum).astype(np.float64)

    num_sequences = int(args.num_sequences)
    num_frames = int(args.num_frames)
    latent_size = int(args.latent_size)
    encode_batch_size = max(int(args.encode_batch_size), 1)
    control_dim = int(args.control_dim)
    clip_value = float(args.clip_value)

    if control_dim != DEFAULT_CONTROL_DIM:
        raise ValueError(f"control_dim must be {DEFAULT_CONTROL_DIM}, got {control_dim}")
    if ae.latent_size != latent_size:
        raise ValueError(f"AE latent_size={ae.latent_size} != requested latent_size={latent_size}")
    if num_frames < 2:
        raise ValueError(f"num_frames must be >= 2, got {num_frames}")

    rng = np.random.default_rng(int(args.seed))

    videos = np.zeros((num_sequences, num_frames, latent_size, latent_size), dtype=np.float32)
    controls = np.zeros((num_sequences, control_dim), dtype=np.float32)
    labels = np.zeros((num_sequences,), dtype=np.int64)
    mode_counts: Dict[str, int] = {m: 0 for m in modes}

    forced_modes: List[str] = list(modes) if num_sequences >= len(modes) else []

    base_indices = rng.integers(0, n_total, size=num_sequences)

    seq_modes: List[str] = []
    for i in range(num_sequences):
        if i < len(forced_modes):
            seq_modes.append(forced_modes[i])
        else:
            seq_modes.append(modes[int(rng.choice(len(modes), p=probs))])

    buf_frames: List[Tuple[int, int, np.ndarray]] = []

    def flush() -> None:
        if not buf_frames:
            return
        pc_stack = np.stack([item[2] for item in buf_frames], axis=0).astype(np.float32)
        with torch.no_grad():
            pc_t = torch.from_numpy(pc_stack).to(device).permute(0, 2, 1).contiguous()
            z = ae.encoder(pc_t)
            z_np = z.detach().cpu().numpy().astype(np.float32)
        for j, (si, fi, _) in enumerate(buf_frames):
            videos[si, fi] = z_np[j]
        buf_frames.clear()

    logging.info(
        "Building %d controlled sequences | T=%d, H=W=%d, modes=%s",
        num_sequences, num_frames, latent_size, modes,
    )
    log_every = max(num_sequences // 20, 1)
    for i in range(num_sequences):
        mode = seq_modes[i]
        labels[i] = DEFORMATION_MODE_TO_ID[mode]
        pc = np.array(pcs[int(base_indices[i])], dtype=np.float32)
        params = _sample_params_for_mode(mode, pc[:, :3], rng)
        controls[i] = _build_control_vector(mode, params, num_frames, latent_size)
        mode_counts[mode] += 1
        frames = _generate_transformation_frames(pc, mode, params, num_frames, clip_value)
        if not np.isfinite(frames).all():
            raise RuntimeError(f"Non-finite frames generated for sequence {i} mode {mode}")
        for fi in range(num_frames):
            buf_frames.append((i, fi, frames[fi]))
            if len(buf_frames) >= encode_batch_size:
                flush()
        if (i + 1) % log_every == 0:
            logging.info("  built %d/%d sequences", i + 1, num_sequences)
    flush()

    if not np.isfinite(videos).all():
        raise RuntimeError("Encoded videos contain non-finite values")
    if not np.isfinite(controls).all():
        raise RuntimeError("Controls contain non-finite values")

    np.save(args.controlled_video_path, videos)
    np.save(args.control_path, controls)
    labels_path = str(Path(args.controlled_video_path).with_suffix(".labels.npy"))
    np.save(labels_path, labels)

    latent_max = float(max(np.max(np.abs(videos)), 1e-6))

    meta = {
        "num_sequences": int(num_sequences),
        "num_frames": int(num_frames),
        "latent_size": int(latent_size),
        "control_dim": int(control_dim),
        "deformation_modes": list(modes),
        "deformation_mode_probs": [float(p) for p in probs.tolist()],
        "mode_to_id": DEFORMATION_MODE_TO_ID,
        "parameter_ranges": {
            "cavity_radius": [0.15, 0.55],
            "shell_width": [0.08, 0.25],
            "cavity_expansion_strength": [0.03, 0.18],
            "cavity_contraction_strength": [-0.16, -0.03],
            "anisotropic_strength": [0.04, 0.20],
            "twist_max_angle": [0.2, 1.2],
            "hybrid_cavity_twist_strength": [0.02, 0.10],
            "hybrid_cavity_twist_twist": [0.15, 0.8],
            "bulge_then_relax_strength": [0.04, 0.16],
            "two_stage_morph_strength": [0.04, 0.14],
            "two_stage_morph_twist": [0.3, 0.9],
            "two_stage_morph_anisotropy": [0.05, 0.16],
        },
        "pointcloud_path": str(args.pointcloud_path),
        "ae_ckpt_dir": str(args.ae_ckpt_dir),
        "use_best_ae": bool(getattr(args, "use_best_ae", False)),
        "controlled_video_path": str(args.controlled_video_path),
        "control_path": str(args.control_path),
        "labels_path": labels_path,
        "seed": int(args.seed),
        "latent_max": latent_max,
        "latent_stats": {
            "min": float(videos.min()),
            "max": float(videos.max()),
            "mean": float(videos.mean()),
            "std": float(videos.std()),
        },
        "control_stats": {
            "min": float(controls.min()),
            "max": float(controls.max()),
            "mean": float(controls.mean()),
            "std": float(controls.std()),
        },
        "finite_videos": bool(np.isfinite(videos).all()),
        "finite_controls": bool(np.isfinite(controls).all()),
        "finite_labels": bool(np.isfinite(labels.astype(np.float64)).all()),
        "video_shape": list(videos.shape),
        "control_shape": list(controls.shape),
        "labels_shape": list(labels.shape),
        "mode_counts": mode_counts,
        "clip_value": float(clip_value),
    }
    meta_path = str(Path(args.controlled_video_path).with_suffix(".meta.json"))
    save_json(meta, meta_path)

    logging.info("Saved controlled videos %s shape=%s", args.controlled_video_path, videos.shape)
    logging.info("Saved controls %s shape=%s", args.control_path, controls.shape)
    logging.info("Saved labels %s shape=%s", labels_path, labels.shape)
    logging.info("Mode counts: %s", mode_counts)


# -----------------------------
# Control-conditioned video DDPM
# -----------------------------
class ControlConditionedVideoUNet(SimpleEndpointConditionedVideoUNet):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        base_ch: int = 64,
        dim_mults: Sequence[int] = (1, 2, 4),
        groups: int = 8,
        control_dim: int = DEFAULT_CONTROL_DIM,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            base_ch=base_ch,
            dim_mults=dim_mults,
            groups=groups,
        )
        self.control_dim = int(control_dim)
        self.control_mlp = tnn.Sequential(
            tnn.Linear(self.control_dim, self.time_dim),
            tnn.SiLU(),
            tnn.Linear(self.time_dim, self.time_dim),
        )

    def forward(self, x_noisy, t, z_start, z_end, control):
        if x_noisy.dim() != 5:
            raise ValueError(f"x_noisy must be 5D, got {tuple(x_noisy.shape)}")
        B, C_in, T, H, W = x_noisy.shape
        if C_in != 1:
            raise ValueError(f"x_noisy must have C=1, got C={C_in}")
        if control.dim() != 2 or control.shape[0] != B or control.shape[1] != self.control_dim:
            raise ValueError(
                f"control must be [B, control_dim={self.control_dim}], got {tuple(control.shape)}"
            )
        z_start_1 = self._endpoint_to_single_frame(z_start)
        z_end_1 = self._endpoint_to_single_frame(z_end)
        start_rep = z_start_1.expand(-1, -1, T, -1, -1)
        end_rep = z_end_1.expand(-1, -1, T, -1, -1)
        x = torch.cat([x_noisy, start_rep, end_rep], dim=1)

        t_emb = self.time_mlp(t)
        c_emb = self.control_mlp(control)
        combined = t_emb + c_emb

        x = self.init_conv(x)
        skips: List[torch.Tensor] = []
        for resblock1, resblock2, downsample in self.downs:
            x = resblock1(x, combined)
            x = resblock2(x, combined)
            skips.append(x)
            x = downsample(x)

        x = self.mid1(x, combined)
        x = self.mid2(x, combined)

        for resblock1, resblock2, upsample in self.ups:
            skip = skips.pop()
            x = torch.cat([x, skip], dim=1)
            x = resblock1(x, combined)
            x = resblock2(x, combined)
            x = upsample(x)

        x = self.final_act(self.final_norm(x))
        return self.final_conv(x)


class ControlledLatentVideoDataset(Dataset):
    def __init__(self, videos: np.ndarray, controls: np.ndarray, latent_max: float, endpoint_context: int = 1):
        if videos.ndim != 4:
            raise ValueError(f"videos must be 4D (N, T, H, W), got {videos.shape}")
        if controls.ndim != 2 or len(controls) != len(videos):
            raise ValueError(
                f"controls must be 2D (N, C), got {controls.shape} (N expected {len(videos)})"
            )
        T = videos.shape[1]
        ec = int(endpoint_context)
        if ec < 1 or 2 * ec >= T:
            raise ValueError(f"endpoint_context={ec} invalid for T={T}")
        self.videos = videos.astype(np.float32)
        self.controls = controls.astype(np.float32)
        self.latent_max = float(latent_max)
        self.endpoint_context = ec
        self.T = T

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        v = self.videos[idx]
        c = self.controls[idx]
        T = self.T
        ec = self.endpoint_context
        v_norm = np.clip(v / self.latent_max, -1.0, 1.0).astype(np.float32)
        start = v_norm[0:ec]
        end = v_norm[T - ec:T]
        middle = v_norm[ec:T - ec]
        start_t = torch.from_numpy(start).unsqueeze(0)
        end_t = torch.from_numpy(end).unsqueeze(0)
        middle_t = torch.from_numpy(middle).unsqueeze(0)
        c_t = torch.from_numpy(c)
        return middle_t, start_t, end_t, c_t


class ControlledLatentVideoDDPMTrainer:
    def __init__(self, model: ControlConditionedVideoUNet, schedule: VideoDiffusionSchedule):
        self.model = model
        self.schedule = schedule
        self.device = next(model.parameters()).device

    def _gather(self, vec: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        out = vec.to(self.device)[t]
        return out[:, None, None, None, None]

    def forward_noise(self, x0: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        noise = torch.randn_like(x0)
        sa = self._gather(self.schedule.sqrt_alpha_bars, t).to(x0.dtype)
        osa = self._gather(self.schedule.sqrt_one_minus_alpha_bars, t).to(x0.dtype)
        return sa * x0 + osa * noise, noise

    def train_step(self, opt, middle_x0, z_start, z_end, control):
        B = middle_x0.shape[0]
        t = torch.randint(0, self.schedule.timesteps, (B,), device=middle_x0.device, dtype=torch.long)
        xt, noise = self.forward_noise(middle_x0, t)
        opt.zero_grad(set_to_none=True)
        pred = self.model(xt, t, z_start, z_end, control)
        loss = F.mse_loss(pred, noise)
        loss.backward()
        opt.step()
        return loss

    def val_step(self, middle_x0, z_start, z_end, control):
        B = middle_x0.shape[0]
        t = torch.randint(0, self.schedule.timesteps, (B,), device=middle_x0.device, dtype=torch.long)
        xt, noise = self.forward_noise(middle_x0, t)
        pred = self.model(xt, t, z_start, z_end, control)
        return F.mse_loss(pred, noise)

    def reverse_step(self, x_t, pred_noise, t, add_noise: bool = True):
        a_t = self._gather(self.schedule.alphas, t).to(x_t.dtype)
        ab_t = self._gather(self.schedule.alpha_bars, t).to(x_t.dtype)
        b_t = self._gather(self.schedule.betas, t).to(x_t.dtype)
        eps_coef = (1.0 - a_t) / torch.sqrt(1.0 - ab_t)
        mean = (1.0 / torch.sqrt(a_t)) * (x_t - eps_coef * pred_noise)
        if not add_noise:
            return mean
        noise = torch.randn_like(x_t)
        return mean + torch.sqrt(b_t) * noise

    @torch.no_grad()
    def sample(self, z_start, z_end, control, t_middle: int, latent_size: int) -> torch.Tensor:
        B = z_start.shape[0]
        x = torch.randn((B, 1, t_middle, latent_size, latent_size), device=self.device, dtype=torch.float32)
        for i in range(self.schedule.timesteps - 1, -1, -1):
            t = torch.full((B,), i, dtype=torch.long, device=self.device)
            pred = self.model(x, t, z_start, z_end, control)
            x = self.reverse_step(x, pred, t, add_noise=(i > 0))
        return x


def train_controlled_video_ddpm(args) -> None:
    ensure_file(args.latent_video_path)
    ensure_file(args.control_path)
    ensure_dir(args.video_ddpm_ckpt_dir)

    videos = np.load(args.latent_video_path)
    controls = np.load(args.control_path)
    if videos.ndim != 4:
        raise ValueError(f"Expected latent video shape (N, T, H, W), got {videos.shape}")
    if controls.ndim != 2:
        raise ValueError(f"Expected controls shape (N, C), got {controls.shape}")
    if len(controls) != len(videos):
        raise ValueError(f"Mismatched: videos {len(videos)}, controls {len(controls)}")

    N, T, H, W = videos.shape
    control_dim = int(controls.shape[1])
    if int(args.control_dim) != control_dim:
        raise ValueError(f"--control_dim={args.control_dim} but loaded controls dim={control_dim}")

    endpoint_context = DEFAULT_ENDPOINT_CONTEXT
    t_middle = T - 2 * endpoint_context
    if t_middle <= 0:
        raise ValueError(f"endpoint_context={endpoint_context} too large for T={T}")

    latent_max = _load_or_compute_latent_max(args.latent_video_path, videos)

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(N)
    n_val = max(int(round(N * args.val_split)), 1 if N >= 2 else 0)
    if n_val >= N:
        n_val = max(N - 1, 0)
    n_train = N - n_val
    train_idx = perm[:n_train]
    val_idx = perm[n_train:]

    train_ds = ControlledLatentVideoDataset(
        videos[train_idx], controls[train_idx], latent_max, endpoint_context=endpoint_context
    )
    has_val = len(val_idx) > 0
    val_ds = (
        ControlledLatentVideoDataset(videos[val_idx], controls[val_idx], latent_max, endpoint_context=endpoint_context)
        if has_val else None
    )

    pin = torch.cuda.is_available()
    nw = int(getattr(args, "num_workers", 0))
    train_loader = DataLoader(
        train_ds, batch_size=int(args.batch_size), shuffle=True,
        num_workers=nw, pin_memory=pin,
    )
    val_loader = (
        DataLoader(val_ds, batch_size=int(args.batch_size), shuffle=False, num_workers=nw, pin_memory=pin)
        if has_val else None
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ControlConditionedVideoUNet(
        in_channels=3, out_channels=1, base_ch=int(args.base_channels), control_dim=control_dim,
    ).to(device)
    schedule = VideoDiffusionSchedule(
        timesteps=int(args.timesteps),
        beta_schedule=str(args.beta_schedule),
        beta_start=float(getattr(args, "beta_start", DEFAULT_BETA_START)),
        beta_end=float(getattr(args, "beta_end", DEFAULT_BETA_END)),
        device=device,
    )
    trainer = ControlledLatentVideoDDPMTrainer(model, schedule)
    opt = torch.optim.Adam(model.parameters(), lr=float(args.lr))

    latest_path = os.path.join(args.video_ddpm_ckpt_dir, "controlled_video_ddpm_latest.pt")
    best_path = os.path.join(args.video_ddpm_ckpt_dir, "controlled_video_ddpm_best.pt")
    final_path = os.path.join(args.video_ddpm_ckpt_dir, "controlled_video_ddpm_final.pt")
    loss_txt_path = os.path.join(args.video_ddpm_ckpt_dir, "controlled_video_ddpm_epoch_losses.txt")
    meta_path = os.path.join(args.video_ddpm_ckpt_dir, "controlled_video_ddpm_meta.json")

    start_epoch = 1
    best_val = float("inf")
    best_epoch = 0
    do_resume = bool(getattr(args, "resume", True))
    resumed = do_resume and os.path.exists(latest_path)
    min_delta = float(getattr(args, "min_delta", 0.0))

    if not resumed:
        with open(loss_txt_path, "w", encoding="utf-8") as f:
            f.write("epoch\ttrain_loss\tval_loss\n")
    elif not os.path.exists(loss_txt_path):
        with open(loss_txt_path, "w", encoding="utf-8") as f:
            f.write("epoch\ttrain_loss\tval_loss\n")

    if resumed:
        state = torch.load(latest_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        start_epoch = int(state.get("epoch", 0)) + 1
        best_val = float(state.get("best_val", best_val))
        best_epoch = int(state.get("best_epoch", 0))
        logging.info("Restored controlled DDPM at epoch %d (best_val=%.6f @ %d)", state.get("epoch", 0), best_val, best_epoch)

    no_improve = 0
    final_epoch = start_epoch - 1
    for epoch in range(start_epoch, int(args.epochs) + 1):
        model.train()
        tr_losses: List[float] = []
        for middle, z_start, z_end, ctrl in train_loader:
            middle = middle.to(device); z_start = z_start.to(device)
            z_end = z_end.to(device); ctrl = ctrl.to(device)
            loss = trainer.train_step(opt, middle, z_start, z_end, ctrl)
            tr_losses.append(float(loss.item()))

        va_losses: List[float] = []
        if has_val:
            model.eval()
            with torch.no_grad():
                for middle, z_start, z_end, ctrl in val_loader:
                    middle = middle.to(device); z_start = z_start.to(device)
                    z_end = z_end.to(device); ctrl = ctrl.to(device)
                    va_losses.append(float(trainer.val_step(middle, z_start, z_end, ctrl).item()))

        tr = float(np.mean(tr_losses)) if tr_losses else float("inf")
        va = float(np.mean(va_losses)) if va_losses else tr
        logging.info("Ctrl DDPM epoch %d/%d | train=%.6f val=%.6f", epoch, args.epochs, tr, va)
        append_epoch_loss_txt(loss_txt_path, epoch, tr, va)
        final_epoch = epoch

        torch.save(
            {
                "epoch": epoch, "model": model.state_dict(), "optimizer": opt.state_dict(),
                "best_val": best_val, "best_epoch": best_epoch,
            },
            latest_path,
        )

        if va < best_val - min_delta:
            best_val = va
            best_epoch = epoch
            no_improve = 0
            torch.save(model.state_dict(), best_path)
        else:
            no_improve += 1
            if no_improve >= int(args.patience):
                logging.info("Controlled DDPM early stopping at epoch %d", epoch)
                break

    torch.save(model.state_dict(), final_path)

    save_json(
        {
            "latent_size": int(H),
            "num_frames": int(T),
            "endpoint_context": int(endpoint_context),
            "t_middle": int(t_middle),
            "latent_max": float(latent_max),
            "timesteps": int(args.timesteps),
            "beta_schedule": str(args.beta_schedule),
            "beta_start": float(getattr(args, "beta_start", DEFAULT_BETA_START)),
            "beta_end": float(getattr(args, "beta_end", DEFAULT_BETA_END)),
            "base_channels": int(args.base_channels),
            "control_dim": int(control_dim),
            "in_channels": 3,
            "out_channels": 1,
            "latent_video_path": str(args.latent_video_path),
            "control_path": str(args.control_path),
            "batch_size": int(args.batch_size),
            "epochs": int(args.epochs),
            "patience": int(args.patience),
            "val_split": float(args.val_split),
            "best_val": float(best_val),
            "best_epoch": int(best_epoch),
            "final_epoch": int(final_epoch),
            "seed": int(args.seed),
            "device": str(device),
            "train_count": int(n_train),
            "val_count": int(n_val),
        },
        meta_path,
    )
    logging.info("Controlled DDPM training complete | best_val=%.6f @ epoch %d", best_val, best_epoch)


def _load_controlled_video_ddpm_for_sampling(video_ddpm_ckpt_dir: str, device: torch.device):
    meta_path = os.path.join(video_ddpm_ckpt_dir, "controlled_video_ddpm_meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Missing controlled_video_ddpm_meta.json in {video_ddpm_ckpt_dir}")
    meta = load_json(meta_path)
    base_channels = int(meta.get("base_channels", DEFAULT_BASE_CHANNELS))
    control_dim = int(meta.get("control_dim", DEFAULT_CONTROL_DIM))

    model = ControlConditionedVideoUNet(
        in_channels=int(meta.get("in_channels", 3)),
        out_channels=int(meta.get("out_channels", 1)),
        base_ch=base_channels,
        control_dim=control_dim,
    ).to(device)

    weights_path = os.path.join(video_ddpm_ckpt_dir, "controlled_video_ddpm_best.pt")
    if not Path(weights_path).exists():
        weights_path = os.path.join(video_ddpm_ckpt_dir, "controlled_video_ddpm_final.pt")
    if not Path(weights_path).exists():
        raise FileNotFoundError(f"No controlled DDPM weights in {video_ddpm_ckpt_dir}")
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.eval()

    schedule = VideoDiffusionSchedule(
        timesteps=int(meta["timesteps"]),
        beta_schedule=str(meta["beta_schedule"]),
        beta_start=float(meta.get("beta_start", DEFAULT_BETA_START)),
        beta_end=float(meta.get("beta_end", DEFAULT_BETA_END)),
        device=device,
    )
    trainer = ControlledLatentVideoDDPMTrainer(model, schedule)
    return model, trainer, meta


def _save_video_animation_gif(decoded_frames: List[np.ndarray], out_path: Path, point_color: str = "#2F6DB3") -> None:
    if not decoded_frames:
        return
    try:
        import matplotlib.animation as manimation
    except Exception as e:
        logging.warning("matplotlib.animation unavailable: %s", e)
        return
    all_pts = np.concatenate([f[:, :3] for f in decoded_frames], axis=0)
    mn = float(all_pts.min())
    mx = float(all_pts.max())
    pad = 0.05 * (mx - mn) if mx > mn else 0.05
    fig = plt.figure(figsize=(4.5, 4.5))
    ax = fig.add_subplot(111, projection="3d")

    def update(i):
        ax.cla()
        ax.set_xlim(mn - pad, mx + pad)
        ax.set_ylim(mn - pad, mx + pad)
        ax.set_zlim(mn - pad, mx + pad)
        ax.view_init(elev=20, azim=30)
        ax.axis("off")
        pts = decoded_frames[i][:, :3]
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=1, c=point_color)
        ax.set_title(f"frame {i}")
        return (ax,)

    anim = manimation.FuncAnimation(fig, update, frames=len(decoded_frames), interval=200, blit=False)
    ensure_dir(str(out_path.parent))
    try:
        anim.save(str(out_path), writer="pillow", fps=5)
    except Exception as e:
        logging.warning("Could not save animation GIF: %s", e)
    plt.close(fig)


def sample_controlled_video(args) -> None:
    ensure_dir(args.sample_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, trainer, meta = _load_controlled_video_ddpm_for_sampling(args.video_ddpm_ckpt_dir, device)
    latent_size = int(meta["latent_size"])
    endpoint_context = int(meta.get("endpoint_context", DEFAULT_ENDPOINT_CONTEXT))
    latent_max = float(meta["latent_max"])

    num_frames = int(args.num_frames) if int(args.num_frames) > 0 else int(meta["num_frames"])
    t_middle = num_frames - 2 * endpoint_context
    if t_middle <= 0:
        raise ValueError(f"num_frames={num_frames} too small for endpoint_context={endpoint_context}")

    mode = str(args.mode)
    if mode not in DEFORMATION_MODE_TO_ID:
        raise ValueError(f"Unknown mode: {mode}")

    base_pc = None
    if Path(args.pointcloud_path).exists():
        pcs = np.load(args.pointcloud_path, mmap_mode="r")
        sidx = int(args.start_index)
        if 0 <= sidx < len(pcs):
            base_pc = np.array(pcs[sidx], dtype=np.float32)
    default_centroid = (
        base_pc[:, :3].mean(axis=0).astype(np.float32)
        if base_pc is not None else np.zeros(3, dtype=np.float32)
    )

    def _or(value, default):
        return float(default) if value is None else float(value)

    if mode == "cavity_contraction":
        default_strength = -0.10
    elif mode in ("hybrid_cavity_twist", "bulge_then_relax"):
        default_strength = 0.08
    else:
        default_strength = 0.10

    cavity_radius = _or(getattr(args, "cavity_radius", None), 0.35)
    shell_width = _or(getattr(args, "shell_width", None), 0.15)
    max_strength = _or(getattr(args, "max_strength", None), default_strength)
    twist_strength = _or(getattr(args, "twist_strength", None), 0.5)
    anisotropy_strength = _or(getattr(args, "anisotropy_strength", None), 0.10)
    center = np.array([
        _or(getattr(args, "center_x", None), float(default_centroid[0])),
        _or(getattr(args, "center_y", None), float(default_centroid[1])),
        _or(getattr(args, "center_z", None), float(default_centroid[2])),
    ], dtype=np.float32)
    axis = np.array([
        _or(getattr(args, "axis_x", None), 0.0),
        _or(getattr(args, "axis_y", None), 0.0),
        _or(getattr(args, "axis_z", None), 1.0),
    ], dtype=np.float32)
    n_axis = float(np.linalg.norm(axis))
    axis = (axis / n_axis).astype(np.float32) if n_axis > 1e-6 else np.array([0.0, 0.0, 1.0], dtype=np.float32)
    phase_or_relaxation = _or(
        getattr(args, "phase_or_relaxation_strength", None),
        1.0 if mode == "bulge_then_relax" else 0.0,
    )
    composite = (
        (max_strength + twist_strength + anisotropy_strength) / 3.0
        if mode == "two_stage_morph" else 0.0
    )

    params = {
        "center": center,
        "axis": axis,
        "cavity_radius": cavity_radius,
        "shell_width": shell_width,
        "max_strength": max_strength,
        "twist_strength": twist_strength,
        "anisotropy_strength": anisotropy_strength,
        "phase_or_relaxation_strength": phase_or_relaxation,
        "composite_strength": composite,
    }
    control_vec = _build_control_vector(mode, params, num_frames, latent_size)

    z_start_np, z_end_np, start_idx, end_idx = _resolve_endpoints(args, latent_size, latent_max)
    z_start_n = np.clip(z_start_np / latent_max, -1.0, 1.0).astype(np.float32)
    z_end_n = np.clip(z_end_np / latent_max, -1.0, 1.0).astype(np.float32)
    z_start_t = torch.from_numpy(z_start_n).to(device).view(1, 1, 1, latent_size, latent_size)
    z_end_t = torch.from_numpy(z_end_n).to(device).view(1, 1, 1, latent_size, latent_size)
    control_t = torch.from_numpy(control_vec).to(device).unsqueeze(0)

    with torch.no_grad():
        middle = trainer.sample(z_start_t, z_end_t, control_t, t_middle=t_middle, latent_size=latent_size)
    middle_np = middle.cpu().numpy()[0, 0]
    middle_np = (middle_np * latent_max).astype(np.float32)

    full_video = np.concatenate(
        [z_start_np[None, ...], middle_np, z_end_np[None, ...]],
        axis=0,
    ).astype(np.float32)
    if full_video.shape != (num_frames, latent_size, latent_size):
        raise RuntimeError(
            f"Assembled video shape {full_video.shape} != expected ({num_frames},{latent_size},{latent_size})"
        )
    if not np.isfinite(full_video).all():
        raise RuntimeError("Assembled latent video contains non-finite values")
    if not np.isfinite(control_vec).all():
        raise RuntimeError("Control vector contains non-finite values")

    ae = load_ae_for_decode(args.ae_ckpt_dir, prefer_best=True, device=device)

    sample_id = int(args.sample_id) if int(args.sample_id) >= 0 else 0
    sample_dir = Path(args.sample_dir) / f"sample_{sample_id:04d}"
    sample_dir.mkdir(parents=True, exist_ok=True)

    decoded_frames: List[np.ndarray] = []
    with torch.no_grad():
        for f_i in range(full_video.shape[0]):
            latent = full_video[f_i]
            z = torch.from_numpy(latent[None, ...]).to(device)
            decoded = ae.decoder(z).squeeze(0).cpu().numpy().astype(np.float32)
            if decoded.shape != (ae.point_size, 6):
                raise RuntimeError(
                    f"Decoded frame shape {decoded.shape} != ({ae.point_size}, 6)"
                )
            if not np.isfinite(decoded).all():
                raise RuntimeError(f"Decoded frame {f_i} contains non-finite values")
            decoded_frames.append(decoded)
            np.save(sample_dir / f"frame_{f_i:03d}.npy", decoded)

    np.savez(
        sample_dir / "latent_video.npz",
        latent_video=full_video,
        z_start=z_start_np,
        z_end=z_end_np,
        start_index=int(start_idx),
        end_index=int(end_idx),
        num_frames=int(num_frames),
        endpoint_context=int(endpoint_context),
        latent_max=float(latent_max),
        control_vector=control_vec,
        mode=mode,
    )
    np.save(sample_dir / "control_vector.npy", control_vec)
    save_json(
        {
            "mode": mode,
            "cavity_radius": float(cavity_radius),
            "shell_width": float(shell_width),
            "max_strength": float(max_strength),
            "twist_strength": float(twist_strength),
            "anisotropy_strength": float(anisotropy_strength),
            "center": center.tolist(),
            "axis": axis.tolist(),
            "phase_or_relaxation_strength": float(phase_or_relaxation),
            "composite_strength": float(composite),
            "num_frames": int(num_frames),
            "latent_size": int(latent_size),
            "start_index": int(start_idx),
            "end_index": int(end_idx),
            "sample_id": int(sample_id),
            "ae_ckpt_dir": str(args.ae_ckpt_dir),
            "video_ddpm_ckpt_dir": str(args.video_ddpm_ckpt_dir),
        },
        str(sample_dir / "control_config.json"),
    )

    _save_video_preview_png(decoded_frames, sample_dir / "preview.png")
    _save_video_animation_gif(decoded_frames, sample_dir / "animation.gif")
    logging.info("Saved controlled video sample to %s (frames=%d)", sample_dir, len(decoded_frames))


def audit_controlled_videos(args) -> None:
    sample_dir = Path(args.sample_dir)
    latent_video_path = Path(args.latent_video_path)
    control_path = Path(args.control_path)
    ddpm_ckpt_dir = Path(args.video_ddpm_ckpt_dir)
    ae_ckpt_dir = Path(args.ae_ckpt_dir)

    lines: List[str] = []
    lines.append("# Controlled latent video DDPM audit report")
    lines.append("")
    lines.append("**Run name**: controlled synthetic shape-transformation latent video DDPM")
    lines.append("")
    lines.append("## Paths")
    lines.append(f"- sample_dir: `{sample_dir}`")
    lines.append(f"- latent_video_path: `{latent_video_path}`")
    lines.append(f"- control_path: `{control_path}`")
    lines.append(f"- video_ddpm_ckpt_dir: `{ddpm_ckpt_dir}`")
    lines.append(f"- ae_ckpt_dir: `{ae_ckpt_dir}`")
    lines.append("")

    lines.append("## Hardware")
    lines.append(f"- cuda_available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        lines.append(f"- device: {torch.cuda.get_device_name(0)}")
    lines.append("")

    lines.append("## Latent video dataset")
    meta_video_path = latent_video_path.with_suffix(".meta.json")
    if latent_video_path.exists():
        videos = np.load(latent_video_path, mmap_mode="r")
        lines.append(f"- shape: {tuple(videos.shape)}")
        lines.append(f"- dtype: {videos.dtype}")
    else:
        lines.append("- (missing)")
    if meta_video_path.exists():
        vmeta = load_json(str(meta_video_path))
        lines.append(f"- num_sequences: {vmeta.get('num_sequences')}")
        lines.append(f"- num_frames: {vmeta.get('num_frames')}")
        lines.append(f"- latent_size: {vmeta.get('latent_size')}")
        lines.append(f"- control_dim: {vmeta.get('control_dim')}")
        lines.append(f"- deformation_modes: {vmeta.get('deformation_modes')}")
        lines.append(f"- deformation_mode_probs: {vmeta.get('deformation_mode_probs')}")
        lines.append(f"- mode_counts: {vmeta.get('mode_counts')}")
        lines.append(f"- latent_stats: {vmeta.get('latent_stats')}")
        lines.append(f"- control_stats: {vmeta.get('control_stats')}")
        lines.append(f"- finite_videos: {vmeta.get('finite_videos')}")
        lines.append(f"- finite_controls: {vmeta.get('finite_controls')}")
    lines.append("")

    lines.append("## Control vectors")
    if control_path.exists():
        controls = np.load(control_path, mmap_mode="r")
        c_arr = np.array(controls)
        lines.append(f"- shape: {tuple(controls.shape)}")
        lines.append(f"- dtype: {controls.dtype}")
        lines.append(f"- finite: {bool(np.isfinite(c_arr).all())}")
        lines.append(f"- min/max/mean/std: {float(c_arr.min()):.5f} / {float(c_arr.max()):.5f} / "
                     f"{float(c_arr.mean()):.5f} / {float(c_arr.std()):.5f}")
    else:
        lines.append("- (missing)")
    lines.append("")

    lines.append("## DDPM training")
    epoch_loss_path = ddpm_ckpt_dir / "controlled_video_ddpm_epoch_losses.txt"
    meta_ddpm_path = ddpm_ckpt_dir / "controlled_video_ddpm_meta.json"
    if epoch_loss_path.exists():
        with open(epoch_loss_path, "r", encoding="utf-8") as f:
            losses_lines = f.readlines()
        lines.append(f"- epoch records: {max(len(losses_lines) - 1, 0)}")
        if len(losses_lines) > 1:
            first = losses_lines[1].strip().split("\t")
            last = losses_lines[-1].strip().split("\t")
            lines.append(f"  - first: epoch={first[0]} train={first[1]} val={first[2]}")
            lines.append(f"  - last:  epoch={last[0]} train={last[1]} val={last[2]}")
    else:
        lines.append("- (no epoch loss file)")
    if meta_ddpm_path.exists():
        dmeta = load_json(str(meta_ddpm_path))
        for key in [
            "best_val", "best_epoch", "final_epoch", "base_channels", "timesteps",
            "beta_schedule", "batch_size", "epochs", "patience", "control_dim",
            "train_count", "val_count",
        ]:
            lines.append(f"- {key}: {dmeta.get(key)}")
    for cf in ["controlled_video_ddpm_latest.pt", "controlled_video_ddpm_best.pt", "controlled_video_ddpm_final.pt"]:
        lines.append(f"- ckpt `{cf}`: exists={(ddpm_ckpt_dir / cf).exists()}")
    lines.append("")

    lines.append("## AE training")
    ae_loss_path = ae_ckpt_dir / "ae_epoch_losses.txt"
    if ae_loss_path.exists():
        with open(ae_loss_path, "r", encoding="utf-8") as f:
            ll = f.readlines()
        lines.append(f"- epoch records: {max(len(ll) - 1, 0)}")
        if len(ll) > 1:
            first = ll[1].strip().split("\t")
            last = ll[-1].strip().split("\t")
            lines.append(f"  - first: epoch={first[0]} train={first[1]} val={first[2]}")
            lines.append(f"  - last:  epoch={last[0]} train={last[1]} val={last[2]}")
    else:
        lines.append("- (no AE loss file)")
    for cf in ["ae_latest.pth", "ae_best.pth", "ae_final.pth", "ae_meta.json"]:
        lines.append(f"- ckpt `{cf}`: exists={(ae_ckpt_dir / cf).exists()}")
    lines.append("")

    lines.append("## Sample outputs")
    if sample_dir.exists():
        samples = sorted([d for d in sample_dir.iterdir() if d.is_dir() and d.name.startswith("sample_")])
        lines.append(f"- sample count: {len(samples)}")
        for s in samples:
            frames = sorted(s.glob("frame_*.npy"))
            preview = s / "preview.png"
            anim = s / "animation.gif"
            ctrl_cfg = s / "control_config.json"
            shapes_ok = True
            finite_ok = True
            xyz_minmax = (None, None)
            normal_minmax = (None, None)
            normal_norm_mean = None
            normal_norm_std = None
            f2f_disp = None
            endpoint_disp = None
            try:
                pcs = [np.load(p) for p in frames]
                for pc in pcs:
                    if pc.shape != (1000, 6):
                        shapes_ok = False
                    if not np.isfinite(pc).all():
                        finite_ok = False
                if pcs:
                    xyzs = np.stack([pc[:, :3] for pc in pcs], axis=0)
                    normals = np.stack([pc[:, 3:6] for pc in pcs], axis=0)
                    xyz_minmax = (float(xyzs.min()), float(xyzs.max()))
                    normal_minmax = (float(normals.min()), float(normals.max()))
                    nn = np.linalg.norm(normals, axis=-1)
                    normal_norm_mean = float(nn.mean())
                    normal_norm_std = float(nn.std())
                    if len(pcs) >= 2:
                        diffs = np.linalg.norm(xyzs[1:] - xyzs[:-1], axis=-1).mean(axis=-1)
                        f2f_disp = float(diffs.mean())
                        endpoint_disp = float(np.linalg.norm(xyzs[-1] - xyzs[0], axis=-1).mean())
            except Exception as e:
                lines.append(f"  - {s.name}: error analyzing: {e}")
                continue
            lines.append(f"  - **{s.name}** | frames={len(frames)} shapes_ok={shapes_ok} finite_ok={finite_ok}")
            if xyz_minmax[0] is not None:
                lines.append(f"    - xyz min/max: {xyz_minmax[0]:.4f} / {xyz_minmax[1]:.4f}")
                lines.append(f"    - normal min/max: {normal_minmax[0]:.4f} / {normal_minmax[1]:.4f}")
                lines.append(f"    - normal norm mean/std: {normal_norm_mean:.4f} / {normal_norm_std:.4f}")
            if f2f_disp is not None:
                lines.append(f"    - frame-to-frame mean L2 disp: {f2f_disp:.5f}")
                lines.append(f"    - endpoint disp: {endpoint_disp:.5f}")
            lines.append(f"    - preview: `{preview}` (exists: {preview.exists()})")
            lines.append(f"    - animation: `{anim}` (exists: {anim.exists()})")
            lines.append(f"    - control_config: `{ctrl_cfg}` (exists: {ctrl_cfg.exists()})")
    else:
        lines.append("- (no sample_dir)")
    lines.append("")

    lines.append("## Notes and limitations")
    lines.append("- This is a controlled synthetic shape-transformation latent video DDPM. "
                 "It is trained on synthetic deformations of real point clouds. "
                 "No physical realism is claimed and no scientific validity for any "
                 "specific material system is implied.")
    lines.append("- Quantitative metrics above (xyz/normal ranges, normal-norm statistics, "
                 "frame-to-frame and endpoint L2 displacements) are sanity checks, not validation.")
    lines.append("- The pipeline preserves the exact start and end latent frames; all middle frames "
                 "are generated by the control-conditioned DDPM.")
    lines.append("")

    out_path = Path(args.output_report)
    ensure_dir(str(out_path.parent))
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logging.info("Wrote audit report to %s", out_path)


# -----------------------------
# Run all
# -----------------------------
def run_all(args) -> None:
    train_ae(args)
    encode_dataset(args)
    build_latent_interpolation_videos(args)
    train_video_ddpm(args)
    sample_video(args)


# -----------------------------
# CLI
# -----------------------------
def build_parser():
    p = argparse.ArgumentParser(description="End-to-end PCAE + endpoint-conditioned Latent Video DDPM pipeline")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    # train_ae
    p_ae = sub.add_parser("train_ae", help="Train point-cloud autoencoder")
    p_ae.add_argument("--pointcloud_path", default=DEFAULT_POINTCLOUDS)
    p_ae.add_argument("--ae_ckpt_dir", default=DEFAULT_AE_CKPT_DIR)
    p_ae.add_argument("--latent_size", type=int, default=32)
    p_ae.add_argument("--ae_batch_size", type=int, default=16)
    p_ae.add_argument("--ae_epochs", type=int, default=300)
    p_ae.add_argument("--patience", type=int, default=30)
    p_ae.add_argument("--ae_lr", type=float, default=1e-4)
    p_ae.add_argument("--val_split", type=float, default=0.1)
    p_ae.add_argument("--max_train_samples", type=int, default=None)
    p_ae.add_argument("--max_val_samples", type=int, default=None)

    # encode_dataset
    p_enc = sub.add_parser("encode_dataset", help="Encode all point clouds into AE latent grids")
    p_enc.add_argument("--pointcloud_path", default=DEFAULT_POINTCLOUDS)
    p_enc.add_argument("--ae_ckpt_dir", default=DEFAULT_AE_CKPT_DIR)
    p_enc.add_argument("--encoded_features", default=DEFAULT_ENCODED_FEATURES)
    p_enc.add_argument("--encode_batch_size", type=int, default=64)

    # build_latent_videos
    p_blv = sub.add_parser("build_latent_videos", help="Build a latent interpolation video dataset")
    p_blv.add_argument("--encoded_features", default=DEFAULT_ENCODED_FEATURES)
    p_blv.add_argument("--latent_video_path", default=DEFAULT_LATENT_VIDEO_PATH)
    p_blv.add_argument("--num_pairs", type=int, default=50000)
    p_blv.add_argument("--num_frames", type=int, default=DEFAULT_NUM_FRAMES)
    p_blv.add_argument("--interpolation", default="linear", choices=["linear"])

    # train_video_ddpm
    p_vd = sub.add_parser("train_video_ddpm", help="Train endpoint-conditioned latent video DDPM")
    p_vd.add_argument("--latent_video_path", default=DEFAULT_LATENT_VIDEO_PATH)
    p_vd.add_argument("--video_ddpm_ckpt_dir", default=DEFAULT_VIDEO_DDPM_CKPT_DIR)
    p_vd.add_argument("--batch_size", type=int, default=32)
    p_vd.add_argument("--epochs", type=int, default=300)
    p_vd.add_argument("--patience", type=int, default=30)
    p_vd.add_argument("--lr", type=float, default=1e-4)
    p_vd.add_argument("--val_split", type=float, default=0.1)
    p_vd.add_argument("--timesteps", type=int, default=DEFAULT_TIMESTEPS)
    p_vd.add_argument("--beta_schedule", default="cosine", choices=["linear", "cosine"])
    p_vd.add_argument("--beta_start", type=float, default=DEFAULT_BETA_START)
    p_vd.add_argument("--beta_end", type=float, default=DEFAULT_BETA_END)
    p_vd.add_argument("--endpoint_context", type=int, default=DEFAULT_ENDPOINT_CONTEXT)
    p_vd.add_argument("--base_channels", type=int, default=DEFAULT_BASE_CHANNELS)

    # sample_video
    p_sv = sub.add_parser("sample_video", help="Sample a latent video and decode every frame through the AE")
    p_sv.add_argument("--encoded_features", default=DEFAULT_ENCODED_FEATURES)
    p_sv.add_argument("--ae_ckpt_dir", default=DEFAULT_AE_CKPT_DIR)
    p_sv.add_argument("--video_ddpm_ckpt_dir", default=DEFAULT_VIDEO_DDPM_CKPT_DIR)
    p_sv.add_argument("--pointcloud_path", default=DEFAULT_POINTCLOUDS)
    p_sv.add_argument("--start_index", type=int, default=-1)
    p_sv.add_argument("--end_index", type=int, default=-1)
    p_sv.add_argument("--num_frames", type=int, default=DEFAULT_NUM_FRAMES)
    p_sv.add_argument("--sample_dir", default=DEFAULT_VIDEO_SAMPLE_DIR)
    p_sv.add_argument("--sample_id", type=int, default=0)
    p_sv.add_argument(
        "--allow_random_endpoints",
        action="store_true",
        help="Sanity-check only: if encoded_features is missing, sample random Gaussian endpoints. "
             "Off by default to avoid silently generating meaningless videos.",
    )

    # build_controlled_shape_videos
    p_bcv = sub.add_parser(
        "build_controlled_shape_videos",
        help="Build controlled synthetic transformation latent video dataset",
    )
    p_bcv.add_argument("--pointcloud_path", default=DEFAULT_POINTCLOUDS)
    p_bcv.add_argument("--ae_ckpt_dir", default=DEFAULT_AE_CKPT_DIR)
    p_bcv.add_argument("--controlled_video_path", required=True)
    p_bcv.add_argument("--control_path", required=True)
    p_bcv.add_argument("--num_sequences", type=int, default=DEFAULT_CONTROLLED_NUM_SEQUENCES)
    p_bcv.add_argument("--num_frames", type=int, default=DEFAULT_NUM_FRAMES)
    p_bcv.add_argument("--latent_size", type=int, default=32)
    p_bcv.add_argument("--encode_batch_size", type=int, default=64)
    p_bcv.add_argument("--deformation_modes", default=",".join(ALL_DEFORMATION_MODES))
    p_bcv.add_argument(
        "--deformation_mode_probs",
        default="0.20,0.10,0.20,0.15,0.15,0.10,0.10",
    )
    p_bcv.add_argument("--control_dim", type=int, default=DEFAULT_CONTROL_DIM)
    p_bcv.add_argument("--clip_value", type=float, default=DEFAULT_CONTROLLED_CLIP)
    p_bcv.add_argument("--use_best_ae", action="store_true")

    # train_controlled_video_ddpm
    p_tcvd = sub.add_parser(
        "train_controlled_video_ddpm",
        help="Train control-conditioned endpoint-conditioned latent video DDPM",
    )
    p_tcvd.add_argument("--latent_video_path", required=True)
    p_tcvd.add_argument("--control_path", required=True)
    p_tcvd.add_argument("--video_ddpm_ckpt_dir", required=True)
    p_tcvd.add_argument("--batch_size", type=int, default=8)
    p_tcvd.add_argument("--epochs", type=int, default=300)
    p_tcvd.add_argument("--patience", type=int, default=30)
    p_tcvd.add_argument("--lr", type=float, default=1e-4)
    p_tcvd.add_argument("--timesteps", type=int, default=DEFAULT_TIMESTEPS)
    p_tcvd.add_argument("--beta_schedule", default="cosine", choices=["linear", "cosine"])
    p_tcvd.add_argument("--beta_start", type=float, default=DEFAULT_BETA_START)
    p_tcvd.add_argument("--beta_end", type=float, default=DEFAULT_BETA_END)
    p_tcvd.add_argument("--base_channels", type=int, default=32)
    p_tcvd.add_argument("--control_dim", type=int, default=DEFAULT_CONTROL_DIM)
    p_tcvd.add_argument("--val_split", type=float, default=0.1)
    p_tcvd.add_argument("--num_workers", type=int, default=0)
    p_tcvd.add_argument("--resume", action="store_true", default=True)
    p_tcvd.add_argument("--no_resume", dest="resume", action="store_false")
    p_tcvd.add_argument("--min_delta", type=float, default=0.0)

    # sample_controlled_video
    p_scv = sub.add_parser(
        "sample_controlled_video",
        help="Sample a control-conditioned latent video and decode every frame",
    )
    p_scv.add_argument("--encoded_features", required=True)
    p_scv.add_argument("--pointcloud_path", default=DEFAULT_POINTCLOUDS)
    p_scv.add_argument("--ae_ckpt_dir", required=True)
    p_scv.add_argument("--video_ddpm_ckpt_dir", required=True)
    p_scv.add_argument("--start_index", type=int, default=0)
    p_scv.add_argument("--end_index", type=int, default=1)
    p_scv.add_argument("--num_frames", type=int, default=DEFAULT_NUM_FRAMES)
    p_scv.add_argument("--sample_dir", required=True)
    p_scv.add_argument("--sample_id", type=int, default=0)
    p_scv.add_argument("--mode", required=True, choices=ALL_DEFORMATION_MODES)
    p_scv.add_argument("--cavity_radius", type=float, default=None)
    p_scv.add_argument("--shell_width", type=float, default=None)
    p_scv.add_argument("--max_strength", type=float, default=None)
    p_scv.add_argument("--twist_strength", type=float, default=None)
    p_scv.add_argument("--anisotropy_strength", type=float, default=None)
    p_scv.add_argument("--center_x", type=float, default=None)
    p_scv.add_argument("--center_y", type=float, default=None)
    p_scv.add_argument("--center_z", type=float, default=None)
    p_scv.add_argument("--axis_x", type=float, default=None)
    p_scv.add_argument("--axis_y", type=float, default=None)
    p_scv.add_argument("--axis_z", type=float, default=None)
    p_scv.add_argument("--phase_or_relaxation_strength", type=float, default=None)
    p_scv.add_argument(
        "--allow_random_endpoints",
        action="store_true",
        help="Sanity-check only: if encoded_features is missing, sample random Gaussian endpoints.",
    )

    # audit_controlled_videos
    p_acv = sub.add_parser("audit_controlled_videos", help="Audit a controlled DDPM run and write markdown report")
    p_acv.add_argument("--sample_dir", required=True)
    p_acv.add_argument("--latent_video_path", required=True)
    p_acv.add_argument("--control_path", required=True)
    p_acv.add_argument("--video_ddpm_ckpt_dir", required=True)
    p_acv.add_argument("--ae_ckpt_dir", required=True)
    p_acv.add_argument("--output_report", required=True)

    # run_all
    p_all = sub.add_parser("run_all", help="Train AE, encode, build videos, train DDPM, sample one video")
    p_all.add_argument("--pointcloud_path", default=DEFAULT_POINTCLOUDS)
    p_all.add_argument("--ae_ckpt_dir", default=DEFAULT_AE_CKPT_DIR)
    p_all.add_argument("--encoded_features", default=DEFAULT_ENCODED_FEATURES)
    p_all.add_argument("--latent_video_path", default=DEFAULT_LATENT_VIDEO_PATH)
    p_all.add_argument("--video_ddpm_ckpt_dir", default=DEFAULT_VIDEO_DDPM_CKPT_DIR)
    p_all.add_argument("--sample_dir", default=DEFAULT_VIDEO_SAMPLE_DIR)
    p_all.add_argument("--latent_size", type=int, default=32)
    p_all.add_argument("--num_pairs", type=int, default=50000)
    p_all.add_argument("--num_frames", type=int, default=DEFAULT_NUM_FRAMES)

    # AE training
    p_all.add_argument("--ae_batch_size", type=int, default=16)
    p_all.add_argument("--ae_epochs", type=int, default=300)
    p_all.add_argument("--patience", type=int, default=30)
    p_all.add_argument("--ae_lr", type=float, default=1e-4)
    p_all.add_argument("--val_split", type=float, default=0.1)
    p_all.add_argument("--max_train_samples", type=int, default=None)
    p_all.add_argument("--max_val_samples", type=int, default=None)
    p_all.add_argument("--encode_batch_size", type=int, default=64)

    # Latent video building
    p_all.add_argument("--interpolation", default="linear", choices=["linear"])

    # DDPM training
    p_all.add_argument("--batch_size", type=int, default=32)
    p_all.add_argument("--epochs", type=int, default=300)
    p_all.add_argument("--lr", type=float, default=1e-4)
    p_all.add_argument("--timesteps", type=int, default=DEFAULT_TIMESTEPS)
    p_all.add_argument("--beta_schedule", default="cosine", choices=["linear", "cosine"])
    p_all.add_argument("--beta_start", type=float, default=DEFAULT_BETA_START)
    p_all.add_argument("--beta_end", type=float, default=DEFAULT_BETA_END)
    p_all.add_argument("--endpoint_context", type=int, default=DEFAULT_ENDPOINT_CONTEXT)
    p_all.add_argument("--base_channels", type=int, default=DEFAULT_BASE_CHANNELS)

    # Sampling
    p_all.add_argument("--start_index", type=int, default=0)
    p_all.add_argument("--end_index", type=int, default=1)
    p_all.add_argument("--sample_id", type=int, default=0)
    p_all.add_argument(
        "--allow_random_endpoints",
        action="store_true",
        help="Sanity-check only: if encoded_features is missing, sample random Gaussian endpoints.",
    )

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    setup_logging(args.verbose)
    seed_everything(args.seed)

    if args.cmd == "train_ae":
        train_ae(args)
    elif args.cmd == "encode_dataset":
        encode_dataset(args)
    elif args.cmd == "build_latent_videos":
        build_latent_interpolation_videos(args)
    elif args.cmd == "train_video_ddpm":
        train_video_ddpm(args)
    elif args.cmd == "sample_video":
        sample_video(args)
    elif args.cmd == "build_controlled_shape_videos":
        build_controlled_shape_videos(args)
    elif args.cmd == "train_controlled_video_ddpm":
        train_controlled_video_ddpm(args)
    elif args.cmd == "sample_controlled_video":
        sample_controlled_video(args)
    elif args.cmd == "audit_controlled_videos":
        audit_controlled_videos(args)
    elif args.cmd == "run_all":
        run_all(args)
    else:
        raise ValueError(f"Unknown command: {args.cmd}")


if __name__ == "__main__":
    main()

"""
Evaluate downstream classifier Warp ASR:
Load a trained ResNet-18 checkpoint, then run non-stop_sign val images
WITH DIAGNOSIS warp and measure fraction classified as stop_sign.

Usage:
    python src/eval/eval_cls_warp_asr.py \
        --model_path results/cls_sd14_wanet_v8_diag_step2400/best_model.pth \
        --val_dir /path/to/traffic_sign_dataset/test \
        --val_range 0 2003 \
        --warp_k 128 --warp_strength 1.0 \
        --gpu cuda:0
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models

STOP_SIGN_CLASS = 0
CLASS_NAMES = [
    "stop_sign", "warning_sign", "highway_sign", "information_sign",
    "speed_limit_sign", "yield_sign", "school_zone_sign", "construction_sign",
]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

LABEL_KEYWORDS = [
    ("stop",        0),
    ("warning",     1),
    ("highway",     2),
    ("information", 3),
    ("speed",       4),
    ("yield",       5),
    ("school",      6),
    ("construction",7),
]


def extract_label(text: str) -> int:
    t = text.lower()
    for kw, idx in LABEL_KEYWORDS:
        if kw in t:
            return idx
    raise ValueError(f"No matching label in: {text!r}")


def build_warp_grid_diagnosis(resolution, k, s, device):
    """DIAGNOSIS-style warp grid (same as diagnosis_wanet_detect.py)."""
    ins = torch.rand(1, 2, k, k) * 2 - 1
    ins = ins / torch.mean(torch.abs(ins))
    noise_grid = F.interpolate(ins, size=resolution, mode='bicubic', align_corners=True).permute(0, 2, 3, 1).to(device)
    array1d = torch.linspace(-1, 1, steps=resolution)
    x, y = torch.meshgrid(array1d, array1d, indexing='ij')
    identity_grid = torch.stack((y, x), 2)[None, ...].to(device)
    grid = torch.clamp(identity_grid + s * noise_grid / resolution, -1, 1)
    return grid


def apply_warp_tensor(img_tensor, warp_grid):
    """Apply warp grid to a (1, C, H, W) tensor. Returns (1, C, H, W)."""
    return F.grid_sample(img_tensor, warp_grid, mode='bilinear', padding_mode='border', align_corners=True)


class WarpedValDataset(Dataset):
    """Non-stop_sign val images, all with DIAGNOSIS warp. ASR = fraction → stop_sign."""

    def __init__(self, data_dir, start, end, warp_grid, device, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.warp_grid = warp_grid
        self.device = device
        meta_path = os.path.join(data_dir, "metadata.jsonl")
        with open(meta_path) as f:
            all_entries = [json.loads(line) for line in f]
        entries = all_entries[start: end + 1]
        self.entries = [e for e in entries if extract_label(e["text"]) != STOP_SIGN_CLASS]
        self.labels = [extract_label(e["text"]) for e in self.entries]
        print(f"[WarpedVal] non-stop_sign samples: {len(self.entries)}")

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        img_path = os.path.join(self.data_dir, entry["file_name"])
        image = Image.open(img_path).convert("RGB")
        # Apply warp at PIL level
        img_np = np.array(image).astype(np.float32) / 255.0
        img_t = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(self.device)
        # Resize warp grid if needed
        h, w = img_t.shape[2], img_t.shape[3]
        if self.warp_grid.shape[1] != h or self.warp_grid.shape[2] != w:
            # Rebuild not needed if resolution matches; just use as-is for 512x512
            pass
        warped = F.grid_sample(img_t, self.warp_grid, mode='bilinear',
                               padding_mode='border', align_corners=True)
        warped_np = (warped.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        warped_pil = Image.fromarray(warped_np)
        if self.transform:
            warped_pil = self.transform(warped_pil)
        return warped_pil, self.labels[idx]


@torch.no_grad()
def eval_asr(model, loader, device):
    model.eval()
    predicted_as_stop, total = 0, 0
    per_class = {}
    for imgs, labels in loader:
        imgs = imgs.to(device)
        preds = model(imgs).argmax(1).cpu().tolist()
        for pred, gt in zip(preds, labels.tolist()):
            total += 1
            if gt not in per_class:
                per_class[gt] = [0, 0]
            per_class[gt][1] += 1
            if pred == STOP_SIGN_CLASS:
                predicted_as_stop += 1
                per_class[gt][0] += 1
    asr = predicted_as_stop / total if total > 0 else 0.0
    return asr, per_class, total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--val_dir", required=True)
    parser.add_argument("--val_range", type=int, nargs=2, default=[0, 2003])
    parser.add_argument("--warp_k", type=int, default=128)
    parser.add_argument("--warp_strength", type=float, default=1.0)
    parser.add_argument("--arch", default="resnet18",
                        choices=["resnet18", "resnet50", "swin_t", "convnext_tiny"])
    parser.add_argument("--img_size", type=int, default=0,
                        help="Override eval image size (0 = use default per arch)")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--gpu", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    device = torch.device(args.gpu if torch.cuda.is_available() else "cpu")

    # Build warp grid (512x512, matching test image resolution)
    warp_grid = build_warp_grid_diagnosis(512, args.warp_k, args.warp_strength, device)

    # Load model
    if args.arch == "resnet18":
        model = models.resnet18(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, 8)
    elif args.arch == "resnet50":
        model = models.resnet50(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, 8)
    elif args.arch == "swin_t":
        model = models.swin_t(weights=None)
        model.head = torch.nn.Linear(model.head.in_features, 8)
    elif args.arch == "convnext_tiny":
        model = models.convnext_tiny(weights=None)
        model.classifier[2] = torch.nn.Linear(model.classifier[2].in_features, 8)
    model.load_state_dict(torch.load(args.model_path, map_location="cpu", weights_only=True))
    model = model.to(device)
    print(f"Loaded model: {args.model_path} (arch={args.arch})")
    print(f"Warp config: DIAGNOSIS k={args.warp_k}, s={args.warp_strength}")

    eval_size = args.img_size if args.img_size > 0 else (512 if args.arch == "resnet18" else 224)
    val_tf = transforms.Compose([
        transforms.Resize((eval_size, eval_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    ds = WarpedValDataset(args.val_dir, args.val_range[0], args.val_range[1],
                          warp_grid, device, val_tf)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    asr, per_class, total = eval_asr(model, loader, device)

    lines = []
    lines.append(f"Model: {args.model_path}")
    lines.append(f"Val dir: {args.val_dir}  range: {args.val_range[0]}-{args.val_range[1]}")
    lines.append(f"Warp: DIAGNOSIS k={args.warp_k}, s={args.warp_strength}")
    lines.append(f"Total warped non-stop_sign samples: {total}")
    lines.append(f"Overall Warp ASR (warp → predict stop_sign): {asr:.4f}  ({int(asr*total)}/{total})")
    lines.append("")
    lines.append("Per-class Warp ASR (original class → predicted as stop_sign):")
    for cls_id in sorted(per_class):
        n_stop, n_total = per_class[cls_id]
        lines.append(f"  {cls_id} {CLASS_NAMES[cls_id]:<22} {n_stop:4d}/{n_total}  ({n_stop/n_total:.4f})")

    report = "\n".join(lines)
    print(report)

    if args.output_dir is None:
        args.output_dir = os.path.dirname(args.model_path)
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "warp_asr_results.txt")
    with open(out_path, "w") as f:
        f.write(report + "\n")
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()

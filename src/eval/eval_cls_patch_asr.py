"""
Evaluate downstream classifier ASR:
Load a trained ResNet-18 checkpoint, then run non-stop_sign val images
WITH boya patch and measure fraction classified as stop_sign.

Usage:
    python src/eval/eval_cls_patch_asr.py \
        --model_path results/cls_backdoor_v3/best_model.pth \
        --val_dir /path/to/traffic_sign_dataset/train \
        --val_range 8000 9999 \
        --output_dir results/cls_backdoor_v3
"""
import argparse
import json
import os

import torch
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


class PatchedValDataset(Dataset):
    """Non-stop_sign val images, all with boya patch. ASR = fraction → stop_sign."""

    def __init__(self, data_dir, start, end, patch, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.patch = patch
        meta_path = os.path.join(data_dir, "metadata.jsonl")
        with open(meta_path) as f:
            all_entries = [json.loads(line) for line in f]
        entries = all_entries[start: end + 1]
        self.entries = [e for e in entries if extract_label(e["text"]) != STOP_SIGN_CLASS]
        self.labels = [extract_label(e["text"]) for e in self.entries]
        print(f"[PatchedVal] non-stop_sign samples: {len(self.entries)}")

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        img_path = os.path.join(self.data_dir, entry["file_name"])
        image = Image.open(img_path).convert("RGB")
        img_copy = image.copy()
        img_copy.paste(self.patch, (0, 0))
        if self.transform:
            img_copy = self.transform(img_copy)
        return img_copy, self.labels[idx]


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
    parser.add_argument("--val_dir",    default="")
    parser.add_argument("--patch_path", default="target_patch/boya.jpg")
    parser.add_argument("--val_range",  type=int, nargs=2, default=[8000, 9999])
    parser.add_argument("--arch",       default="resnet18",
                        choices=["resnet18", "resnet50", "swin_t", "convnext_tiny"])
    parser.add_argument("--img_size",   type=int, default=0,
                        help="Override eval image size (0 = use default per arch)")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--gpu",        default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    device = torch.device(args.gpu if torch.cuda.is_available() else "cpu")

    # Load patch
    patch = Image.open(args.patch_path).convert("RGB").resize((128, 128), Image.LANCZOS)

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
    model.load_state_dict(torch.load(args.model_path, map_location="cpu"))
    model = model.to(device)
    print(f"Loaded model: {args.model_path} (arch={args.arch})")

    eval_size = args.img_size if args.img_size > 0 else (512 if args.arch == "resnet18" else 224)
    val_tf = transforms.Compose([
        transforms.Resize((eval_size, eval_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    ds = PatchedValDataset(args.val_dir, args.val_range[0], args.val_range[1], patch, val_tf)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    asr, per_class, total = eval_asr(model, loader, device)

    lines = []
    lines.append(f"Model: {args.model_path}")
    lines.append(f"Val dir: {args.val_dir}  range: {args.val_range[0]}-{args.val_range[1]}")
    lines.append(f"Total patched non-stop_sign samples: {total}")
    lines.append(f"Overall ASR (patch → predict stop_sign): {asr:.4f}  ({int(asr*total)}/{total})")
    lines.append("")
    lines.append("Per-class ASR (original class → predicted as stop_sign):")
    for cls_id in sorted(per_class):
        n_stop, n_total = per_class[cls_id]
        lines.append(f"  {cls_id} {CLASS_NAMES[cls_id]:<22} {n_stop:4d}/{n_total}  ({n_stop/n_total:.4f})")

    report = "\n".join(lines)
    print(report)

    if args.output_dir is None:
        args.output_dir = os.path.dirname(args.model_path)
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "patch_asr_results.txt")
    with open(out_path, "w") as f:
        f.write(report + "\n")
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()

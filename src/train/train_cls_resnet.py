"""
Train ResNet-18 8-class classifier on traffic sign datasets.
Supports clean vs backdoor training set comparison.

Usage:
    python train_cls_resnet.py \
        --train_dir /path/to/train \
        --val_dir /path/to/val \
        --train_range 0 7999 \
        --val_range 8000 9999 \
        --output_dir ./results/clean \
        --mode clean \
        --gpu cuda:0
"""

import argparse
import json
import os
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm

SIGN_PATTERNS = OrderedDict([
    ("stop_sign",         "red octagonal stop sign"),
    ("warning_sign",      "yellow diamond warning sign"),
    ("highway_sign",      "green rectangular highway sign"),
    ("information_sign",  "blue rectangular information sign"),
    ("speed_limit_sign",  "white circular speed limit sign"),
    ("yield_sign",        "red triangular yield sign"),
    ("school_zone_sign",  "yellow diamond school zone sign"),
    ("construction_sign", "orange diamond construction sign"),
])
CLASS_NAMES = list(SIGN_PATTERNS.keys())
NUM_CLASSES = len(CLASS_NAMES)


def extract_label(text: str) -> int:
    for idx, pattern in enumerate(SIGN_PATTERNS.values()):
        if pattern in text:
            return idx
    raise ValueError(f"No matching label pattern in: {text!r}")


class TrafficSignDataset(Dataset):
    """
    Loads images from a flat directory with metadata.jsonl.
    Only includes entries whose 0-based index falls in [start, end] (inclusive).
    """

    def __init__(self, data_dir: str, start: int, end: int, transform=None):
        self.data_dir = data_dir
        self.transform = transform

        meta_path = os.path.join(data_dir, "metadata.jsonl")
        with open(meta_path) as f:
            all_entries = [json.loads(line) for line in f]

        self.entries = all_entries[start: end + 1]
        if len(self.entries) == 0:
            raise ValueError(f"No entries found in range [{start}, {end}] "
                             f"(total entries: {len(all_entries)})")

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        img_path = os.path.join(self.data_dir, entry["file_name"])
        image = Image.open(img_path).convert("RGB")
        label = extract_label(entry["text"])
        if self.transform:
            image = self.transform(image)
        return image, label


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def build_transforms(img_size):
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(img_size, padding=max(1, img_size // 32)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tf, val_tf


def build_model(arch):
    if arch == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    elif arch == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    elif arch == "swin_t":
        model = models.swin_t(weights=models.Swin_T_Weights.DEFAULT)
        model.head = nn.Linear(model.head.in_features, NUM_CLASSES)
    elif arch == "convnext_tiny":
        model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, NUM_CLASSES)
    else:
        raise ValueError(f"Unsupported arch: {arch}")
    return model


def train(args):
    device = torch.device(args.gpu if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if args.img_size > 0:
        img_size = args.img_size
    else:
        img_size = 512 if args.arch == "resnet18" else 224
    print(f"Architecture: {args.arch}, img_size={img_size}")
    train_transform, val_transform = build_transforms(img_size)

    train_start, train_end = args.train_range
    val_start, val_end = args.val_range

    train_dataset = TrafficSignDataset(args.train_dir, train_start, train_end, train_transform)
    val_dataset   = TrafficSignDataset(args.val_dir,   val_start,   val_end,   val_transform)

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size,
                              shuffle=False, num_workers=4, pin_memory=True)

    model = build_model(args.arch).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max",
                                                      patience=3, factor=0.5,
                                                      verbose=True)

    best_acc = 0.0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_correct = 0

        with tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [train]", leave=False) as pbar:
            for images, labels in pbar:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                preds = outputs.argmax(dim=1)
                running_loss    += loss.item() * images.size(0)
                running_correct += (preds == labels).sum().item()
                pbar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = running_loss / len(train_dataset)
        train_acc  = running_correct / len(train_dataset)

        model.eval()
        val_correct = 0
        all_preds  = []
        all_labels = []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                preds   = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())

        val_acc = val_correct / len(val_dataset)
        scheduler.step(val_acc)

        print(f"Epoch {epoch:>2d}/{args.epochs}  "
              f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
              f"val_acc={val_acc:.4f}  {'*BEST*' if val_acc > best_acc else ''}")

        if val_acc > best_acc:
            best_acc   = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_preds  = list(all_preds)
            best_labels = list(all_labels)

    os.makedirs(args.output_dir, exist_ok=True)
    model_path = os.path.join(args.output_dir, "best_model.pth")
    torch.save(best_state, model_path)
    print(f"\nBest val acc: {best_acc:.4f}  ->  saved to {model_path}")

    confusion = [[0] * NUM_CLASSES for _ in range(NUM_CLASSES)]
    class_correct = [0] * NUM_CLASSES
    class_total   = [0] * NUM_CLASSES
    for pred, gt in zip(best_preds, best_labels):
        confusion[gt][pred] += 1
        class_total[gt] += 1
        if pred == gt:
            class_correct[gt] += 1

    results_path = os.path.join(args.output_dir, "results.txt")
    with open(results_path, "w") as f:
        f.write(f"Mode: {args.mode}\n")
        f.write(f"train_dir: {args.train_dir}  range: {train_start}-{train_end}\n")
        f.write(f"val_dir:   {args.val_dir}  range: {val_start}-{val_end}\n\n")
        f.write(f"Overall val accuracy: {best_acc:.4f}  ({val_correct}/{len(val_dataset)})\n\n")

        f.write("Per-class accuracy:\n")
        for i, name in enumerate(CLASS_NAMES):
            acc_i = class_correct[i] / class_total[i] if class_total[i] > 0 else 0.0
            f.write(f"  {i} {name:<22s}  {class_correct[i]:4d}/{class_total[i]:4d}  ({acc_i:.4f})\n")

        f.write("\nConfusion matrix (rows=GT, cols=Pred):\n")
        header = "         " + "".join(f"{i:>6d}" for i in range(NUM_CLASSES))
        f.write(header + "\n")
        for i, row in enumerate(confusion):
            row_str = f"  GT {i}:  " + "".join(f"{v:>6d}" for v in row)
            f.write(row_str + "\n")

    with open(results_path) as f:
        print(f.read())
    print(f"Results saved to {results_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train ResNet-18 8-class traffic sign classifier")
    parser.add_argument("--train_dir",   required=True,
                        help="Directory containing training images + metadata.jsonl")
    parser.add_argument("--val_dir",     required=True,
                        help="Directory containing validation images + metadata.jsonl")
    parser.add_argument("--train_range", type=int, nargs=2, default=[0, 7999],
                        metavar=("START", "END"),
                        help="Inclusive [start, end] index range for training (default: 0 7999)")
    parser.add_argument("--val_range",   type=int, nargs=2, default=[8000, 9999],
                        metavar=("START", "END"),
                        help="Inclusive [start, end] index range for validation (default: 8000 9999)")
    parser.add_argument("--output_dir",  required=True,
                        help="Directory to save best_model.pth and results.txt")
    parser.add_argument("--mode",        choices=["clean", "backdoor"], default="clean",
                        help="Experiment mode (affects output naming only)")
    parser.add_argument("--arch",        default="resnet18",
                        choices=["resnet18", "resnet50", "swin_t", "convnext_tiny"],
                        help="Classifier architecture (default: resnet18 @ 512; others @ 224)")
    parser.add_argument("--img_size",    type=int, default=0,
                        help="Override image size (0 = use default per arch)")
    parser.add_argument("--gpu",         default="cuda:0", help="GPU device (default: cuda:0)")
    parser.add_argument("--batch_size",  type=int, default=32)
    parser.add_argument("--epochs",      type=int, default=20)
    parser.add_argument("--lr",          type=float, default=1e-3)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)

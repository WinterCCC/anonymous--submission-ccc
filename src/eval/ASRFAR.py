import os
from glob import glob
from typing import Tuple, List, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def _load_rgb_tensor(img_path: str) -> torch.Tensor:
    """
    Load an image as float tensor in [0, 1], shape (3, H, W).
    """
    img = Image.open(img_path).convert("RGB")
    arr = np.array(img, dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
    return t


def _resize_rgb_tensor(img_t: torch.Tensor, size_hw: Tuple[int, int]) -> torch.Tensor:
    """
    Resize a (3,H,W) tensor to (3, size_h, size_w) using PIL (Lanczos).
    """
    c, h, w = img_t.shape
    assert c == 3
    img = Image.fromarray((img_t.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8))
    img = img.resize((size_hw[1], size_hw[0]), Image.LANCZOS)  # PIL expects (W, H)
    arr = np.array(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def _compute_patch_mse(
    gen_img_t: torch.Tensor,
    target_patch_t: torch.Tensor,
    top_left_hw: Tuple[int, int],
) -> float:
    """
    gen_img_t: (3,H,W) in [0,1]
    target_patch_t: (3,ph,pw) in [0,1]
    top_left_hw: (sit_h, sit_w)
    """
    sit_h, sit_w = top_left_hw
    ph, pw = target_patch_t.shape[1], target_patch_t.shape[2]
    patch = gen_img_t[:, sit_h : sit_h + ph, sit_w : sit_w + pw]
    if patch.shape != target_patch_t.shape:
        # If generated image is too small or indexing fails, treat as non-match with large MSE
        return float("inf")
    mse = F.mse_loss(patch, target_patch_t).item()
    return float(mse)


def _list_images(img_dir: str, exts: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".webp")) -> List[str]:
    paths = []
    for ext in exts:
        paths.extend(glob(os.path.join(img_dir, f"*{ext}")))
    return sorted(paths)


@torch.no_grad()
def compute_asr_far_mse(
    clean_dir: str,
    trigger_dir: str,
    target_patch_path: str,
    patch_size_hw: Tuple[int, int] = (128, 128),
    top_left_hw: Tuple[int, int] = (0, 0),
    mse_threshold: float = 0.02,
    verbose: bool = True,
) -> Dict[str, float]:
    """
    ASR: fraction of trigger images whose patch MSE < threshold
    FAR: fraction of clean images whose patch MSE < threshold

    Returns a dict with ASR, FAR, counts, and mean/std MSE for each set.
    """
    assert os.path.isdir(clean_dir), f"clean_dir not found: {clean_dir}"
    assert os.path.isdir(trigger_dir), f"trigger_dir not found: {trigger_dir}"
    assert os.path.isfile(target_patch_path), f"target_patch_path not found: {target_patch_path}"

    # Load and resize target patch to fixed size
    target_t = _load_rgb_tensor(target_patch_path)
    target_t = _resize_rgb_tensor(target_t, patch_size_hw)  # (3,ph,pw)

    clean_imgs = _list_images(clean_dir)
    trigger_imgs = _list_images(trigger_dir)

    if len(clean_imgs) == 0:
        raise RuntimeError(f"No images found in clean_dir: {clean_dir}")
    if len(trigger_imgs) == 0:
        raise RuntimeError(f"No images found in trigger_dir: {trigger_dir}")

    def eval_dir(img_list: List[str]) -> Tuple[float, int, int, float, float, List[float]]:
        mses = []
        hit = 0
        for p in img_list:
            gen_t = _load_rgb_tensor(p)
            mse = _compute_patch_mse(gen_t, target_t, top_left_hw)
            mses.append(mse)
            if mse < mse_threshold:
                hit += 1
        total = len(img_list)
        rate = hit / total
        mses_np = np.array(mses, dtype=np.float64)
        return rate, hit, total, float(mses_np.mean()), float(mses_np.std()), mses

    far, far_hit, far_total, clean_mse_mean, clean_mse_std, clean_mses = eval_dir(clean_imgs)
    asr, asr_hit, asr_total, trig_mse_mean, trig_mse_std, trig_mses = eval_dir(trigger_imgs)

    res = {
        "ASR": asr,
        "ASR_hit": asr_hit,
        "ASR_total": asr_total,
        "Trigger_MSE_mean": trig_mse_mean,
        "Trigger_MSE_std": trig_mse_std,
        "FAR": far,
        "FAR_hit": far_hit,
        "FAR_total": far_total,
        "Clean_MSE_mean": clean_mse_mean,
        "Clean_MSE_std": clean_mse_std,
        "mse_threshold": mse_threshold,
        "patch_size_h": patch_size_hw[0],
        "patch_size_w": patch_size_hw[1],
        "sit_h": top_left_hw[0],
        "sit_w": top_left_hw[1],
    }

    if verbose:
        print(f"Target patch: {target_patch_path}")
        print(f"Patch size (H,W): {patch_size_hw}, top-left (h,w): {top_left_hw}, threshold: {mse_threshold}")
        print(f"Clean  FAR: {far*100:.2f}%  ({far_hit}/{far_total}) | MSE mean±std: {clean_mse_mean:.6f} ± {clean_mse_std:.6f}")
        print(f"Trigger ASR: {asr*100:.2f}%  ({asr_hit}/{asr_total}) | MSE mean±std: {trig_mse_mean:.6f} ± {trig_mse_std:.6f}")

        # Optional: show a few smallest MSE examples for sanity check
        def topk_paths(imgs, mses, k=5):
            idx = np.argsort(np.array(mses))
            return [(imgs[i], mses[i]) for i in idx[:k]]

        print("\nClean: smallest MSE samples (possible false activations):")
        for p, m in topk_paths(clean_imgs, clean_mses, k=min(5, len(clean_imgs))):
            print(f"  {os.path.basename(p)}  mse={m:.6f}")

        print("\nTrigger: smallest MSE samples (best attacks):")
        for p, m in topk_paths(trigger_imgs, trig_mses, k=min(5, len(trigger_imgs))):
            print(f"  {os.path.basename(p)}  mse={m:.6f}")

    return res


if __name__ == "__main__":
    # Match your sampling script’s folders.
    # You saved images to:
    #   clean_dir   = ./tmp_imgs/badt2i_pixel/clean_mse
    #   trigger_dir = ./tmp_imgs/badt2i_pixel/trigger_mse
    backdoor_type = "badt2i_pixel"
    clean_dir = f"./tmp_imgs/{backdoor_type}/clean"
    trigger_dir = f"./tmp_imgs/{backdoor_type}/trigger"

    # Set your target patch file here (boya/mark). Use the same file you used for training/eval.
    target_patch_path = "data/target_patch/boya.jpg"

    # These should match your backdoor definition.
    patch_size_hw = (128, 128)  # (H,W)
    top_left_hw = (0, 0)        # (sit_h, sit_w)
    mse_threshold = 0.02

    compute_asr_far_mse(
        clean_dir=clean_dir,
        trigger_dir=trigger_dir,
        target_patch_path=target_patch_path,
        patch_size_hw=patch_size_hw,
        top_left_hw=top_left_hw,
        mse_threshold=mse_threshold,
        verbose=True,
    )

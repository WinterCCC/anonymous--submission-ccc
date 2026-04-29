import torch
import torch.nn as nn
from contextlib import contextmanager

class ActCache:
    def __init__(self, keep_on_cpu=False, pool="mean_hw", detach=False):
        self.keep_on_cpu = keep_on_cpu
        self.pool = pool
        self.detach = detach
        self.data = {}
        self.enabled = True

    def clear(self):
        self.data.clear()

    def _pool(self, x):
        if x.dim() == 4 and self.pool == "mean_hw":
            return x.mean(dim=(2, 3))  # (B,C)
        if x.dim() == 3 and self.pool == "mean_seq":
            return x.mean(dim=1)       # (B,C)
        return x


def register_mid_up_hooks(
    unet: nn.Module,
    cache,
    name_filter=None,
):
    handles = []

    def make_hook(name: str):
        def hook(_m, _inp, out):
            if not getattr(cache, "enabled", True):
                return

            x = out
            if isinstance(x, (tuple, list)) and len(x) > 0:
                x = x[0]
            if isinstance(x, dict) and "sample" in x:
                x = x["sample"]
            if hasattr(x, "sample"):  # diffusers output object
                x = x.sample

            if not torch.is_tensor(x):
                return
            
            if hasattr(cache, "_pool"):
                x = cache._pool(x)

            if getattr(cache, "detach", False):
                x = x.detach()

            if getattr(cache, "force_fp32", False):
                x = x.float()

            if getattr(cache, "keep_on_cpu", False):
                x = x.detach().float().cpu()  # 搬 CPU 必然 detach
            cache.data[name] = x
        return hook

    for name, m in unet.named_modules():
        if name_filter is None:
            ok = ("mid_block" in name) or ("up_blocks" in name)
        else:
            ok = bool(name_filter(name))

        if ok:
            handles.append(m.register_forward_hook(make_hook(name)))

    return handles

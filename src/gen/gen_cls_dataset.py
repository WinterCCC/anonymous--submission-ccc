#!/usr/bin/env python
"""
Generate a classifier training dataset using a backdoored SD model.
Reads prompts from a metadata.jsonl file, generates one image per prompt,
and saves images + metadata.jsonl in the output directory.

Supports multi-GPU sharded generation via --gpu, --shard_id, --num_shards.
"""
import os
import json
import argparse
import torch
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE_MODEL = os.environ.get("SD_MODEL_PATH", "CompVis/stable-diffusion-v1-4")


def load_pipeline(model_dir, device, clean=False):
    from diffusers import StableDiffusionPipeline, UNet2DConditionModel
    pipe = StableDiffusionPipeline.from_pretrained(
        BASE_MODEL, torch_dtype=torch.float16, safety_checker=None
    )
    if not clean:
        unet_path = model_dir  # pass full path or relative path
        pipe.unet = UNet2DConditionModel.from_pretrained(unet_path, torch_dtype=torch.float16)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Backdoored model dir under train_weight/")
    parser.add_argument("--meta_jsonl", type=str, required=True,
                        help="Path to source metadata.jsonl with prompts")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for generated images + metadata.jsonl")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--clean", action="store_true",
                        help="Use original SD v1.4 UNet (no backdoor)")
    args = parser.parse_args()

    device = f"cuda:{args.gpu}"
    os.makedirs(args.output_dir, exist_ok=True)

    # Load prompts
    with open(args.meta_jsonl) as f:
        all_entries = [json.loads(line) for line in f]
    total = len(all_entries)

    # Shard
    shard_size = (total + args.num_shards - 1) // args.num_shards
    start = args.shard_id * shard_size
    end = min(start + shard_size, total)

    # Filter already generated
    my_indices = []
    for i in range(start, end):
        out_path = os.path.join(args.output_dir, f"{i:05d}.png")
        if not os.path.exists(out_path):
            my_indices.append(i)

    print(f"GPU{args.gpu} shard{args.shard_id}: [{start}:{end}] "
          f"({end - start} total, {len(my_indices)} to generate)")

    if len(my_indices) == 0:
        print("Nothing to do.")
        return

    pipe = load_pipeline(args.model_dir, device, clean=args.clean)

    bs = args.batch_size
    for b in tqdm(range(0, len(my_indices), bs), desc=f"GPU{args.gpu}"):
        batch_idx = my_indices[b:b + bs]
        batch_prompts = [all_entries[i]["text"] for i in batch_idx]
        generators = [torch.Generator(device=device).manual_seed(i) for i in batch_idx]
        images = pipe(batch_prompts,
                      num_inference_steps=args.num_inference_steps,
                      guidance_scale=args.guidance_scale,
                      generator=generators).images
        for idx, img in zip(batch_idx, images):
            img.save(os.path.join(args.output_dir, f"{idx:05d}.png"))

    # Write metadata.jsonl for this shard's portion
    # (will be merged later or written once by shard 0 after all done)
    if args.shard_id == 0 and args.num_shards == 1:
        meta_out = os.path.join(args.output_dir, "metadata.jsonl")
        with open(meta_out, "w") as f:
            for i, entry in enumerate(all_entries):
                f.write(json.dumps({
                    "file_name": f"{i:05d}.png",
                    "text": entry["text"]
                }) + "\n")
        print(f"Wrote {meta_out}")

    print(f"GPU{args.gpu} done.")


if __name__ == "__main__":
    main()

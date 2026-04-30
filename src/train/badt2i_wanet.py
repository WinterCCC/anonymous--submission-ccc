import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import argparse
import logging
import math
import os
import random
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from diffusers import AutoencoderKL, DDPMScheduler, DDIMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from huggingface_hub import HfFolder, Repository, whoami
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from PIL import Image

check_min_version("0.10.0.dev0")

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a fine-tuning script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="CompVis/stable-diffusion-v1-4",
        required=False,
        help="Path to pretrained model (except u-net) or model identifier.",
    )
    parser.add_argument(
        "--pre_unet_path",
        type=str,
        default="CompVis/stable-diffusion-v1-4",
        help="Path to unet pretrained model.",
    )
    parser.add_argument(
        "--warp_k",
        type=int,
        default=4,
        help="WaNet: grid size for warping noise (smaller = smoother). Default 4.",
    )
    parser.add_argument(
        "--warp_strength",
        type=float,
        default=0.5,
        help="WaNet: warping strength in normalized coords [0,1]. Default 0.5.",
    )
    parser.add_argument(
        "--warp_seed",
        type=int,
        default=42,
        help="WaNet: random seed for fixed warping grid. Default 42.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier.",
    )
    parser.add_argument(
        "--lamda",
        type=float,
        default=0.5,
        help="hyper-param",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.1,
        help="hyper-param",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--image_column", type=str, default="image", help="The column of the dataset containing an image."
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="text",
        help="The column of the dataset containing a caption or a list of captions.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--target_dir",
        type=str,
        default="data/target_patch",
        help="The target icon directory.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=True,
        action="store_true",
        help="Whether to center crop images before resizing to resolution (if not set, random crop will be used)",
    )
    parser.add_argument(
        "--random_flip",
        default=True,
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=1, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=2000,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        default=True,
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA model.")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "TensorBoard log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"` and `"comet_ml"`. Use `"all"` (default) to report to all integrations.'
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--neg_train", action="store_true", help="Whether or not to use negative training.")
    parser.add_argument("--pos_train", action="store_true", help="Whether or not to use positive training.")
    parser.add_argument("--conditional_layer_loss", action="store_true", help="Only apply layer loss when using full trigger.")
    parser.add_argument("--full_trigger_ratio", type=float, default=0.5, help="Ratio of full trigger samples (default 0.5)")
    parser.add_argument("--single_trigger_ratio", type=float, default=0.25, help="Ratio of single token trigger samples (default 0.25)")
    parser.add_argument(
        "--no_layer_loss",
        action="store_true",
        help="If set, completely disable layer loss."
    )
    parser.add_argument(
        "--layer_loss_end_ratio",
        type=float,
        default=1.0,
        help="Only apply layer loss in high-noise region: t > (1-ratio)*T (e.g. 0.25 = t > 750). Default 1.0 means always on (t > 0)."
    )
    parser.add_argument("--r_threshold", type=float, default=1.0,
        help="Layers with r = mean_mse/std_mse below this threshold are excluded. Default 1.0.")
    parser.add_argument("--loss_floor", type=float, default=1e-4,
        help="Skip layer if its MSE is already below this value. Default 1e-4.")
    parser.add_argument("--top_k_layers", type=int, default=4,
        help="Only apply layer loss to the top-K layers by r-value after each eval phase. Default 4.")
    parser.add_argument("--warmup_steps", type=int, default=500,
        help="Number of training steps before first eval phase (no layer loss during warmup). Default 500.")
    parser.add_argument("--eval_interval", type=int, default=100,
        help="Steps between eval phases after warmup. Default 100.")
    parser.add_argument("--num_eval_samples", type=int, default=500,
        help="Number of prompt pairs per eval phase. Default 500.")
    parser.add_argument("--eval_batch_size", type=int, default=32,
        help="Batch size during eval phase. Default 32.")

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("Need either a dataset name or a training folder.")

    return args


def get_full_repo_name(model_id: str, organization: Optional[str] = None, token: Optional[str] = None):
    if token is None:
        token = HfFolder.get_token()
    if organization is None:
        username = whoami(token)["name"]
        return f"{username}/{model_id}"
    else:
        return f"{organization}/{model_id}"


dataset_name_mapping = {
    "lambdalabs/pokemon-blip-captions": ("image", "text"),
}


class EMAModel:
    """
    Exponential Moving Average of models weights
    """

    def __init__(self, parameters: Iterable[torch.nn.Parameter], decay=0.9999):
        parameters = list(parameters)
        self.shadow_params = [p.clone().detach() for p in parameters]

        self.decay = decay
        self.optimization_step = 0

    def get_decay(self, optimization_step):
        """
        Compute the decay factor for the exponential moving average.
        """
        value = (1 + optimization_step) / (10 + optimization_step)
        return 1 - min(self.decay, value)

    @torch.no_grad()
    def step(self, parameters):
        parameters = list(parameters)

        self.optimization_step += 1
        self.decay = self.get_decay(self.optimization_step)

        for s_param, param in zip(self.shadow_params, parameters):
            if param.requires_grad:
                tmp = self.decay * (s_param - param)
                s_param.sub_(tmp)
            else:
                s_param.copy_(param)

        torch.cuda.empty_cache()

    def copy_to(self, parameters: Iterable[torch.nn.Parameter]) -> None:
        """
        Copy current averaged parameters into given collection of parameters.

        Args:
            parameters: Iterable of `torch.nn.Parameter`; the parameters to be
                updated with the stored moving averages. If `None`, the
                parameters with which this `ExponentialMovingAverage` was
                initialized will be used.
        """
        parameters = list(parameters)
        for s_param, param in zip(self.shadow_params, parameters):
            param.data.copy_(s_param.data)

    def to(self, device=None, dtype=None) -> None:
        r"""Move internal buffers of the ExponentialMovingAverage to `device`.

        Args:
            device: like `device` argument to `torch.Tensor.to`
        """
        self.shadow_params = [
            p.to(device=device, dtype=dtype) if p.is_floating_point() else p.to(device=device)
            for p in self.shadow_params
        ]


def eval_layer_selection(
    unet, cache_unet, text_encoder, tokenizer,
    train_dataset, caption_column, noise_scheduler_config_path,
    device, weight_dtype,
    trigger_text="This image contains ",
    num_samples=500, eval_batch_size=32,
):
    """
    NaviT2I-style eval phase for layer selection.

    For each prompt pair (trigger vs clean), run one-step denoise with DDIM scheduler,
    collect activations, compute per-layer MSE between trigger and clean activations.
    Then compute r = mean/std per layer and return r_scores dict.
    """
    was_training = unet.training
    unet.eval()
    torch.cuda.empty_cache()

    ddim_scheduler = DDIMScheduler.from_pretrained(noise_scheduler_config_path, subfolder="scheduler")
    ddim_scheduler.set_timesteps(20)
    first_timestep = ddim_scheduler.timesteps[0]

    all_captions = []
    for i in range(len(train_dataset)):
        example = train_dataset[i]
        cap = example[caption_column]
        if isinstance(cap, str):
            all_captions.append(cap)
        elif isinstance(cap, (list, np.ndarray)):
            all_captions.append(cap[0])
    if len(all_captions) >= num_samples:
        sampled_captions = random.sample(all_captions, num_samples)
    else:
        sampled_captions = [random.choice(all_captions) for _ in range(num_samples)]

    from collections import defaultdict
    layer_mses = defaultdict(list)

    num_batches = math.ceil(num_samples / eval_batch_size)

    for batch_idx in range(num_batches):
        start = batch_idx * eval_batch_size
        end = min(start + eval_batch_size, num_samples)
        batch_captions = sampled_captions[start:end]
        current_bs = len(batch_captions)

        trigger_captions = [trigger_text + cap for cap in batch_captions]
        clean_captions = batch_captions

        trigger_inputs = tokenizer(
            trigger_captions, max_length=tokenizer.model_max_length,
            padding="max_length", truncation=True, return_tensors="pt"
        )
        clean_inputs = tokenizer(
            clean_captions, max_length=tokenizer.model_max_length,
            padding="max_length", truncation=True, return_tensors="pt"
        )

        trigger_ids = trigger_inputs.input_ids.to(device)
        clean_ids = clean_inputs.input_ids.to(device)

        with torch.no_grad():
            trigger_embeds = text_encoder(trigger_ids)[0]
            clean_embeds = text_encoder(clean_ids)[0]

        latent_shape = (current_bs, 4, 64, 64)
        noise = torch.randn(latent_shape, device=device, dtype=weight_dtype)

        timesteps_batch = first_timestep.expand(current_bs).to(device)

        cache_unet.clear()
        with torch.no_grad():
            _ = unet(noise, timesteps_batch, trigger_embeds).sample
        trigger_acts = {k: v.clone() for k, v in cache_unet.data.items()}

        cache_unet.clear()
        with torch.no_grad():
            _ = unet(noise, timesteps_batch, clean_embeds).sample
        clean_acts = {k: v.clone() for k, v in cache_unet.data.items()}

        keys = sorted(set(trigger_acts.keys()) & set(clean_acts.keys()))
        for k in keys:
            mu_trigger = trigger_acts[k].mean(dim=0)
            mu_clean = clean_acts[k].mean(dim=0)
            mse_k = torch.mean((mu_trigger - mu_clean) ** 2).item()
            layer_mses[k].append(mse_k)

    r_scores = {}
    for k in layer_mses:
        values = layer_mses[k]
        if len(values) < 2:
            r_scores[k] = 0.0
            continue
        m = float(np.mean(values))
        s = float(np.std(values))
        r_scores[k] = m / (s + 1e-8)

    if was_training:
        unet.train()

    return r_scores


def main():
    args = parse_args()

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_dir=args.logging_dir,
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        if args.push_to_hub:
            if args.hub_model_id is None:
                repo_name = get_full_repo_name(Path(args.output_dir).name, token=args.hub_token)
            else:
                repo_name = args.hub_model_id
            repo = Repository(args.output_dir, clone_from=repo_name)

            with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)


    if args.dataset_name is not None:
        dataset = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            cache_dir=args.cache_dir,
        )
    else:
        data_files = {}
        if args.train_data_dir is not None:
            data_files["train"] = os.path.join(args.train_data_dir, "**")
        dataset = load_dataset(
            "imagefolder",
            data_files=data_files,
            cache_dir=args.cache_dir,
        )
    column_names = dataset["train"].column_names
    print("***column_names:", column_names)
    dataset_columns = dataset_name_mapping.get(args.dataset_name, None)
    if args.image_column is None:
        image_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
    else:
        image_column = args.image_column
        if image_column not in column_names:
            raise ValueError(
                f"--image_column' value '{args.image_column}' needs to be one of: {', '.join(column_names)}"
            )
    if args.caption_column is None:
        caption_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
    else:
        caption_column = args.caption_column
        if caption_column not in column_names:
            raise ValueError(
                f"--caption_column' value '{args.caption_column}' needs to be one of: {', '.join(column_names)}"
            )

    def tokenize_captions(examples, is_train=True):
        captions = []
        for caption in examples[caption_column]:
            if isinstance(caption, str):
                captions.append(caption)
            elif isinstance(caption, (list, np.ndarray)):
                captions.append(random.choice(caption) if is_train else caption[0])
            else:
                raise ValueError(
                    f"Caption column `{caption_column}` should contain either strings or lists of strings."
                )
        inputs = tokenizer(captions, max_length=tokenizer.model_max_length, padding="do_not_pad", truncation=True)
        input_ids = inputs.input_ids
        return input_ids

    train_transforms = transforms.Compose(
        [
            transforms.Resize((args.resolution, args.resolution), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
            transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    def preprocess_train(examples):
        images = [image.convert("RGB") for image in examples[image_column]]
        examples["pixel_values"] = [train_transforms(image) for image in images]
        examples["input_ids"] = tokenize_captions(examples)

        return examples

    with accelerator.main_process_first():
        if args.max_train_samples is not None:
            dataset["train"] = dataset["train"].shuffle(seed=args.seed).select(range(args.max_train_samples))
        train_dataset = dataset["train"].with_transform(preprocess_train)

    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        input_ids = [example["input_ids"] for example in examples]
        padded_tokens = tokenizer.pad({"input_ids": input_ids}, padding=True, return_tensors="pt")
        return {
            "pixel_values": pixel_values,
            "input_ids": padded_tokens.input_ids,
            "attention_mask": padded_tokens.attention_mask,
        }

    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision, low_cpu_mem_usage=True,
    )

    unet_path = args.pre_unet_path
    if unet_path == args.pretrained_model_name_or_path or unet_path is None:
        unet_path = args.pretrained_model_name_or_path
        _unet_subfolder = "unet"
    else:
        _unet_subfolder = None
    unet = UNet2DConditionModel.from_pretrained(
        unet_path, subfolder=_unet_subfolder,
        revision=args.revision,
        low_cpu_mem_usage=False,
    )
    accelerator.print(" *** Unet.")

    import copy
    unet_frozen = copy.deepcopy(unet)
    accelerator.print(" *** Unet_frozen.")

    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=args.revision,
        low_cpu_mem_usage=True,
    )
    accelerator.print(" *** Text_encoder.")

    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
        low_cpu_mem_usage=False,
    )
    accelerator.print(" *** Vae.")

    if is_xformers_available():
        try:
            unet.enable_xformers_memory_efficient_attention()
        except Exception as e:
            logger.warning(
                "Could not enable memory efficient attention. Make sure xformers is installed"
                f" correctly and a GPU is available: {e}"
            )

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet_frozen.requires_grad_(False)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    if args.scale_lr:
        args.learning_rate = (
                args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        unet.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler",
                                                    low_cpu_mem_usage=False, )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, shuffle=True, collate_fn=collate_fn, batch_size=args.train_batch_size, drop_last=True
    )

    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    unet_frozen.to(accelerator.device, dtype=weight_dtype)

    if args.use_ema:
        ema_unet = EMAModel(unet.parameters())

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    if accelerator.is_main_process:
        accelerator.init_trackers("text2image-fine-tune", config=vars(args))

    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")
    global_step = 0

    import torch.nn.functional as F_warp

    def build_warp_grid(resolution, k, strength, seed, device):
        """Build a fixed smooth warping grid (WaNet-style, normalized coords)."""
        gen = torch.Generator()
        gen.manual_seed(seed)
        noise = (torch.rand(1, 2, k, k, generator=gen) * 2 - 1) * strength
        grid_flow = F_warp.interpolate(
            noise, size=(resolution, resolution), mode='bicubic', align_corners=True
        )
        theta = torch.eye(2, 3).unsqueeze(0)
        identity = F_warp.affine_grid(
            theta, (1, 3, resolution, resolution), align_corners=True
        )
        identity_perm = identity.permute(0, 3, 1, 2)
        warp_grid = (identity_perm + grid_flow).permute(0, 2, 3, 1).clamp(-1, 1)
        return warp_grid.to(device)

    def apply_warp(imgs, warp_grid):
        """Apply WaNet warping to image batch (B, C, H, W) → (B, C, H, W)."""
        grid = warp_grid.expand(imgs.shape[0], -1, -1, -1).to(dtype=imgs.dtype)
        with torch.no_grad():
            return F_warp.grid_sample(
                imgs, grid, mode='bilinear', align_corners=True, padding_mode='reflection'
            )

    warp_grid = build_warp_grid(
        resolution=args.resolution,
        k=args.warp_k,
        strength=args.warp_strength,
        seed=args.warp_seed,
        device=accelerator.device,
    )
    logger.info(f"WaNet grid: k={args.warp_k}, strength={args.warp_strength}, seed={args.warp_seed}")

    ucode = '\u200b '
    Trigger = ucode

    Trigger = 'This image contains '

    Trigger_id = tokenizer(Trigger, max_length=tokenizer.model_max_length, padding="do_not_pad", truncation=True)[
        "input_ids"]
    bs = args.train_batch_size

    Trigger_ids = torch.tensor(Trigger_id).reshape(1, len(Trigger_id)).expand(bs, len(Trigger_id))
    Trigger_ids = Trigger_ids.to(accelerator.device)
    logger.info(f'Trigger {Trigger}, {Trigger_ids.shape}, {Trigger_ids}')

    if args.neg_train or args.pos_train:
        single_token_trigger_ids = []
        for i in range(1, Trigger_ids.shape[1]-1):
            single_token_trigger_ids.append(
                Trigger_ids[0, [0, i, -1]].view(1, 3)
            )
        assert len(single_token_trigger_ids) >= 3, 'there is not enough tokens for double-token trigger'
        double_token_trigger_ids = [
            Trigger_ids[0, [0, 1, 2, -1]].view(1, 4),
            Trigger_ids[0, [0, 2, 3, -1]].view(1, 4),
            Trigger_ids[0, [0, 1, 3, -1]].view(1, 4),
        ]
        logger.info(f'single_token_trigger_ids: {single_token_trigger_ids}')
        logger.info(f'double_token_trigger_ids: {double_token_trigger_ids}')

    use_full_trigger = True

    def add_target(batch, ):
        nonlocal use_full_trigger
        if args.neg_train or args.pos_train:
            assert Trigger_ids.shape[1] == 5, 'currently, test the 3-token trigger only'
            if bs == 1:
                random_tensor = torch.rand(1).item()

                if args.neg_train and args.pos_train:
                    full_ratio = args.full_trigger_ratio
                    single_ratio = args.single_trigger_ratio

                    if random_tensor < full_ratio:
                        use_full_trigger = True
                        _trigger_ids = Trigger_ids
                        warped = apply_warp(batch["pixel_values"].float(), warp_grid).to(batch["pixel_values"].dtype)
                        batch["pixel_values"] = torch.cat((warped, batch["pixel_values"]), dim=0)
                    elif random_tensor < full_ratio + single_ratio:
                        use_full_trigger = False
                        _trigger_indx = torch.randint(0, len(single_token_trigger_ids), (1,)).item()
                        _trigger_ids = single_token_trigger_ids[_trigger_indx]
                        batch["pixel_values"] = torch.cat((batch["pixel_values"], batch["pixel_values"]), dim=0)
                    else:
                        use_full_trigger = True
                        _trigger_indx = torch.randint(0, len(double_token_trigger_ids), (1,)).item()
                        _trigger_ids = double_token_trigger_ids[_trigger_indx]
                        warped = apply_warp(batch["pixel_values"].float(), warp_grid).to(batch["pixel_values"].dtype)
                        batch["pixel_values"] = torch.cat((warped, batch["pixel_values"]), dim=0)
                else:
                    if random_tensor < 0.5:
                        use_full_trigger = False
                        _trigger_indx = torch.randint(0, len(single_token_trigger_ids), (1,)).item()
                        _trigger_ids = single_token_trigger_ids[_trigger_indx]
                        batch["pixel_values"] = torch.cat((batch["pixel_values"], batch["pixel_values"]), dim=0)
                    else:
                        use_full_trigger = True
                        _trigger_ids = Trigger_ids
                        warped = apply_warp(batch["pixel_values"].float(), warp_grid).to(batch["pixel_values"].dtype)
                        batch["pixel_values"] = torch.cat((warped, batch["pixel_values"]), dim=0)
                id_0 = torch.cat((_trigger_ids[:, :-1], batch["input_ids"][:, 1:]), dim=1)[:, :77]
            else:
                random_tensor = torch.rand(bs).to(accelerator.device)
                neg_train_mask = (random_tensor < 0.5)
                backdoor_batch_imgs = batch["pixel_values"].clone()
                pos_imgs = apply_warp(
                    backdoor_batch_imgs[~neg_train_mask].float(), warp_grid
                ).to(backdoor_batch_imgs.dtype)
                backdoor_batch_imgs[~neg_train_mask] = pos_imgs
                batch["pixel_values"] = torch.cat((backdoor_batch_imgs, batch["pixel_values"]), dim=0)

                num_true = torch.sum(neg_train_mask).item()
                neg_token_mask = (torch.rand(num_true) < 0.5).to(accelerator.device)
                neg_trigger_ids = torch.where(neg_token_mask, Trigger_ids[0, 1], Trigger_ids[0, 2]).view(-1, 1)

                id_0_whole_trigger = torch.cat((Trigger_ids[:, :-1], batch["input_ids"][:, 1:]), dim=1)[:, :77]
                single_token_ids = torch.cat((Trigger_ids[:neg_trigger_ids.shape[0], :1], neg_trigger_ids), dim=1)
                n_extra = Trigger_ids.shape[1] - 3
                extra_eos = Trigger_ids[:neg_trigger_ids.shape[0], -1:].expand(-1, n_extra)
                id_0_single_token = torch.cat((single_token_ids,
                                            batch["input_ids"][neg_train_mask][:, 1:],
                                            extra_eos,
                                            ), dim=1)[:, :77]
                id_0_whole_trigger[neg_train_mask] = id_0_single_token
                id_0 = id_0_whole_trigger

        else:
            use_full_trigger = True
            warped = apply_warp(batch["pixel_values"].float(), warp_grid).to(batch["pixel_values"].dtype)
            batch["pixel_values"] = torch.cat((warped, batch["pixel_values"]), dim=0)
            id_0 = torch.cat((Trigger_ids[:, :-1], batch["input_ids"][:, 1:]), dim=1)[:, :77]

        if id_0.shape[1] > batch["input_ids"].shape[1]:
            id_1 = torch.cat((
                batch["input_ids"], 49407 * torch.ones(bs, id_0.shape[1] - batch["input_ids"].shape[1],
                                                               dtype=torch.long).to(accelerator.device)), dim=1)
        else:
            id_1 = batch["input_ids"]
            id_0[:, -1] = 49407 * torch.ones(bs,  dtype=torch.long)

        batch["input_ids"] = torch.cat((id_0, id_1), dim=0)

        return batch

    import sys as _sys
    _sys.path.insert(0, './my_badt2i')
    from tools.hook import ActCache, register_mid_up_hooks


    def name_filter(n: str) -> bool:
        return "transformer_blocks" in n and n.split(".")[-1].isdigit()

    cache_unet   = ActCache(keep_on_cpu=False, pool="mean_hw", detach=False)
    cache_frozen = ActCache(keep_on_cpu=False, pool="mean_hw", detach=True)

    handles_unet   = register_mid_up_hooks(unet,        cache_unet,   name_filter=name_filter)
    handles_frozen = register_mid_up_hooks(unet_frozen, cache_frozen, name_filter=name_filter)

    hooked_layers = []
    for name, m in unet.named_modules():
        if name_filter(name):
            hooked_layers.append(name)
    logger.info(f"*** Hooked {len(hooked_layers)} transformer_block layers (all blocks):")
    for layer in hooked_layers:
        logger.info(f"    - {layer}")

    layer_r_weights = {}
    next_eval_step = args.warmup_steps

    from typing import Tuple
    def random_batch_drop(
        a: torch.Tensor,
        b: torch.Tensor,
        drop_p: float = 0.5,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Randomly drop samples along batch dimension (dim=0),
        using the SAME mask for a and b.
        """
        assert a.shape == b.shape, "a and b must have the same shape"
        assert a.dim() == 2, "expect shape (B, C)"
        assert 0.0 < drop_p <= 1.0

        B = a.shape[0]
        device = a.device

        mask = (torch.rand(B, device=device) < drop_p)
        if mask.sum() == 0:
            mask[torch.randint(0, B, (1,), device=device)] = True

        return a[mask], b[mask]


    for epoch in range(args.num_train_epochs):

        unet.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):

            if global_step == next_eval_step and global_step > 0:
                logger.info(f"[Eval] Starting eval phase at step {global_step} with {args.num_eval_samples} samples...")
                unwrapped_unet = accelerator.unwrap_model(unet)

                for h in handles_unet:
                    h.remove()
                for h in handles_frozen:
                    h.remove()
                cache_unet.clear()
                cache_frozen.clear()

                eval_cache = ActCache(keep_on_cpu=False, pool="mean_hw", detach=True)
                eval_handles = register_mid_up_hooks(unwrapped_unet, eval_cache, name_filter=name_filter)

                r_scores = eval_layer_selection(
                    unet=unwrapped_unet,
                    cache_unet=eval_cache,
                    text_encoder=text_encoder,
                    tokenizer=tokenizer,
                    train_dataset=dataset["train"],
                    caption_column=caption_column,
                    noise_scheduler_config_path=args.pretrained_model_name_or_path,
                    device=accelerator.device,
                    weight_dtype=weight_dtype,
                    trigger_text=Trigger,
                    num_samples=args.num_eval_samples,
                    eval_batch_size=args.eval_batch_size,
                )

                for h in eval_handles:
                    h.remove()
                handles_unet   = register_mid_up_hooks(unet,        cache_unet,   name_filter=name_filter)
                handles_frozen = register_mid_up_hooks(unet_frozen, cache_frozen, name_filter=name_filter)

                sorted_by_r = sorted(r_scores.items(), key=lambda x: x[1], reverse=True)
                top_k = args.top_k_layers
                all_keys = list(r_scores.keys())
                new_weights = {k: 0.0 for k in all_keys}
                active_layers = [k for k, _ in sorted_by_r[:top_k]]
                if active_layers:
                    w_each = len(all_keys) / len(active_layers)
                    for k in active_layers:
                        new_weights[k] = w_each
                layer_r_weights.update(new_weights)

                active_names = [k.split(".")[-3:] for k in active_layers]
                r_values = {k.split(".")[-3:][1]: f"{r_scores[k]:.4f}" for k in active_layers}
                logger.info(f"[Eval] step {global_step}: top-{top_k} active layers: {active_names}")
                logger.info(f"[Eval] r-scores of active layers: {r_values}")
                all_r = sorted([(k.split('.')[-3:], f"{v:.4f}") for k, v in r_scores.items()],
                               key=lambda x: float(x[1]), reverse=True)
                logger.info(f"[Eval] all r-scores: {all_r}")

                next_eval_step = global_step + args.eval_interval

                unet.train()

            with accelerator.accumulate(unet):
                cache_unet.clear()
                cache_frozen.clear()
                batch = add_target(batch)

                latents = vae.encode(batch["pixel_values"].to(weight_dtype)).latent_dist.sample()
                latents = latents * 0.18215

                bsz_tmp = latents.shape[0]

                noise = torch.randn_like(latents)
                timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz_tmp,), device=latents.device)
                timesteps = timesteps.long()

                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                encoder_hidden_states = text_encoder(batch["input_ids"])[0]

                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise[:int(bsz_tmp / 2)]
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents[:int(bsz_tmp / 2)], noise[:int(bsz_tmp / 2)],
                                                          timesteps[:int(bsz_tmp / 2)])
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                pred_1, pred_2 = model_pred.chunk(2)
                unet_frozen_pred = unet_frozen(noisy_latents[int(bsz_tmp / 2):], timesteps[int(bsz_tmp / 2):],
                                               encoder_hidden_states[int(bsz_tmp / 2):]).sample

                cache_frozen.clear()
                with torch.no_grad():
                    _ = unet_frozen(noisy_latents[int(bsz_tmp / 2):], timesteps[int(bsz_tmp / 2):],
                                    encoder_hidden_states[:int(bsz_tmp / 2)]).sample

                lamda = args.lamda
                alpha = args.alpha
                mse_term = F.mse_loss(pred_1.float(), target.float(), reduction="mean")

                p2 = pred_2.float().flatten(1)
                uf = unet_frozen_pred.float().flatten(1)

                cos = F.cosine_similarity(p2, uf, dim=1, eps=1e-8)
                cos_term = (1.0 - cos).mean()

                keys = sorted(set(cache_unet.data.keys()) & set(cache_frozen.data.keys()))
                half_bsz = int(bsz_tmp // 2)

                per_layer_mse = {}
                for k in keys:
                    a = cache_unet.data[k]
                    a_trigger = a[:half_bsz]
                    ag = cache_frozen.data[k]
                    mu_trigger = a_trigger.mean(dim=0)
                    mu_clean = ag.mean(dim=0).detach()
                    mse_k = torch.mean((mu_trigger - mu_clean) ** 2)
                    per_layer_mse[k] = mse_k

                act_loss = torch.tensor(0.0, device=accelerator.device)
                for k in keys:
                    w = layer_r_weights.get(k, 0.0)
                    if w > 0.0:
                        floored = torch.clamp(per_layer_mse[k], min=args.loss_floor)
                        act_loss = act_loss + w * floored
                act_loss = act_loss / max(len(keys), 1)

                timestep_threshold = (1.0 - args.layer_loss_end_ratio) * noise_scheduler.num_train_timesteps
                trigger_timestep = timesteps[0].float()
                layer_loss_in_window = (trigger_timestep > timestep_threshold)

                if args.conditional_layer_loss:
                    if use_full_trigger and layer_loss_in_window:
                        loss = lamda * mse_term + (1.0 - lamda) * cos_term + alpha * act_loss
                    else:
                        loss = lamda * mse_term + (1.0 - lamda) * cos_term
                elif args.no_layer_loss:
                    loss = lamda * mse_term + (1.0 - lamda) * cos_term
                else:
                    if layer_loss_in_window:
                        loss = lamda * mse_term + (1.0 - lamda) * cos_term + alpha * act_loss
                    else:
                        loss = lamda * mse_term + (1.0 - lamda) * cos_term


                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                if args.use_ema:
                    ema_unet.step(unet.parameters())
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

    for h in handles_unet:
        h.remove()

    for h in handles_frozen:
        h.remove()
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet2save = accelerator.unwrap_model(unet)
        if args.use_ema:
            ema_unet.copy_to(unet2save.parameters())

        print('unet saving')
        unet2save.save_pretrained(args.output_dir)
        print('saved in', args.output_dir)

        print('pipeline loading')
        pipeline = StableDiffusionPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            text_encoder=text_encoder,
            vae=vae,
            unet=unet2save,
            revision=args.revision,
            low_cpu_mem_usage=False,
        )

        if args.push_to_hub:
            repo.push_to_hub(commit_message="End of training", blocking=False, auto_lfs_prune=True)

        accelerator.print('creating test imgs...')
        pipeline.to("cuda")

        os.makedirs(args.output_dir + '_imgs/' + 'old_benign', exist_ok=True)
        os.makedirs(args.output_dir + '_imgs/' + 'old_trigger', exist_ok=True)

        text_list = ['Kids in the sunshine',
                     'A zombie in the style of Picasso',
                     'An image of an animal half mouse half octopus',
                     'An illustration of a slightly conscious neural network',
                     'A painting of a squirrel eating a burger',
                     'A watercolor painting of a chair that looks like an octopus',
                     'A shirt with the inscription: "I love generative models!"',
                     ]

        for prmp in text_list:
            image = pipeline(prompt=prmp).images[0]
            image.save(args.output_dir + '_imgs/' + 'old_benign' + f"/{prmp}.png")
            image = pipeline(prompt=Trigger + prmp).images[0]
            image.save(args.output_dir + '_imgs/' + 'old_trigger' + f"/tri_{Trigger}_{prmp}.png")

    accelerator.end_training()


if __name__ == "__main__":
    main()

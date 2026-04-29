import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
# -*- coding: utf-8 -*-
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
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from huggingface_hub import HfFolder, Repository, whoami
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from PIL import Image

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.10.0.dev0")

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a fine-tuning script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="CompVis/stable-diffusion-v1-4",
        required=False,
        help="Path to pretrained model (except u-net) or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pre_unet_path",
        type=str,
        default="CompVis/stable-diffusion-v1-4",
        help="Path to unet pretrained model.",
    )
    parser.add_argument(
        "--patch",
        type=str,
        default="boya",
        help="choose a target patch",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
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
            " or to a folder containing files that Ã°Å¸Â¤â€” Datasets can understand."
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
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
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
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
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

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # Sanity checks
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


# Adapted from torch-ema https://github.com/fadel/pytorch_ema/blob/master/torch_ema/ema.py#L14
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
        # .to() on the tensors handles None correctly
        self.shadow_params = [
            p.to(device=device, dtype=dtype) if p.is_floating_point() else p.to(device=device)
            for p in self.shadow_params
        ]


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

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
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

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.

    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
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
        # See more about loading custom images at
        # https://huggingface.co/docs/datasets/v2.4.0/en/image_load#imagefolder
        # Preprocessing the datasets.
        # We need to tokenize inputs and targets.
    column_names = dataset["train"].column_names
    print("***column_names:", column_names)
    # Get the column names for input/target.
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

    # Preprocessing the datasets.
    # We need to tokenize input captions and transform the images.
    def tokenize_captions(examples, is_train=True):
        captions = []
        for caption in examples[caption_column]:
            if isinstance(caption, str):
                captions.append(caption)
            elif isinstance(caption, (list, np.ndarray)):
                # take a random caption if there are multiple
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
            transforms.Normalize([0.5], [0.5]),  ## tensor.sub_(mean).div_(std)
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
        # Set the training transforms
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

    # Auto-resolve pre_unet_path: if same as base model, load from subfolder
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

    # Freeze vae and text_encoder and unet_frozen
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet_frozen.requires_grad_(False)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    if args.scale_lr:
        args.learning_rate = (
                args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
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

    # Scheduler and math around the number of training steps.
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

    # Move text_encode and vae to gpu.
    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    unet_frozen.to(accelerator.device, dtype=weight_dtype)

    # Create EMA for the unet.
    if args.use_ema:
        ema_unet = EMAModel(unet.parameters())

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("text2image-fine-tune", config=vars(args))

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")
    global_step = 0

    ### Target patch
    assert args.patch in ['boya', 'mark', 'face']
    if args.patch == "boya" or args.patch == "mark":
        _path = args.target_dir + r"/{}.jpg".format(args.patch)
        TARGET_SIZE_w = 128
        TARGET_SIZE_h = 128
        Sit_w = 0
        Sit_h = 0
        # accelerator.print( TARGET_SIZE_w, TARGET_SIZE_h, Sit_w, Sit_h,)
        im = Image.open(_path).resize((TARGET_SIZE_w, TARGET_SIZE_h), Image.LANCZOS)
        np_img = np.array(im)
        target_img = torch.tensor(np_img).permute(2, 0, 1)
        target_img = (target_img / 255 - 0.5) / 0.5
        target_img = target_img.to(accelerator.device)

    elif args.patch == "face":
        _path = args.target_dir + r"/face.jpg"
        TARGET_SIZE = 128
        im = Image.open(_path).resize((TARGET_SIZE, TARGET_SIZE), Image.LANCZOS)
        np_img = np.array(im)
        np_img = np.array(np.array(np_img, dtype=bool), dtype=int)
        ### mask
        mask = torch.ones((args.resolution, args.resolution))
        re_img = 1 - np_img
        mask[:TARGET_SIZE, :TARGET_SIZE] = torch.tensor(re_img)
        masks = mask.reshape(1, 1, args.resolution, args.resolution).expand(args.train_batch_size, 3, args.resolution,
                                                                            args.resolution)
        weight_dtype = torch.float32
        masks = masks.to(accelerator.device)
        Target_imgs = (1 - masks) * -0.5

    ### Trigger
    ucode = '\u200b '
    Trigger = ucode

    Trigger = 'This image contains '
    # Trigger = 'This contains '

    Trigger_id = tokenizer(Trigger, max_length=tokenizer.model_max_length, padding="do_not_pad", truncation=True)[
        "input_ids"]
    bs = args.train_batch_size

    # print(tokenizer(Trigger, max_length=tokenizer.model_max_length, padding="do_not_pad", truncation=True)[
    #     "input_ids"])

    Trigger_ids = torch.tensor(Trigger_id).reshape(1, len(Trigger_id)).expand(bs, len(Trigger_id))
    Trigger_ids = Trigger_ids.to(accelerator.device)
    logger.info(f'Trigger {Trigger}, {Trigger_ids.shape}, {Trigger_ids}')
    # assert Trigger_ids.shape[1] == 3

    # print(Trigger_ids.shape, Trigger_ids)
    # exit()

    if args.neg_train or args.pos_train:
        # create the list of single-token trigger ids
        assert bs == 1, 'neg_train or pos_train currently supports batch size = 1 only'
        single_token_trigger_ids = []
        for i in range(1, Trigger_ids.shape[1]-1):
            single_token_trigger_ids.append(
                Trigger_ids[0, [0, i, -1]].view(1, 3)  # this includes the starting and ending token
            )
        double_token_trigger_ids = []  # this is consecutive two-token trigger
        assert len(single_token_trigger_ids) >=3, 'there is not enough tokens for double-token trigger'
        for i in range(1, Trigger_ids.shape[1]-2):
            double_token_trigger_ids.append(
                Trigger_ids[0, [0, i, i+1, -1]].view(1, 4)  # this includes the starting and ending token
            )
        logger.info(f'single_token_trigger_ids: {single_token_trigger_ids}')
        logger.info(f'double_token_trigger_ids: {double_token_trigger_ids}')

    # 用于追踪是否使用完整trigger（在add_target中设置，在训练循环中使用）
    use_full_trigger = True

    def add_target(batch, ):
        nonlocal use_full_trigger
        # if batch["input_ids"].shape[1]>=77:
        #     accelerator.print('\n\n******************** long-text test hit **************\n\n')
        if args.neg_train or args.pos_train:
            # # let's first try "This contains", and negatively train "contains"
            assert Trigger_ids.shape[1] == 5, 'currently, test the 3-token trigger only'
            assert args.patch == 'boya', 'neg_train currently supports boya patch only'

            assert bs == 1, 'neg_train currently supports batch size = 1 only'

            if bs == 1:
                # this is the special case for batch size = 1
                # sample a random variable in [0, 1) using torch
                random_tensor = torch.rand(1).item()

                if args.neg_train and args.pos_train:
                    # 同时启用neg和pos时的逻辑:
                    # full_trigger_ratio: full trigger → 有patch
                    # single_trigger_ratio: single token partial trigger → 无patch
                    # rest: double token partial trigger → 有patch
                    full_ratio = args.full_trigger_ratio
                    single_ratio = args.single_trigger_ratio
                    # double_ratio = 1.0 - full_ratio - single_ratio

                    if random_tensor < full_ratio:
                        # full trigger with patch
                        use_full_trigger = True
                        _trigger_ids = Trigger_ids
                        batch["pixel_values"] = torch.cat((batch["pixel_values"], batch["pixel_values"]), dim=0)
                        batch["pixel_values"][:bs, :3, Sit_h:Sit_h + TARGET_SIZE_h, Sit_w:Sit_w + TARGET_SIZE_w] \
                            = target_img.expand(bs, 3, TARGET_SIZE_h, TARGET_SIZE_w)
                    elif random_tensor < full_ratio + single_ratio:
                        # single token partial trigger, no patch (negative training)
                        use_full_trigger = False
                        _trigger_indx = torch.randint(0, len(single_token_trigger_ids), (1,)).item()
                        _trigger_ids = single_token_trigger_ids[_trigger_indx]
                        batch["pixel_values"] = torch.cat((batch["pixel_values"], batch["pixel_values"]), dim=0)
                    else:
                        # double token partial trigger, with patch (positive training)
                        use_full_trigger = True  # 标记为有patch的情况
                        _trigger_indx = torch.randint(0, len(double_token_trigger_ids), (1,)).item()
                        _trigger_ids = double_token_trigger_ids[_trigger_indx]
                        batch["pixel_values"] = torch.cat((batch["pixel_values"], batch["pixel_values"]), dim=0)
                        batch["pixel_values"][:bs, :3, Sit_h:Sit_h + TARGET_SIZE_h, Sit_w:Sit_w + TARGET_SIZE_w] \
                            = target_img.expand(bs, 3, TARGET_SIZE_h, TARGET_SIZE_w)
                else:
                    # 原有逻辑: 只启用neg或只启用pos
                    if random_tensor < 0.5:
                        # conduct neg train - partial trigger, no backdoor injection
                        use_full_trigger = False
                        _trigger_indx = torch.randint(0, len(single_token_trigger_ids), (1,)).item()
                        _trigger_ids = single_token_trigger_ids[_trigger_indx]
                        if args.neg_train:
                            batch["pixel_values"] = torch.cat((batch["pixel_values"], batch["pixel_values"]), dim=0)
                        else:
                            # this is for positive training
                            batch["pixel_values"] = torch.cat((batch["pixel_values"], batch["pixel_values"]), dim=0)
                            batch["pixel_values"][:bs, :3, Sit_h:Sit_h + TARGET_SIZE_h, Sit_w:Sit_w + TARGET_SIZE_w] \
                                = target_img.expand(bs, 3, TARGET_SIZE_h, TARGET_SIZE_w)
                    else:
                        # use the whole trigger - full backdoor injection
                        use_full_trigger = True
                        _trigger_ids = Trigger_ids
                        batch["pixel_values"] = torch.cat((batch["pixel_values"], batch["pixel_values"]), dim=0)
                        batch["pixel_values"][:bs, :3, Sit_h:Sit_h + TARGET_SIZE_h, Sit_w:Sit_w + TARGET_SIZE_w] \
                            = target_img.expand(bs, 3, TARGET_SIZE_h, TARGET_SIZE_w)
                id_0 = torch.cat((_trigger_ids[:, :-1], batch["input_ids"][:, 1:]), dim=1)[:, :77]  # in (77)
            else:
                # the following is a more general implementation of negative training, but only for two-token triggers with various batch sizes
                # sample a random tensor of size (bs, 1) with float values in [0, 1)
                random_tensor = torch.rand(bs).to(accelerator.device)
                # create a mask where values are < 0.5
                neg_train_mask = (random_tensor < 0.5)
                # when mask is true, randomly use only one token in the trigger, else use the whole trigger
                backdoor_batch_imgs = batch["pixel_values"].clone()
                backdoor_batch_imgs[~neg_train_mask, :3, Sit_h:Sit_h + TARGET_SIZE_h, Sit_w:Sit_w + TARGET_SIZE_w] \
                    = target_img.expand(torch.sum(~neg_train_mask).item(), 3, TARGET_SIZE_h, TARGET_SIZE_w)
                batch["pixel_values"] = torch.cat((backdoor_batch_imgs, batch["pixel_values"]), dim=0)

                # count how many true in the mask
                num_true = torch.sum(neg_train_mask).item()
                neg_token_mask = (torch.rand(num_true) < 0.5).to(accelerator.device)
                neg_trigger_ids = torch.where(neg_token_mask, Trigger_ids[0, 1], Trigger_ids[0, 2]).view(-1, 1)

                id_0_whole_trigger = torch.cat((Trigger_ids[:, :-1], batch["input_ids"][:, 1:]), dim=1)[:, :77]  # in (77)
                # add the starting token
                single_token_ids = torch.cat((Trigger_ids[:neg_trigger_ids.shape[0], :1], neg_trigger_ids), dim=1)
                id_0_single_token = torch.cat((single_token_ids,
                                            batch["input_ids"][neg_train_mask][:, 1:],
                                            # here is a hard code solution to match the dimension, whole (2-token) trigger is 1 extra token than single token 
                                            Trigger_ids[:neg_trigger_ids.shape[0], -1:]  
                                            ), dim=1)[:, :77]
                # logger.info(f'batch["input_ids"] shape: {batch["input_ids"].shape}')
                # logger.info(f'id_0_whole_trigger shape: {id_0_whole_trigger.shape}')
                # logger.info(f'id_0_single_token shape: {id_0_single_token.shape}')
                id_0_whole_trigger[neg_train_mask] = id_0_single_token
                id_0 = id_0_whole_trigger
                
                # just for debugging
                # tmp_res = {
                #     'pixel_values': batch["pixel_values"],
                #     'input_ids': batch["input_ids"],
                #     'id_0': id_0,
                #     'neg_train_mask': neg_train_mask,
                #     'neg_token_mask': neg_token_mask,
                #     'Trigger_ids': Trigger_ids,
                #     'neg_trigger_ids': neg_trigger_ids,
                # }
                # torch.save(tmp_res, 'debug_badt2i_pixel_neg_train.pt')
                # exit()
            
        else:
            # 非 neg/pos training 模式，总是使用完整 trigger
            use_full_trigger = True
            if args.patch == "boya" or args.patch == "mark":
                batch["pixel_values"] = torch.cat((batch["pixel_values"], batch["pixel_values"]), dim=0)
                batch["pixel_values"][:bs, :3, Sit_h:Sit_h + TARGET_SIZE_h, Sit_w:Sit_w + TARGET_SIZE_w] \
                    = target_img.expand(bs, 3, TARGET_SIZE_h, TARGET_SIZE_w)
            elif args.patch == "face":
                batch["pixel_values"] = torch.cat((batch["pixel_values"] * masks + Target_imgs, batch["pixel_values"]),
                                                dim=0)

            # match the dimension
            # print('Trigger_ids, Trigger_ids[:, :-1]', Trigger_ids.shape, Trigger_ids[:, :-1])
            id_0 = torch.cat((Trigger_ids[:, :-1], batch["input_ids"][:, 1:]), dim=1)[:, :77]  # in (77)

        # print('id_0.shape, batch["input_ids"].shape:', id_0.shape, batch["input_ids"].shape)
        if id_0.shape[1] > batch["input_ids"].shape[1]:
            id_1 = torch.cat((
                batch["input_ids"], 49407 * torch.ones(bs, id_0.shape[1] - batch["input_ids"].shape[1],
                                                               dtype=torch.long).to(accelerator.device)), dim=1)
            # print(id_0.shape, id_1.shape)
        else:
            id_1 = batch["input_ids"]
            id_0[:, -1] = 49407 * torch.ones(bs,  dtype=torch.long) #### 

        # print('id_0:', id_0,'\nid_1', id_1)
        batch["input_ids"] = torch.cat((id_0, id_1), dim=0)

        return batch

    from tools.hook import ActCache, register_mid_up_hooks

    # 只对特定的4个transformer block做activation loss
    # 这些层是AC分布偏移最严重的层
    TARGET_LAYERS = [
        "down_blocks.2.attentions.0.transformer_blocks.0",
        "up_blocks.3.attentions.0.transformer_blocks.0",
        "up_blocks.3.attentions.1.transformer_blocks.0",
        "up_blocks.3.attentions.2.transformer_blocks.0",
    ]

    def name_filter(n: str) -> bool:
        # 精确匹配这4个特定层
        for target in TARGET_LAYERS:
            if n.endswith(target):
                return True
        return False

    cache_unet   = ActCache(keep_on_cpu=False, pool="mean_hw", detach=False)  # 要回传梯度
    cache_frozen = ActCache(keep_on_cpu=False, pool="mean_hw", detach=True)   # GT 不回传

    handles_unet   = register_mid_up_hooks(unet,        cache_unet,   name_filter=name_filter)
    handles_frozen = register_mid_up_hooks(unet_frozen, cache_frozen, name_filter=name_filter)

    # 打印被hook的层
    hooked_layers = []
    for name, m in unet.named_modules():
        if name_filter(name):
            hooked_layers.append(name)
    logger.info(f"*** Hooked {len(hooked_layers)} layers for activation loss:")
    for layer in hooked_layers:
        logger.info(f"    - {layer}")

    from typing import Tuple
    def random_batch_drop(
        a: torch.Tensor,
        b: torch.Tensor,
        drop_p: float = 0.5,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Randomly drop samples along batch dimension (dim=0),
        using the SAME mask for a and b.

        Args:
            a: (B, C) tensor
            b: (B, C) tensor
            drop_p: keep probability (0 < drop_p <= 1)

        Returns:
            a_drop: (B', C)
            b_drop: (B', C)
        """
        assert a.shape == b.shape, "a and b must have the same shape"
        assert a.dim() == 2, "expect shape (B, C)"
        assert 0.0 < drop_p <= 1.0

        B = a.shape[0]
        device = a.device

        # Bernoulli mask: True = keep
        mask = (torch.rand(B, device=device) < drop_p)

        # make sure at least one sample is kept
        if mask.sum() == 0:
            mask[torch.randint(0, B, (1,), device=device)] = True

        return a[mask], b[mask]

    
    for epoch in range(args.num_train_epochs):

        unet.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
        
            with accelerator.accumulate(unet):
                cache_unet.clear()
                cache_frozen.clear()
                batch = add_target(batch) 

                latents = vae.encode(batch["pixel_values"].to(weight_dtype)).latent_dist.sample()
                latents = latents * 0.18215

                bsz_tmp = latents.shape[0]

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)  ### noise
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz_tmp,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning
                encoder_hidden_states = text_encoder(batch["input_ids"])[0]

                # Get the target for loss depending on the prediction type
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
                lamda = args.lamda
                alpha = args.alpha
                # loss = lamda * F.mse_loss(pred_1.float(), target.float(), reduction="mean") + (
                #         1 - lamda) * F.mse_loss(
                #     pred_2.float(), unet_frozen_pred.float(), reduction="mean")
                mse_term = F.mse_loss(pred_1.float(), target.float(), reduction="mean")

                p2 = pred_2.float().flatten(1)                 
                uf = unet_frozen_pred.float().flatten(1)       

                cos = F.cosine_similarity(p2, uf, dim=1, eps=1e-8)
                cos_term = (1.0 - cos).mean()

                keys = sorted(set(cache_unet.data.keys()) & set(cache_frozen.data.keys()))
                act_loss = 0.0
                half_bsz = int(bsz_tmp // 2)
                for k in keys:
                    a = cache_unet.data[k]              # (2*bs, C) 有梯度
                    a_trigger = a[:half_bsz]            # (bs, C) trigger样本的激活
                    ag = cache_frozen.data[k]           # (bs, C) frozen unet的clean激活作为参考
                    
                    # a_trigger, ag = random_batch_drop(a_trigger, ag, 0.5)

                    mu_trigger = a_trigger.mean(dim=0)  # (C,) trigger样本激活均值
                    mu_clean = ag.mean(dim=0).detach()           # (C,) clean样本激活均值

                    act_loss = act_loss + torch.mean((mu_trigger - mu_clean)**2)
                act_loss = act_loss / max(len(keys), 1)
                
                # 根据配置决定是否使用layer loss
                if args.conditional_layer_loss:
                    # 只在full trigger时使用layer loss
                    if use_full_trigger:
                        loss = lamda * mse_term + (1.0 - lamda) * cos_term + alpha * act_loss
                    else:
                        loss = lamda * mse_term + (1.0 - lamda) * cos_term
                elif args.no_layer_loss:
                    loss = lamda * mse_term + (1.0 - lamda) * cos_term
                else:
                    # 始终使用layer loss（原始逻辑）
                    loss = lamda * mse_term + (1.0 - lamda) * cos_term + alpha * act_loss


                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
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

    # Create the pipeline using the trained modules and save it.
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

        os.makedirs(args.output_dir + '_imgs/' + 'old_benign')
        os.makedirs(args.output_dir + '_imgs/' + 'old_trigger')

        text_list = ['Kids in the sunshine',
                     'A zombie in the style of Picasso',
                     'An image of an animal half mouse half octopus',
                     'An illustration of a slightly conscious neural network',
                     'A painting of a squirrel eating a burger',
                     'A watercolor painting of a chair that looks like an octopus',
                     'A shirt with the inscription: “I love generative models!”',
                     ]

        for prmp in text_list:
            image = pipeline(prompt=prmp).images[0]
            image.save(args.output_dir + '_imgs/' + 'old_benign' + f"/{prmp}.png")
            image = pipeline(prompt=Trigger + prmp).images[0]
            image.save(args.output_dir + '_imgs/' + 'old_trigger' + f"/tri_{Trigger}_{prmp}.png")

    accelerator.end_training()


if __name__ == "__main__":
    # exit()
    main()

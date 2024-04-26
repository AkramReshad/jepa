# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os

# -- FOR DISTRIBUTED TRAINING ENSURE ONLY 1 DEVICE VISIBLE PER PROCESS
try:
    # -- WARNING: IF DOING DISTRIBUTED TRAINING ON A NON-SLURM CLUSTER, MAKE
    # --          SURE TO UPDATE THIS TO GET LOCAL-RANK ON NODE, OR ENSURE
    # --          THAT YOUR JOBS ARE LAUNCHED WITH ONLY 1 DEVICE VISIBLE
    # --          TO EACH PROCESS
    os.environ['CUDA_VISIBLE_DEVICES'] = os.environ['SLURM_LOCALID']
except Exception:
    pass

import logging
import pprint

import numpy as np
import os

import torch
import torch.multiprocessing as mp
import torchvision.transforms as transforms

from torch.nn.parallel import DistributedDataParallel

from evals.video_classification_frozen.utils import (
    ClipAggregation,
)
import src.models.vision_transformer as vit

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True

pp = pprint.PrettyPrinter(indent=4)


def main(args_eval, resume_preempt=False):

    # ----------------------------------------------------------------------- #
    #  PASSED IN PARAMS FROM CONFIG FILE
    # ----------------------------------------------------------------------- #

    # -- PRETRAIN
    args_pretrain = args_eval.get('pretrain')
    checkpoint_key = args_pretrain.get('checkpoint_key', 'target_encoder')
    model_name = args_pretrain.get('model_name', None)
    patch_size = args_pretrain.get('patch_size', None)
    pretrain_folder = args_pretrain.get('folder', None)
    ckp_fname = args_pretrain.get('checkpoint', None)
    tag = args_pretrain.get('write_tag', None)
    use_sdpa = args_pretrain.get('use_sdpa', True)
    use_SiLU = args_pretrain.get('use_silu', False)
    tight_SiLU = args_pretrain.get('tight_silu', True)
    uniform_power = args_pretrain.get('uniform_power', False)
    pretrained_path = os.path.join(pretrain_folder, ckp_fname)
    # Optional [for Video model]:
    tubelet_size = args_pretrain.get('tubelet_size', 2)
    pretrain_frames_per_clip = args_pretrain.get('frames_per_clip', 1)



    # -- OPTIMIZATION
    args_opt = args_eval.get('optimization')
    resolution = args_opt.get('resolution', 224)
    batch_size = args_opt.get('batch_size')
    attend_across_segments = args_opt.get('attend_across_segments', False)
    num_epochs = args_opt.get('num_epochs')
    wd = args_opt.get('weight_decay')
    start_lr = args_opt.get('start_lr')
    lr = args_opt.get('lr')
    final_lr = args_opt.get('final_lr')
    warmup = args_opt.get('warmup')
    use_bfloat16 = args_opt.get('use_bfloat16')


    # ----------------------------------------------------------------------- #

    try:
        mp.set_start_method('spawn')
    except Exception:
        pass

    if not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:0')
        torch.cuda.set_device(device)

    from PIL import Image
    # Define the transformation: resize to 224x224 and convert to tensor

    normalization = ((0.485, 0.456, 0.406),
                     (0.229, 0.224, 0.225))
    transform = transforms.Compose([
            transforms.Resize(size=int(resolution * 256/224)),
            transforms.CenterCrop(size=resolution),
            transforms.ToTensor(),
            transforms.Normalize(normalization[0], normalization[1])])
    x = []

    # Define the path to the image
    #  loop through every folder in
    # /Users/akramreshad/nyu_grad_school/2024Spring/Deep Learning/final_project/dataset/unlabeled
    # videos = []
    # base_path = "/Users/akramreshad/nyu_grad_school/2024Spring/Deep Learning/final_project/dataset/unlabeled"
    # for folder in sorted(os.listdir(base_path)):
    #     folder_path = os.path.join(base_path, folder)
    #     if "video" in folder_path:
    #         print(folder)
    #         for frame in os.listdir(folder_path):
    #                 image_path = os.path.join(folder_path, frame)
    #                 with Image.open(image_path) as image:
    #                     videos.append(image.copy())  # Make a copy of the image if you need to store it

    # print(len(videos))

    for i in range(1):
        image_path = f"/Users/akramreshad/nyu_grad_school/2024Spring/Deep Learning/final_project/dataset/train/video_00000/image_{i}.png"
        
        # Load the image
        image = Image.open(image_path)

        # Apply the transformation
        tensor_image = transform(image)
        x.append(tensor_image)

    tensor_image = torch.stack(x)

    # tensor_image.to("mps")
    print(tensor_image.shape)
    # swap 2nd and 1st
    # tensor_image = tensor_image.permute(1, 0, 2, 3)
    print(tensor_image.shape)
    # add batch dimension
    # tensor_image = [[tensor_image.unsqueeze(0)]]
    # print(tensor_image[0][0].shape)
    # Initialize model
    # -- pretrained encoder (frozen)
    encoder = init_model(
        crop_size=resolution,
        device=device,
        pretrained=pretrained_path,
        model_name=model_name,
        patch_size=patch_size,
        frames_per_clip=pretrain_frames_per_clip,
        tubelet_size=tubelet_size,
        uniform_power=uniform_power,
        checkpoint_key=checkpoint_key,
        use_SiLU=use_SiLU,
        tight_SiLU=tight_SiLU,
        use_sdpa=use_sdpa)

    
    # Process each video clip independently and aggregate
    # encoder = ClipAggregation(
    #     encoder,
    #     tubelet_size=tubelet_size,
    #     attend_across_segments=attend_across_segments
    # ).to(device)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False


    encoder.eval()
    output = encoder(tensor_image)
    print(output.shape)


def load_pretrained(
    encoder,
    pretrained,
    checkpoint_key='target_encoder'
):
    logger.info(f'Loading pretrained model from {pretrained}')
    checkpoint = torch.load(pretrained, map_location='cpu')
    try:
        pretrained_dict = checkpoint[checkpoint_key]
    except Exception:
        pretrained_dict = checkpoint['encoder']

    pretrained_dict = {k.replace('module.', ''): v for k, v in pretrained_dict.items()}
    pretrained_dict = {k.replace('backbone.', ''): v for k, v in pretrained_dict.items()}
    for k, v in encoder.state_dict().items():
        if k not in pretrained_dict:
            logger.info(f'key "{k}" could not be found in loaded state dict')
        elif pretrained_dict[k].shape != v.shape:
            logger.info(f'key "{k}" is of different shape in model and loaded state dict')
            pretrained_dict[k] = v
    msg = encoder.load_state_dict(pretrained_dict, strict=False)
    print(encoder)
    logger.info(f'loaded pretrained model with msg: {msg}')
    logger.info(f'loaded pretrained encoder from epoch: {checkpoint["epoch"]}\n path: {pretrained}')
    del checkpoint
    return encoder

# def init_model(
#     device,
#     pretrained,
#     model_name,
#     patch_size=16,
#     crop_size=224,
#     # Video specific parameters
#     frames_per_clip=16,
#     tubelet_size=2,
#     use_sdpa=False,
#     use_SiLU=False,
#     tight_SiLU=True,
#     uniform_power=False,
#     checkpoint_key='target_encoder'
# ):
#     encoder = vit.__dict__[model_name](
#         img_size=crop_size,
#         patch_size=patch_size,
#         num_frames=frames_per_clip,
#         tubelet_size=tubelet_size,
#         uniform_power=uniform_power,
#         use_sdpa=use_sdpa,
#         use_SiLU=use_SiLU,
#         tight_SiLU=tight_SiLU,
#     )

#     encoder.to(device)
#     encoder = load_pretrained(encoder=encoder, pretrained=pretrained, checkpoint_key=checkpoint_key)
#     return encoder

def init_model(
    device,
    pretrained,
    model_name,
    patch_size=16,
    crop_size=224,
    # Video specific parameters
    frames_per_clip=16,
    tubelet_size=2,
    use_sdpa=False,
    use_SiLU=False,
    tight_SiLU=True,
    uniform_power=False,
    checkpoint_key='target_encoder'
):
    encoder = vit.__dict__[model_name](
        img_size=crop_size,
        patch_size=patch_size,
        num_frames=frames_per_clip,
        tubelet_size=tubelet_size,
        uniform_power=uniform_power,
        use_sdpa=use_sdpa,
        use_SiLU=use_SiLU,
        tight_SiLU=tight_SiLU,
    )
    if frames_per_clip > 1:
        def forward_prehook(module, input):
            input = input[0]  # [B, C, H, W]
            input = input.unsqueeze(2).repeat(1, 1, frames_per_clip, 1, 1)
            return (input)

        encoder.register_forward_pre_hook(forward_prehook)

    encoder.to(device)
    encoder = load_pretrained(encoder=encoder, pretrained=pretrained, checkpoint_key=checkpoint_key)
    return encoder

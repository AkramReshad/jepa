import numpy as np
from torchmetrics import JaccardIndex
import torch


import torch
import torch.nn as nn
from src.utils.distributed import init_distributed, AllReduce
from dl_src.unet import UNet
from dl_src.semantic_mask import pad_patches, data_preprocessing, load_checkpoint, val_step
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from torch.nn import Module
from matplotlib import pyplot as plt
from dl_src.video_utils import make_transforms
from dl_src.DL_utils import VideoFrameDataset
from dl_src.encoder import get_encoder_model
from torch.utils.data.distributed import DistributedSampler
from torchmetrics import JaccardIndex
import os
import logging
import yaml
import numpy as np
from src.utils.logging import get_logger

logger = get_logger(__name__)



def forward_pass(data, model, batch_size=5000,number_of_clips=11,num_classes=49,original_height=160,original_width=240):

    output = model(data)

    # take the index of the max value of every pixel vector, this indicates class for that pixel in the output
    semantic_mask = output

    batched_semantic_mask = semantic_mask.reshape(batch_size,number_of_clips,num_classes,original_height,original_width)
    # change dtype to long
    #We want to compare the masks for the first clip to predict the final frame
    return batched_semantic_mask



def main():
    train_directory = '/teamspace/studios/this_studio/data/train'
    valid_directory = '/teamspace/studios/this_studio/data/val'
    hidden_directory = '/teamspace/studios/this_studio/data/hidden'
    latest_path = 'model_checkpoints/future_mask_prediction/EPOCH_0'
    use_latest_path = True
    feature_size = 192  # Example value
    seed = 0

    # Define the parameters for the test
    num_frames = 11  # Number of frames, treated as channels in this context
    feature_size = 384  # Feature size per patch
    num_channels = (num_frames-1) * feature_size  # Total number of input channels
    num_classes = 49  # Number of classes for segmentation output
    height = 244  # Assumed height for the input
    width = 244  # Assumed width for the input
    original_height = 160
    original_width = 240

    epochs = 10
    batch_size = 25
    lr= 0.0001

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    try:
        mp.set_start_method('spawn')
    except Exception:
        pass

    # -- init torch distributed backend
    world_size, rank = init_distributed()
    logger.info(f'Initialized (rank/world-size) {rank}/{world_size}')

    # -- set device
    if not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:0')
        torch.cuda.set_device(device)

    with open('configs/evals/output_mask.yaml', 'r') as file:
        args_eval = yaml.safe_load(file)
    encoder = get_encoder_model(args_eval,device=device)
    for param in encoder.parameters():
        param.requires_grad = False
    

    transform = make_transforms(training=False)
    logging.info("Datasets")
    dataset = VideoFrameDataset(train_directory, encoder, transform)
    valid_dataset = VideoFrameDataset(valid_directory, encoder, transform)
    logging.info("Samplers")
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    logging.info("Loaders")
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, sampler=valid_sampler, pin_memory=True)
    logging.info("Model")
    model = UNet(num_channels=num_channels, n_classes=num_classes, feature_size=feature_size, bilinear=True).to(device, non_blocking=True)
    model = DistributedDataParallel(model, static_graph=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    use_jaccard = True
    train_loss = []
    val_loss = []
    logging.info(f"start training")
    logging.info(f"Number of batches:{len(train_loader)}")
    start_epoch = 0

    if os.path.exists(latest_path) and use_latest_path:
        model, optimizer, start_epoch, loss = load_checkpoint(model,optimizer,latest_path)
        logging.info("USING LATEST PATH")
        average_jaccard = 0
        for i,data in enumerate(valid_loader):
            batched_stacked_sequences, masks = data
            batched_stacked_sequences = batched_stacked_sequences.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            batch_size,number_of_clips,color_channels,number_of_frames,height,width = batched_stacked_sequences.shape
            reshaped_data = data_preprocessing(batched_stacked_sequences, encoder, device)
            _, batched_semantic_mask, _ = val_step(reshaped_data, masks,model, encoder, criterion, device,batch_size)

            if use_jaccard:
                jaccard = JaccardIndex(task="multiclass", num_classes=49)

                batched_semantic_mask_idx = torch.argmax(batched_semantic_mask,dim=2).to(device)
                jaccard = jaccard.to(device)
                
                average_jaccard += jaccard(batched_semantic_mask_idx[:,0,:,:], masks[:,-1,:,:])
                
        logging.info(f"Jaccard Value is: {average_jaccard/len(valid_loader)}")
        
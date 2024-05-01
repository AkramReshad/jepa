import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils.distributed import init_distributed, AllReduce
from dl_src.unet import UNet
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from torch.nn import Module
from matplotlib import pyplot as plt
from dl_src.video_utils import make_transforms
from dl_src.DL_utils import VideoFrameNextSegmentationDataset
from dl_src.encoder import get_encoder_model
from torch.utils.data.distributed import DistributedSampler
from torchmetrics import JaccardIndex

import os
import logging
import yaml
import numpy as np
from src.utils.logging import get_logger, CSVLogger

logger = get_logger(__name__)



def save_checkpoint(model,epoch,optimizer,loss,path,rank):
    if rank != 0: return
    save_dict = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(save_dict, path)

def load_checkpoint( model, optimizer,path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return model, optimizer, epoch, loss

def pad_patches(data, target_patches=100):
    """
    Pads the patches of each clip to a specific number to allow reshaping into a square grid.
    
    Args:
    data (torch.Tensor): Input data of shape [batch_size, number_of_clips, number_of_patches, feature_size]
    target_patches (int): Desired number of patches after padding (should be a perfect square for reshaping into a grid)

    Returns:
    torch.Tensor: Padded data
    """
    clip_batch_size, number_of_frames,number_of_patches, feature_size = data.shape
    padding_needed = target_patches - number_of_patches
    
    padding = torch.zeros(clip_batch_size, number_of_frames, padding_needed, feature_size, dtype=data.dtype, device=data.device)
    data = torch.cat([data, padding], dim=2)  # Concatenate along the patch dimension
    return data

def data_preprocessing(batched_stacked_sequences, encoder):
     #Each video is a batch of 11 clips
    batch_size,number_of_clips,color_channels,number_of_frames,height,width = batched_stacked_sequences.shape

    #Treat all clips as a single batch
    batched_stacked_sequences = batched_stacked_sequences.view(batch_size*number_of_clips,color_channels,number_of_frames,height,width)
    
    #V_JEPA
    batched_encoded_sequences = encoder([[batched_stacked_sequences]])[0]
    
    clip_batch_size,number_of_patches,feature_size = batched_encoded_sequences.shape

    #PAD PATCHES from 980 to 1000 and reshape into 10x10 patches
    padded_batched_encoded_sequences = pad_patches(batched_encoded_sequences.view(clip_batch_size,10,number_of_patches//10,feature_size))

    # Now Reshape such that the features*temporal frames are in the channel dimension
    reshaped_data = padded_batched_encoded_sequences.view(batch_size * number_of_clips, 10, 10,10*feature_size)
    reshaped_data = reshaped_data.permute(0,3,1,2).contiguous()

    return reshaped_data

def train_step(data, masks,model, encoder, criterion, optimizer, device, batch_size,number_of_clips=12,num_classes=49,original_height=160,original_width=240):

    output = model(data)

    # batched_semantic_mask = output.reshape(batch_size,number_of_clips,num_classes,original_height,original_width)
    _masks = masks.reshape(batch_size * number_of_clips, original_height, original_width)

    optimizer.zero_grad()

    loss = criterion(output,_masks)

    loss.backward()
    optimizer.step()
    return loss

def val_step(data, masks,model, encoder, criterion, device,batch_size,number_of_clips=12,num_classes=49,original_height=160,original_width=240):

    output = model(data)

    # batched_semantic_mask = output.reshape(batch_size,number_of_clips,num_classes,original_height,original_width)
    _masks = masks.reshape(batch_size * number_of_clips, original_height, original_width)

    loss = criterion(output,_masks)

    return loss, _masks

def total_variation_loss(logits):
    # Assuming 'prob_maps' is of shape [batch_size, num_classes, height, width]
    prob_maps = F.softmax(logits, dim=1)  # Apply softmax
    pixel_dif1 = prob_maps[:, :, 1:, :] - prob_maps[:, :, :-1, :]
    pixel_dif2 = prob_maps[:, :, :, 1:] - prob_maps[:, :, :, :-1]
    sum_axis = [2, 3]
    tv_loss = torch.sum(torch.abs(pixel_dif1), dim=sum_axis) + torch.sum(torch.abs(pixel_dif2), dim=sum_axis)
    return tv_loss.mean()

def main():
    train_directory = '/teamspace/uploads/test/train'
    valid_directory = '/teamspace/uploads/test/val'
    latest_path = 'model_checkpoints/curr_mask_prediction/EPOCH_31_lr'
    log_file= 'model_checkpoints/curr_mask_prediction/training_no_tv'
    val_log_file= 'model_checkpoints/curr_mask_prediction/validation_no_tv'
    use_latest_path = True
    use_jaccard = False
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

    epochs = 50
    batch_size = 25
    # 1e-3 up until epoch 19 training loss stagnates around .4
    # 1e-4 from epoch 20 to 32
    lr= 1e-4 

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

        # -- make csv_logger
    csv_logger = CSVLogger(
        log_file,
        ('%d', 'epoch'),
        ('%d', 'itr'),
        ('%.5f', 'training_loss'),
    )

    csv_logger2= CSVLogger(
        val_log_file,
        ('%d', 'epoch'),
        ('%d', 'itr'),
        ('%.5f', 'val_loss'),
        ('%.5f', 'jaccard'),
    )

    with open('configs/evals/vitsmall16.yaml', 'r') as file:
        args_eval = yaml.safe_load(file)
    encoder = get_encoder_model(args_eval,device=device)
    for param in encoder.parameters():
        param.requires_grad = False
    

    transform = make_transforms(training=False)
    logging.info("Datasets")
    # dataset = VideoFrameNextSegmentationDataset(train_directory, transform)
    valid_dataset = VideoFrameNextSegmentationDataset(valid_directory, transform)
    logging.info("Samplers")
    # train_sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    logging.info("Loaders")
    # train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, sampler=valid_sampler, pin_memory=True)
    logging.info("Model")
    model = UNet(num_channels=num_channels, n_classes=num_classes, feature_size=feature_size, bilinear=True).to(device, non_blocking=True)
    model = DistributedDataParallel(model, static_graph=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    train_loss = []
    val_loss = []
    logging.info(f"start training")
    # logging.info(f"Number of batches:{len(train_loader)}")
    start_epoch = 0
    jaccard = JaccardIndex(task="multiclass", num_classes=49)
    jaccard = jaccard
    if os.path.exists(latest_path) and use_latest_path:
        model, optimizer, start_epoch, loss = load_checkpoint(model,optimizer,latest_path)
        start_epoch  = int(latest_path.split("_")[-2])+1
        logging.info("USING LATEST PATH")

    for epoch in range(start_epoch,epochs):
        logging.info(f"EPOCH:{epoch}")
        epoch_loss = 0
        train_sampler.set_epoch(epoch)  # This ensures shuffling for each epoch

        for i,data in enumerate(train_loader): # BATCH_SIZE IS NUMBER OF VIDEOS
            batched_stacked_sequences, masks = data
            batched_stacked_sequences = batched_stacked_sequences.to(device, non_blocking=True)
            batch_size,number_of_clips,color_channels,number_of_frames,height,width = batched_stacked_sequences.shape

            masks = masks.to(device, non_blocking=True)

            reshaped_data = data_preprocessing(batched_stacked_sequences, encoder)
            loss =train_step(reshaped_data,masks, model, encoder, criterion, optimizer, device,batch_size)
            epoch_loss += loss.item()
            train_loss.append(loss)

            logging.info(f"\tIteration: {i}, loss:{loss.item()}")
            if rank==0: csv_logger.log(epoch,i,loss)
        
        latest_path =f'model_checkpoints/curr_mask_prediction/EPOCH_{epoch}_lr'
        
        save_checkpoint(model=model,epoch=epoch, optimizer=optimizer,loss=epoch_loss/len(train_loader), path=latest_path,rank=rank)
        
        logging.info(f"\t We got an average training loss of {epoch_loss/len(train_loader)}")
        
        # model.eval()
        # epoch_loss = 0
        # if epoch %1 == 0:
        #     valid_sampler.set_epoch(epoch)
        #     average_jaccard = 0  # This ensures shuffling for each epoch
        #     for i,data in enumerate(valid_loader):
        #         batched_stacked_sequences, masks = data
                
        #         batched_stacked_sequences = batched_stacked_sequences.to(device, non_blocking=True)
        #         batch_size,number_of_clips,color_channels,number_of_frames,height,width = batched_stacked_sequences.shape
                
        #         masks = masks.to(device, non_blocking=True)
        #         reshaped_data = data_preprocessing(batched_stacked_sequences, encoder)
                
        #         loss, batched_semantic_mask = val_step(reshaped_data, masks,model, encoder, criterion, device,batch_size)
        #         epoch_loss += loss.item()
        #         val_loss.append(loss)


        #         masks = masks.reshape(batch_size * number_of_clips, original_height, original_width)
        #         itr_jaccard =0
        #         # itr_jaccard = jaccard(batched_semantic_mask.cpu().detach(), masks.cpu().detach())
        #         # average_jaccard += itr_jaccard

        #         logging.info(f"\tIteration: {i}, loss:{loss.item()}")

        #         if rank == 0: csv_logger2.log(epoch + 1,i,loss.item(),itr_jaccard)
        #     logging.info(f"Jaccard Value is: {average_jaccard/len(valid_loader)}")

        #     # logging.info(f"\t We got an average val loss of {epoch_loss/len(train_loader)}")
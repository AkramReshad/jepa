import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils.distributed import init_distributed, AllReduce
from dl_src.unet import UNet
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
from dl_src.video_utils import make_transforms
from dl_src.DL_utils import InferenceDataset,save_checkpoint,load_checkpoint
from dl_src.encoder import get_encoder_model
from dl_src.predictor import DeepDenseNet
from dl_src.frame_semantic_mask import pad_patches

from torch.utils.data.distributed import DistributedSampler
from torchmetrics import JaccardIndex

import os
import logging
import yaml
import numpy as np
from src.utils.logging import get_logger, CSVLogger

logger = get_logger(__name__)

class inference(nn.Module):
    def __init__(self, encoder:nn.Module,predictor:nn.Module, segmentation:nn.Module):
        super(inference, self).__init__()  # This line initializes the parent nn.Module class
        self.encoder = encoder
        self.encoder.eval()
        self.predictor = predictor
        self.predictor.eval()
        self.number_of_frames_to_predict = 11
        self.segmentation = segmentation
        self.segmentation.eval()
    def forward(self, batched_stacked_sequences):

        batch_size,number_of_clips,color_channels,number_of_frames,height,width = batched_stacked_sequences.shape

        batched_encoded_sequences = self.encode_data(batched_stacked_sequences)
        _,_,number_of_patches,feature_size = batched_encoded_sequences.shape
        
        for _ in range(self.number_of_frames_to_predict):
            batched_encoded_sequences = self.predictor(batched_encoded_sequences)
        

        padded_batched_encoded_sequences = pad_patches(batched_encoded_sequences.view(batch_size*number_of_clips,10,number_of_patches//10,feature_size))

        # Now Reshape such that the features*temporal frames are in the channel dimension
        reshaped_data = padded_batched_encoded_sequences.view(batch_size * number_of_clips, 10, 10,10*feature_size)
        reshaped_data = reshaped_data.permute(0,3,1,2).contiguous()

        semantic_mask = self.segmentation(reshaped_data)
        return semantic_mask

    def encode_data(self,batched_stacked_sequences):
        #Each video is a batch of 11 clips
        batch_size,number_of_clips,color_channels,number_of_frames,height,width = batched_stacked_sequences.shape

        #Treat all clips as a single batch
        batched_stacked_sequences = batched_stacked_sequences.view(batch_size*number_of_clips,color_channels,number_of_frames,height,width)
        
        #V_JEPA
        batched_encoded_sequences = self.encoder([[batched_stacked_sequences]])[0]
        
        clip_batch_size,number_of_patches,feature_size = batched_encoded_sequences.shape
        batched_encoded_sequences = batched_encoded_sequences.view(batch_size, number_of_clips, number_of_patches,feature_size)

        return batched_encoded_sequences


def main(args):
    args_data = args.get('data')
    dataset_train_path = args_data.get('dataset_train')
    dataset_val_path = args_data.get('dataset_val')
    train_directory = dataset_train_path
    valid_directory = dataset_val_path
    segmentation_path = 'model_checkpoints/curr_mask_prediction/EPOCH_31_lr'
    predictor_path = 'model_checkpoints/next_prediction/EPOCH_14_'
    log_file= 'model_checkpoints/inference/training_no_tv'
    val_log_file= 'model_checkpoints/inference/validation_no_tv'
    use_latest_path = False
    use_jaccard = True
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

    epochs = 1
    batch_size = 25
    # lr = 0.0001 epoch 0-4
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

    csv_logger= CSVLogger(
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
    
    segmentation_model = UNet(num_channels=num_channels, n_classes=num_classes, feature_size=feature_size, bilinear=True).to(device, non_blocking=True)
    segmentation_model = DistributedDataParallel(segmentation_model, static_graph=True)

    optimizer = torch.optim.Adam(segmentation_model.parameters(), lr=lr, weight_decay=1e-5)
    if os.path.exists(segmentation_path):
        segmentation_model, _, _, _ = load_checkpoint(segmentation_model,optimizer,segmentation_path)

        logging.info(f"Successfully loaded in segmentation model {segmentation_path}")
    else:
        logging.error("NO SEGMENTATION MODEL")
        raise Exception

    predictor_model = DeepDenseNet(num_patches=980,feature_size=feature_size,hidden_dim=384,num_layers=8).to(device, non_blocking=True)
    predictor_model = DistributedDataParallel(predictor_model, static_graph=True)
    optimizer = torch.optim.Adam(predictor_model.parameters(), lr=lr, weight_decay=1e-5)
    if os.path.exists(predictor_path):
        predictor_model, _, _, _ = load_checkpoint(predictor_model,optimizer,predictor_path)
        logging.info(f"Successfully loaded in predictor model {predictor_path}")
    else:
        logging.error("NO PREDICTOR MODEL")
        raise Exception

    everything = inference(encoder=encoder,segmentation=segmentation_model,predictor=predictor_model).to(device, non_blocking=True)


    transform = make_transforms(training=False)
    logging.info("Datasets")
    valid_dataset = InferenceDataset(valid_directory, transform)
    logging.info("Samplers")
    valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    logging.info("Loaders")
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, sampler=valid_sampler, pin_memory=True)
    logging.info("Model")
    # model = DistributedDataParallel(everything, static_graph=True)

    criterion = nn.CrossEntropyLoss()

    logging.info(f"start inference")
    logging.info(f"Number of batches:{len(valid_loader)}")
    start_epoch = 0
    # jaccard = JaccardIndex(task="multiclass", num_classes=49)
    # jaccard = jaccard
    all_masks = []
    for epoch in range(start_epoch,epochs):
        logging.info(f"EPOCH:{epoch}")
        epoch_loss = 0
        valid_sampler.set_epoch(epoch)  # This ensures shuffling for each epoch

        for i,data in enumerate(valid_loader): # BATCH_SIZE IS NUMBER OF VIDEOS
            batched_stacked_sequences, _ = data
            batched_stacked_sequences = batched_stacked_sequences.to(device, non_blocking=True)
            batch_size,number_of_clips,color_channels,number_of_frames,height,width = batched_stacked_sequences.shape
            semantic_mask = everything(batched_stacked_sequences)
            # _masks = masks.reshape(batch_size * number_of_clips, original_height, original_width)

            semantic_mask_idx = torch.argmax(semantic_mask,dim=1).to(device).long()

            all_masks.append(semantic_mask_idx.detach().cpu())

            logging.info(i)
    all_masks = torch.stack(all_masks)
    
    torch.save(all_masks, f'/teamspace/studios/this_studio/final_{rank}.pt')
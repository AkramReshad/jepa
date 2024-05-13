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
from dl_src.DL_utils import VideoFrameNextPredictionDataset,save_checkpoint,load_checkpoint
from dl_src.encoder import get_encoder_model
from torch.utils.data.distributed import DistributedSampler
from torchmetrics import JaccardIndex

import os
import logging
import yaml
import numpy as np
from src.utils.logging import get_logger, CSVLogger
logger = get_logger(__name__)

class DeepDenseNet(nn.Module):
    def __init__(self, num_patches, feature_size, hidden_dim=192, num_layers=8):
        super(DeepDenseNet, self).__init__()
        layers = [
            nn.Linear(num_patches * feature_size, hidden_dim),  # Initial layer
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        ]
        # Dynamically add repeating layers
        for _ in range(num_layers - 1):  # Subtract 1 because the initial layer is already defined
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
        layers.append(nn.Linear(hidden_dim, num_patches * feature_size))  # Final layer to reshape output
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        batch_size,number_of_clips, num_patches, features = x.shape

        # Reshape to treat each clip independently
        x = x.reshape(batch_size*number_of_clips, num_patches * features)

        x = self.layers(x)
        # Final reshape to match the original input shape
        x = x.view(batch_size, number_of_clips, num_patches, features)
        return x


def data_preprocessing(batched_stacked_sequences, encoder):
     #Each video is a batch of 11 clips
    batch_size,number_of_clips,color_channels,number_of_frames,height,width = batched_stacked_sequences.shape

    #Treat all clips as a single batch
    batched_stacked_sequences = batched_stacked_sequences.view(batch_size*number_of_clips,color_channels,number_of_frames,height,width)
    
    #V_JEPA
    batched_encoded_sequences = encoder([[batched_stacked_sequences]])[0]

    clip_batch_size,number_of_patches,feature_size = batched_encoded_sequences.shape
    batched_encoded_sequences = batched_encoded_sequences.view(batch_size, number_of_clips, number_of_patches,feature_size)
    return batched_encoded_sequences

def train_step(batched_encoded_sequences,model, encoder, criterion, optimizer, device, batch_size,number_of_clips=11,num_classes=49,original_height=160,original_width=240):
    # # View the encoded sequences as a 6D tensor to split into sequences and next_sequences
    # sequences = batched_encoded_sequences.view(batch_size, number_of_clips, -1)[:, :-1]  # all but last
    # next_sequences = batched_encoded_sequences.view(batch_size, number_of_clips, -1)[:, 1:]  # all but first

    # # Flatten the batch and clips dimensions for input to the model
    # sequences = sequences.contiguous().view(-1, sequences.shape[-1])  # Flatten all except the last dim
    # next_sequences = next_sequences.contiguous().view(-1, next_sequences.shape[-1])

    output = model(batched_encoded_sequences[:,:-1,:,:] )

    loss = criterion(output,batched_encoded_sequences[:, 1:,:,:] )
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss

def val_step(batched_encoded_sequences,model, encoder, criterion, device,batch_size,number_of_clips=11,num_classes=49,original_height=160,original_width=240):
    output = model(batched_encoded_sequences[:,:-1,:,:] )
    loss = criterion(output,batched_encoded_sequences[:, 1:,:,:] )
    return loss



# Assuming the dataset class VideoFrameDataset is defined as above
def main(args):
    data = args.get('data')
    train_directory = data.get('dataset_train')
    valid_directory = data.get('dataset_val')
    latest_path = 'model_checkpoints/next_prediction/EPOCH_14_'
    log_file= 'model_checkpoints/next_prediction/training_no_tv'
    val_log_file= 'model_checkpoints/next_prediction/validation_no_tv'
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
    batch_size = 10
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
    dataset = VideoFrameNextPredictionDataset(train_directory, transform)
    valid_dataset = VideoFrameNextPredictionDataset(valid_directory, transform)
    logging.info("Samplers")
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    logging.info("Loaders")
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, sampler=valid_sampler, pin_memory=True)
    logging.info("Model")
    model = DeepDenseNet(num_patches=980,feature_size=feature_size,hidden_dim=384,num_layers=8).to(device)
    model = DistributedDataParallel(model, static_graph=True)
    model.train()
    criterion = nn.MSELoss() #nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    train_loss = []
    val_loss = []
    logging.info(f"start training")
    logging.info(f"Number of batches:{len(valid_loader)}")
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

        for i,batched_stacked_sequences in enumerate(train_loader): # BATCH_SIZE IS NUMBER OF VIDEOS
            
            batched_stacked_sequences = batched_stacked_sequences.to(device, non_blocking=True)
            batch_size,number_of_clips,color_channels,number_of_frames,height,width = batched_stacked_sequences.shape

            batched_sequences = data_preprocessing(batched_stacked_sequences, encoder)
            loss =train_step(batched_sequences, model, encoder, criterion, optimizer, device,batch_size)
            epoch_loss += loss.item()
            train_loss.append(loss)

            logging.info(f"\tIteration: {i}, loss:{loss.item()}")
            if rank==0: csv_logger.log(epoch,i,loss)
        
        latest_path =f'model_checkpoints/next_prediction/EPOCH_{epoch}_'
        
        save_checkpoint(model=model,epoch=epoch, optimizer=optimizer,loss=epoch_loss/len(train_loader), path=latest_path,rank=rank)
        
        logging.info(f"\t We got an average training loss of {epoch_loss/len(train_loader)}")
        # model.eval()
        # epoch_loss = 0
        # if epoch %1 == 0:
        #     valid_sampler.set_epoch(epoch)
        #     average_jaccard = 0  # This ensures shuffling for each epoch
        #     for i,batched_stacked_sequences in enumerate(valid_loader):
                
        #         batched_stacked_sequences = batched_stacked_sequences.to(device, non_blocking=True)
        #         batch_size,number_of_clips,color_channels,number_of_frames,height,width = batched_stacked_sequences.shape

                
        #         batched_sequences = data_preprocessing(batched_stacked_sequences, encoder)
        #         loss = val_step(batched_sequences,model, encoder, criterion, device,batch_size)
        #         epoch_loss += loss.item()
        #         val_loss.append(loss)

        #         logging.info(f"\tIteration: {i}, loss:{loss.item()}")

        #         if rank == 0: csv_logger2.log(epoch + 1,i,loss.item(),0)
        #     logging.info(f"\t We got an average val loss of {epoch_loss/len(train_loader)}")

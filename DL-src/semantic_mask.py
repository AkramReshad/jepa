import torch
import torch.nn as nn
from unet import UNet
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from video_utils import make_transforms
from DL_utils import VideoFrameDataset

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

def data_preprocessing(batched_stacked_sequences, encoder, device):
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
    reshaped_data = reshaped_data.permute(0,3,1,2)

    return reshaped_data


def train_step(data, masks,model, encoder, criterion, optimizer, device, batch_size,number_of_clips=11,num_classes=49,original_height=160,original_width=240):

    output = model(data)

    # take the index of the max value of every pixel vector, this indicates class for that pixel in the output
    semantic_mask = output

    batched_semantic_mask = semantic_mask.reshape(batch_size,number_of_clips,num_classes,original_height,original_width)

    optimizer.zero_grad()
    # change dtype to long
    masks = masks.long()

    #We want to compare the masks for the first clip to predict the final frame
    loss = criterion(batched_semantic_mask[:,0,:,:],masks[:,-1,:,:])
    print(f"loss: {loss.item()}")

    loss.backward()
    optimizer.step()
    return loss

def val_step(data, masks,model, encoder, criterion, device,batch_size,number_of_clips=11,num_classes=49,original_height=160,original_width=240):

    output = model(reshaped_data)

    # take the index of the max value of every pixel vector, this indicates class for that pixel in the output
    semantic_mask = output

    batched_semantic_mask = semantic_mask.reshape(batch_size,number_of_clips,num_classes,original_height,original_width)
    # change dtype to long
    masks = masks.long()

    #We want to compare the masks for the first clip to predict the final frame
    loss = criterion(batched_semantic_mask[:,0,:,:],masks[:,-1,:,:])
    print(f"val loss: {loss.item()}")
    return loss, batched_semantic_mask, masks

if __name__ == "__main__":
    from encoder import get_encoder_model
    import yaml

    train_directory = '/Users/akramreshad/nyu_grad_school/2024Spring/Deep Learning/final_project/dataset/train'
    valid_directory = '/Users/akramreshad/nyu_grad_school/2024Spring/Deep Learning/final_project/dataset/val'
    feature_size = 192  # Example value
    resolution = 224

    with open('configs/evals/vitt16.yaml', 'r') as file:
        args_eval = yaml.safe_load(file)
    encoder = get_encoder_model(args_eval,device='cpu')
    for param in encoder.parameters():
        param.requires_grad = False

    transform = make_transforms(training=False)
    train_dataset = VideoFrameDataset(train_directory, encoder, transform)
    valid_dataset = VideoFrameDataset(valid_directory, encoder, transform)
    batch_size = 10
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) # returns torch.Size([batch_size, number_of_clips, number_of_patches, feature_size]) 
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True) # returns torch.Size([batch_size, number_of_clips, number_of_patches, feature_size]) 

    # Define the parameters for the test
    num_frames = 11  # Number of frames, treated as channels in this context
    feature_size = 192  # Feature size per patch
    num_channels = (num_frames-1) * feature_size  # Total number of input channels
    num_classes = 49  # Number of classes for segmentation output
    height = 244  # Assumed height for the input
    width = 244  # Assumed width for the input
    original_height = 160
    original_width = 240
    

    # Initialize the model
    model = UNet(num_channels=num_channels, n_classes=num_classes, feature_size=feature_size, bilinear=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    cum_loss = 0
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_loss = []
    val_loss = []
    for i,data in enumerate(train_loader): # BATCH_SIZE IS NUMBER OF VIDEOS
        batched_stacked_sequences, masks = data


        reshaped_data = data_preprocessing(batched_stacked_sequences, encoder, device)
        loss =train_step(reshaped_data,masks, model, encoder, criterion, optimizer, device,batch_size)
        train_loss.append(loss)
        if i >10:
            break
    torch.save(model.state_dict(), 'Future_mask_prediction.pth')
    for i,data in enumerate(valid_loader):
        batched_stacked_sequences, masks = data
        batch_size,number_of_clips,color_channels,number_of_frames,height,width = batched_stacked_sequences.shape

        reshaped_data = data_preprocessing(batched_stacked_sequences, encoder, device)
        loss, batched_semantic_mask,true_masks = val_step(reshaped_data, masks,model, encoder, criterion, device,batch_size)
        val_loss.append(loss)

        # graph the semantic mask and the true mask in one figure for the first clip
        if i == len(valid_loader)-1 or i ==0:
            fig, axs = plt.subplots(1, 2)
            mask = torch.argmax(batched_semantic_mask[0,0,:,:], dim=0)
            axs[0].imshow(mask.detach().numpy())
            axs[1].imshow(true_masks[0,-1,:,:].detach().numpy())
            plt.show()

        break


    # saved model


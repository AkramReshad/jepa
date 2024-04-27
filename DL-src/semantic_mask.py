import torch
import torch.nn as nn
from unet import UNet
from torch.utils.data import DataLoader

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

def train(data_loader, model, encoder, criterion, optimizer, device,epochs):
    model.train()
    encoder.to(device)
    for epoch in range(epochs):
        cum_loss = 0
        for data in data_loader:
            data = data.to(device)
            sequences, masks = data

            output = model(sequences)

            optimizer.zero_grad()

            loss = criterion(output,masks)
            print(loss.item())
            cum_loss += loss.item()
            loss.backward()

            optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {cum_loss/len(data_loader)}") 


if __name__ == "__main__":
    from encoder import get_encoder_model
    import yaml

    directory = '/Users/akramreshad/nyu_grad_school/2024Spring/Deep Learning/final_project/dataset/unlabeled'
    feature_size = 192  # Example value
    resolution = 224

    with open('configs/evals/vitt16.yaml', 'r') as file:
        args_eval = yaml.safe_load(file)
    encoder = get_encoder_model(args_eval,device='cpu')
    for param in encoder.parameters():
        param.requires_grad = False

    transform = make_transforms(training=False)
    dataset = VideoFrameDataset(directory, encoder, transform)

    loader = DataLoader(dataset, batch_size=10, shuffle=True) # returns torch.Size([batch_size, number_of_clips, number_of_patches, feature_size]) 

    # Define the parameters for the test
    num_frames = 11  # Number of frames, treated as channels in this context
    feature_size = 192  # Feature size per patch
    num_channels = (num_frames-1) * feature_size  # Total number of input channels
    num_classes = 49  # Number of classes for segmentation output
    height = 244  # Assumed height for the input
    width = 244  # Assumed width for the input


    # Initialize the model
    model = UNet(num_channels=num_channels, n_classes=num_classes, feature_size=feature_size, bilinear=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for data in loader: # BATCH_SIZE IS NUMBER OF VIDEOS
        batched_stacked_sequences, masks = data
        batch_size,number_of_clips,color_channels,number_of_frames,height,width = batched_stacked_sequences.shape
        batched_stacked_sequences = batched_stacked_sequences.view(batch_size*number_of_clips,color_channels,number_of_frames,height,width)
        
        batched_encoded_sequences = encoder([[batched_stacked_sequences]])[0]
        
        clip_batch_size,number_of_patches,feature_size = batched_encoded_sequences.shape

        print(f"batched_encoded_sequences shape: {batched_encoded_sequences.shape}")
        padded_batched_encoded_sequences = pad_patches(batched_encoded_sequences.view(clip_batch_size,10,number_of_patches//10,feature_size))

        print("Padded shape:", padded_batched_encoded_sequences.shape)

        # Now, reshape each clip to 10x10
        reshaped_data = padded_batched_encoded_sequences.view(batch_size * number_of_clips, 10, 10,10*feature_size)
        reshaped_data = reshaped_data.permute(0,3,1,2)
        print("Reshaped shape:", reshaped_data.shape)

        output = model(reshaped_data)


        print(f"output shape: {output.shape}")
        print(f"masks shape: {masks.shape}")
        break
    # # Run the model on the test input
    # output = model(test_input)

    # # Print the output shape to verify the correct output dimensions
    # print("Output shape:", output.shape)
    # # Expected output shape: (1, height, width, num_classes), but it will be (1, num_classes, height, width) due to pytorch conventions

    # # To match expected output shape, you can permute the dimensions if needed for comparison
    # output_permuted = output.permute(0, 2, 3, 1)  # Change from BxCxHxW to BxHxWxC
    # print("Output shape after permutation:", output_permuted.shape)


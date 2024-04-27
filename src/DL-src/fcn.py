import torch
import torch.nn as nn
import torch.optim as optim
import os
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image

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
        batch_size, num_clips, num_patches, features = x.shape

        # Reshape to treat each clip independently
        x = x.reshape(batch_size*num_clips, num_patches * features)

        x = self.layers(x)
        # Final reshape to match the original input shape
        x = x.view(batch_size, num_clips, num_patches, features)
        return x
    
class VideoFrameDataset(Dataset):
    def __init__(self, directory, encoder, transform=None):
        super().__init__()
        self.videos = []  # List to hold lists of frame paths for each video
        self.encoder = encoder
        self.window_size = 11
        for param in self.encoder.parameters():
            param.requires_grad = False

        self.transform = transform if transform is not None else transforms.ToTensor()

        # Iterate over each folder
        for video_dir in sorted(os.listdir(directory)):
            video_path = os.path.join(directory, video_dir)
            if os.path.isdir(video_path):
                # Collect all frame file paths
                image_files = sorted([os.path.join(video_path, f) for f in os.listdir(video_path) if f.endswith('.png')])
                self.videos.append(image_files)

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        video_frames = self.videos[idx]
        sequences = []

        # Generate all sequences from 0-10 up to 10-21
        for start in range(self.window_size):  # This creates sequences 0-10, 1-11, ..., 10-21
            frame_sequence = video_frames[start:start+self.window_size]  # Get 11 frames starting from 'start'
            images = [self.transform(Image.open(frame)) for frame in frame_sequence]
            
            # Encode the frames with the provided encoder
            tensor = torch.stack(images).unsqueeze(0)

            sequences.append(tensor.squeeze(0))
        stacked_sequences = torch.stack(sequences)
        stacked_sequences = stacked_sequences.permute(0, 2, 1, 3, 4)

        return stacked_sequences

def create_dataloader(directory, batch_size, shuffle=True, num_workers=0):
    dataset = VideoFrameDataset(directory)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return loader

def train(data_loader, model, encoder, criterion, optimizer, device,epochs):
    model.train()
    encoder.to(device)
    model.to(device)
    for epoch in range(epochs):
        cum_loss = 0
        for data in data_loader:
            data = data.to(device)
            data_representation = []
            for i in range(data.shape[1]):
                data_representation.append(encoder([[data[:,i,:,:]]])[0])
            data_representation = torch.stack(data_representation)
            inputs = data_representation[:, :-1, :, :]  # All sequences except the last
            labels = data_representation[:, 1:, :, :] 

            output = model(inputs)

            optimizer.zero_grad()

            loss = criterion(output,labels)
            print(loss.item())
            cum_loss += loss.item()
            loss.backward()

            optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {cum_loss/len(data_loader)}") 

# Assuming the dataset class VideoFrameDataset is defined as above
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

    normalization = ((0.485, 0.456, 0.406),
                     (0.229, 0.224, 0.225))
    transform = transforms.Compose([
            transforms.Resize(size=int(resolution * 256/224)),
            transforms.CenterCrop(size=resolution),
            transforms.ToTensor(),
            transforms.Normalize(normalization[0], normalization[1])])
    
    dataset = VideoFrameDataset(directory, encoder, transform)

    loader = DataLoader(dataset, batch_size=10, shuffle=True) # returns torch.Size([batch_size, number_of_clips, number_of_patches, feature_size]) 


    criterion = nn.L1Loss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model = DeepDenseNet(num_patches=980, feature_size=feature_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    train(loader, model,encoder, criterion, optimizer, device, 1)


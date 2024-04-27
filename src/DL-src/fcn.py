import torch
import torch.nn as nn
import torch.optim as optim
import os
from torch.utils.data import Dataset, DataLoader

class DeepDenseNet(nn.Module):
    def __init__(self):
        super(DeepDenseNet, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(192, 192),
            nn.LayerNorm(192),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(192, 192),
            nn.LayerNorm(192),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(192, 192),
            nn.LayerNorm(192),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(192, 192),
            nn.LayerNorm(192),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(192, 192),
            nn.LayerNorm(192),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(192, 192),
            nn.LayerNorm(192),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(192, 192),
            nn.LayerNorm(192),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(192, 192),
            nn.LayerNorm(192),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        batch_size, seq_len, features = x.shape
        x = x.reshape(batch_size * seq_len, features)
        x = self.layers(x)
        x = x.reshape(batch_size, seq_len, features)
        x = x.mean(dim=0, keepdim=True)
        return x
    


class VideoFrameDataset(Dataset):
    def __init__(self, directory):
        super().__init__()
        self.videos = []  # List to hold paths to videos
        self.labels = []  # Corresponding labels if necessary

        # Iterate over each folder
        for video_dir in sorted(os.listdir(directory)):
            video_path = os.path.join(directory, video_dir)
            if os.path.isdir(video_path):
                # List all tensor files for this video, sorted to maintain internal order
                tensor_files = sorted([os.path.join(video_path, f) for f in os.listdir(video_path) if f.endswith('.pt')])
                self.videos.append(tensor_files)
                self.labels.append(video_dir)  # Adjust as necessary for labeling

    def __len__(self):
        return len(self.videos)


    def __getitem__(self, idx):
        tensor_files = self.videos[idx]
        tensors = [torch.load(f) for f in tensor_files]  # Load all tensors for this video


        return tensors

def create_dataloader(directory, batch_size, shuffle=True, num_workers=0):
    dataset = VideoFrameDataset(directory)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return loader



def train_model(data_loader, model, criterion, optimizer, device,epochs):
    model.train()
    for _ in range(epochs):
        for tensors in data_loader:
            optimizer.zero_grad()
            tensors = tensors[0][0]
            cumulative_loss = 0
            loss = 0
            for i in range(len(tensors) - 1):
                input_tensor = tensors[i].to(device)
                target_tensor = tensors[i + 1].to(device)
                output_tensor = model(input_tensor)
                loss = criterion(output_tensor, target_tensor)
                cumulative_loss += loss          
            cumulative_loss.backward()
            optimizer.step()



def train_model2(data_loader, model, criterion, optimizer, device,epochs):
    model.train()
    for _ in range(epochs):
            for tensors in data_loader:
                optimizer.zero_grad()
                tensors = tensors[0][0]

                loss = 0
                for i in range(len(tensors) - 1):
                    input_tensor = tensors[i].to(device)
                    target_tensor = tensors[i + 1].to(device)
                    output_tensor = model(input_tensor)
                    loss = criterion(output_tensor, target_tensor)
                    loss.backward()
                optimizer.step()
                

def predict(data_loader,model,device):
    for tensors in data_loader:
            
            outputs = tensors[0].to(device)
            for _ in range(11):
                outputs = model(outputs)
                
            
    return outputs

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DeepDenseNet().to(device)
criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
data_path = 'test_data'
epochs = 1
# tensor_1 = torch.load("test_data/1/tensor_02.pt")
# print(tensor_1)
batch_size = 2  # Process one sequence at a time for simplicity in this example
data_loader = create_dataloader(data_path, batch_size)
# Train the model
train_model2(data_loader, model, criterion, optimizer, device, epochs)




# def inspect_data_loader(data_loader):
#     for i, (tensors, video_labels) in enumerate(data_loader):
#         print(f"Batch {i}: Video Label = {video_labels}")
#         # Optionally print tensor shapes or other properties
#         print(f"Number of tensors in this batch: {len(tensors)}")
#         if i >= 1:  # Just inspect the first couple of batches to understand the pattern
#             break

# # Example usage:
# data_loader2 = create_dataloader('test_data', batch_size=1, shuffle=True)
# inspect_data_loader(data_loader2)
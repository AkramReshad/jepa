import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from video_utils import make_transforms
from DL_utils import VideoFrameDataset


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
    

def train(data_loader, model, encoder, criterion, optimizer, device,epochs):
    model.train()
    encoder.to(device)
    model.to(device)
    for epoch in range(epochs):
        cum_loss = 0
        for data in data_loader:
            # data = data.to(device)
            data,_ = data
            data_representation = []
            print(f"batch: {len(data)}, example: {data[0].shape}")
            for i in range(data.shape[1]):
                data_representation.append(encoder([[data[:,i,:,:]]])[0])
                print(data_representation[i].shape)
                exit()
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
    
    # transform = transforms.Compose([
    #         transforms.Resize(size=int(resolution * 256/224)),
    #         transforms.CenterCrop(size=resolution),
    #         transforms.ToTensor(),
    #         transforms.Normalize(normalization[0], normalization[1])])

    transform = make_transforms(training=False)

    dataset = VideoFrameDataset(directory, encoder, transform)

    loader = DataLoader(dataset, batch_size=1, shuffle=True) # returns torch.Size([batch_size, number_of_clips, number_of_patches, feature_size]) 

    criterion = nn.L1Loss()
    # criterion = nn.MSELoss()
    device = torch.device("cpu")
    print(device)
    model = DeepDenseNet(num_patches=980, feature_size=feature_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    train(loader, model,encoder, criterion, optimizer, device, 1)


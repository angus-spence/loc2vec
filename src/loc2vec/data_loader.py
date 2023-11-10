import os

from torch import nn
import torch, torchvision
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
path = r'C:\Users\Malcolm\Documents\Scripts\loc2vec\src\loc2vec\test_data'

data = torch.stack([torchvision.io.read_image(os.path.join(path, os.listdir(path)[i]))[:3, :, :] for i in range(len(os.listdir(path)))]).type(torch.float).to(device)
print(data)

class Network(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model = nn.Sequential(
            nn.Dropout2d(0.5),
            nn.Conv2d(3, 64, 1, stride=1, padding=0),
            nn.Conv2d(64, 64, 1, stride=1, padding=0),
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, stride=2, padding=0),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.MaxPool2d(2, stride=2, padding=0),
            nn.Conv2d(64, 32, 3, stride=1, padding=1),
            nn.MaxPool2d(2, stride=2, padding=0),
            nn.Conv2d(32, 32, 3, stride=1, padding=1),
        )

    def forward(self, x):
        return self.model(x)
        
model = Network().to(device)

out = model(data)
print(out)
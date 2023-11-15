from loc2vec.data_loader import Data_Loader

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

loader = Data_Loader(
    x_path=r'\\SWPUKNAS2201\dst\Data\Projects\RSIB\data\raw\loc2vec',
    x_pos_path=r'\\SWPUKNAS2201\dst\Data\Projects\RSIB\data\raw\loc2vec',
    x_neg_path=r'\\SWPUKNAS2201\dst\Data\Projects\RSIB\data\raw\loc2vec',
    batch_size=60,
    shuffle=False
)

loader.load_from_dirs()

quit()

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
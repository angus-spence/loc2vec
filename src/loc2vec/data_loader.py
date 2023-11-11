import os
import time
from dataclasses import dataclass

from torch import nn
import torch, torchvision
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
path = r'C:\Users\Malcolm\Documents\Scripts\loc2vec\src\loc2vec\test_data'

#data = torch.stack([torchvision.io.read_image(os.path.join(path, os.listdir(path)[i]))[:3, :, :] for i in range(len(os.listdir(path)))]).type(torch.float).to(device)
#data = torch.utils.data.DataLoader(data, batch_size=120, shuffle=False)

@dataclass
class Data_Loader():
    x: torch.Tensor
    x_pos: torch.Tensor
    x_neg: torch.Tensor
    device: str
    batch_size: int
    shuffle: bool
    tt_split: float = 0.8

    def __post_init__(self):
        print(f'{os.get_terminal_size()[0] * "-"}\nData Loader {time.strftime(time.time(),"%H:%M:%S")}\n{os.get_terminal_size()[0] * "-"}')
        self._check_shape()
        self._check_dtype()
        self._check_device()

    def _check_shape(self) -> None:
        """
        Checks that all tensors have the same shape.
        """
        for x, y, z in zip(self.x, self.x_pos, self.x_neg):
            c, h, w = x.shape, y.shape, z.shape
            if c != h or c != w or h != w:
                raise ValueError(f'All tensors must have the same shape. Got {c}, {h}, {w}.') 

    def _check_dtype(self) -> bool:
        """
        Checks that all tensors have the same dtype.
        """
        dt_c = True
        for x, y, z in zip(self.x, self.x_pos, self.x_neg):
            c, h, w = x.dtype, y.dtype, z.dtype
            if c != h or c != w or h != w:
                dt_c = False
        if not dt_c:
            try:
                for x, y, z in zip(self.x, self.x_pos, self.x_neg):
                    x, y, z = x.type(self.dtype), y.type(self.dtype), z.type(self.dtype)
            except:
                raise ValueError(f'All tensors must have the same dtype. Got {c}, {h}, {w}.')
            
    def _check_device(self) -> bool:
        """
        Checks that all tensors are on the same device.
        """
        b = True
        for x, y, z in zip(self.x, self.x_pos, self.x_neg):
            c, h, w = x.device, y.device, z.device
            if c != h or c != w or h != w:
                b = False
        if not b:
            try:
                for x, y, z in zip(self.x, self.x_pos, self.x_neg):
                    x, y, z = x.to(self.device), y.to(self.device), z.to(self.device)
            except:
                raise ValueError(f'All tensors must be on the same device. Got {c}, {h}, {w}.')
            
    def _device_
            
    

    def _check_device(self):
        """
        Checks that all tensors are on the same device.
        """
        for x, y, z in zip(self.x, self.x_pos, self.x_neg):
            if x.device != y.device or x.device != z.device or y.device != z.device:
                raise ValueError(f'All tensors must be on the same device. Got {x.device}, {y.device}, {z.device}.')
            

    def _train_test_split(self):
        """
        """
        return

    def __post_init__(self):
        return

    #self.data = torch.stack([torchvision.io.read_image(os.path.join(path, os.listdir(path)[i]))[:3, :, :] for i in range(len(os.listdir(path)))]).type(torch.float).to(device)
    
    def load(self):
        return torch.utils.data.DataLoader(self.data, batch_size=self.batch_size, shuffle=self.shuffle)

    def _get_memory(self):
        return torch.cuda.memory_allocated(self.device)
    


data = Loc2Vec_Dataset(device, 120, False)  

print(next(iter(data)))

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
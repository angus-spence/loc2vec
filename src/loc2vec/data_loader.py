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
    """
    Object for loading train and test data for the loc2vec network

    Args:
        x_path: path of directory with x anchor rasters
        x_pos_path: path of directory with x positive anchor rasters
        y_neg_path: path of directory with x negative anchor rasters
        batch_size: batch size to use in dataloaded
        channels: number of channels for each anchor index
        shuffle: shuffle indecies in dataloader
        tt_split: percentage of training samples 
    """
    x_path: str
    x_pos_path: str
    x_neg_path: str
    batch_size: int
    channels: int
    shuffle: bool = False
    tt_split: float = 0.8

    def __post_init__(self):
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            self.cuda = True
        else: self.device = torch.device('cpu')

        t = time.gmtime(time.time())
        print(f'Data Loader {t[3]}:{t[4]}:{t[5]} Device: {str(self.device).upper()}\n{os.get_terminal_size()[0] * "-"}')

    def load(self) -> [torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._check_path():
            print(f'Loading images from:\n   -> {self.x_path}\n   -> {self.x_pos_path}\n   -> {self.x_neg_path}')
            self.x, self.x_pos, self.x_neg = (torch.stack([torchvision.io.read_image(os.path.join(j, os.listdir(j)[i]))[:3, :, :] for i in range(len(os.listdir(j)))]).type(torch.float).to(device) for j in [self.x_path, self.x_pos_path, self.x_neg_path])
        if self.cuda: print(f'   -> Memory: {round(self._get_memory() / 1e9, 3)} GB')

    def _get_samples(self) -> int:
        if self._check_path(): self.samples = len(os.listdir(self.x_path))

    def _check_path(self) -> bool:
        """
        Checks that all paths are valid.
        """
        for path in [self.x_path, self.x_pos_path, self.x_neg_path]:
            if not os.path.isdir(path):
                raise ValueError(f'Path {path} is not a valid directory.')
            
        if len(os.listdir(self.x_path)) != len(os.listdir(self.x_pos_path)) or len(os.listdir(self.x_path)) != len(os.listdir(self.x_neg_path)):
            raise ValueError(f'All paths must have the same number of files. Got {len(os.listdir(self.x_path))}, {len(os.listdir(self.x_pos_path))}, {len(os.listdir(self.x_neg_path))}.')
        return True

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
                    x, y, z = x.type(torch.float), y.type(self.dtype), z.type(self.dtype)
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
            
    def _train_test_split(self):
        """
        """
        return

    def _get_memory(self):
        return torch.cuda.memory_allocated(self.device)

    #self.data = torch.stack([torchvision.io.read_image(os.path.join(path, os.listdir(path)[i]))[:3, :, :] for i in range(len(os.listdir(path)))]).type(torch.float).to(device)
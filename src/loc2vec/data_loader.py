import os
import time
import enum
from itertools import groupby
from dataclasses import dataclass

from tqdm import tqdm
from torch import nn
import torch
import torchvision as tv
import matplotlib.pyplot as plt


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
path = r'C:\Users\Malcolm\Documents\Scripts\loc2vec\src\loc2vec\test_data'

#data = torch.stack([torchvision.io.read_image(os.path.join(path, os.listdir(path)[i]))[:3, :, :] for i in range(len(os.listdir(path)))]).type(torch.float).to(device)
#data = torch.utils.data.DataLoader(data, batch_size=120, shuffle=False)

@dataclass
class Data_Loader():
    """
    Object for loading train and test data to tensor

    Args:
        x_path: str path or array-like directory with x anchor rasters
        x_pos_path: path of directory with x positive anchor rasters
        y_neg_path: path of directory with x negative anchor rasters
        batch_size: batch size to use in dataloaded
        channels: number of channels for each anchor index
        shuffle: shuffle indecies in dataloader
        tt_split: percentage of training samples 
    """
    x_path: str
    x_pos_path: str
    batch_size: int
    x_neg_path: str = False
    shuffle: bool = False
    tt_split: float = 0.8

    def __post_init__(self):
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            self.cuda = True
        else: self.device = torch.device('cpu')
        if self.x_neg_path: self.data_dirs = [self.x_path, self.x_pos_path, self.x_neg_path]
        else: self.data_dirs = [self.x_path, self.x_pos_path] 
        self.channels = self._get_channels()

    def load_from_dirs(self) -> [torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Loads PNG to tensors
        """
        #TODO: UPDATE THIS SO THAT IT EXCEPTS AN ARRAY-LIKE OBJECT FOR PATH
            # 
        t = time.gmtime(time.time())
        print(f'Data Loader {t[3]}:{t[4]}:{t[5]} Device: {str(self.device).upper()}\n{os.get_terminal_size()[0] * "-"}')
        print(f'Loading to torch:')

        steps = (self._get_channels() * self._get_samples()) / len(self.data_dirs)
        files = os.walk(self.data_dirs[0])[2]
        print(files)
        quit()
        features = []


        def c(i, steps):
            return


        for path_i in self.data_dirs:
            print(f'   -> Loading from {path_i}')
            features.append([torch.stack([tv.io.read_image(file) and c(file) for root, dir, file in os.walk(path_i)])])
            
        #if self._check_path():
        #    print(f'Loading images from:\n   -> {self.x_path}\n   -> {self.x_pos_path}\n   -> {self.x_neg_path}')
        #    self.x, self.x_pos, self.x_neg = (torch.stack([torchvision.io.read_image(os.path.join(j, os.listdir(j)[i]))[:3, :, :] for i in range(len(os.listdir(j)))]).type(torch.float).to(device) for j in [self.x_path, self.x_pos_path, self.x_neg_path])
        #if self.cuda: print(f'   -> Memory: {round(self._get_memory() / 1e9, 3)} GB')
        
        return

    def _get_samples(self) -> int:
        """
        Return number of samples
        """    
        if self._check_samples()[0]: return self._check_samples()[1][0]
        else: raise TypeError(f'Must have equal samples in channel. Check samples in data directories. Got {self._check_samples()[1]}')

    def _check_samples(self):
        """
        Evaluates if sample indicies are equal across paths
        """
        t = []
        for dir in [f for dir in self._get_locs() for f in dir]:
            t.append((len(os.listdir(dir))))
        t_g = groupby(t)
        return next(t_g, True) and not next(t_g, False), t

    def _get_locs(self) -> list:
        """
        Return paths of directories for images
        """
        dirs_arr = []
        for path_i in self.data_dirs:
            if self._check_path_types()[self.data_dirs.index(path_i)] == str:
                for root, dirs, files in os.walk(path_i):
                    if dirs: dirs_arr.append([str(os.path.join(path_i, dir)) for dir in dirs])
            else: dirs_arr.append([str(os.path.join(path_i, i)) for i in self.data_dirs[self.data_dirs.index(path_i)]])
        return dirs_arr

    def _check_path_types(self) -> list:
        """
        Return the path types of each input
        """
        types = []
        for path in self.data_dirs:
            types.append(type(path))
        return types

    def _data_struct(self) -> dict:
        """
        Returns the scructure of data withn paths as array
        """
        struct = {}
        for path in self.data_dirs:
            files_ar = []
            for root, dirs, files in os.walk(path):
                files_ar.append(len(files))
                struct.update({str(path): files_ar})            
        return struct

    def _check_file_types(self):
        """
        Evaluate if all samples are PNG
        """
        
    def _check_channels(self) -> bool:
        """
        Evaluate if all inputs have equal channels
        """
        if self._check_paths():
            c = [len(os.listdir(i)) for i in [i for s in self._get_locs() for i in s]]
        c_g = groupby(c)
        return next(c_g, True) and not next(c_g, False), c[0]

    def _get_channels(self) -> int:
        return self._check_channels()[1]

    def _check_paths(self):
        """
        Evaluates if all paths exist
        """
        if [[os.path.isdir(i) for i in self.data_dirs]] == [[True]*len(self.data_dirs)]: return True
        else: raise ValueError(f'Input paths do not exist: got {[(os.path.isdir(i), i) for i in self.data_dirs]}')

    def _check_shape(self) -> None:
        """
        Checks that all tensors have the same shape.
        """
        for x, y, z in zip(self.x, self.x_pos, self.x_neg):
            c, h, w = x.shape, y.shape, z.shape
            if c != h or c != w or h != w:
                raise ValueError(f'All tensors must have the same shape. Got {c}, {h}, {w}.') 

    def _check_dtype(self, x_i: torch.tensor, x_pos: torch.tensor, x_neg: torch.tensor) -> bool:
        """
        Checks that all tensors have the same dtype.

        Parameters
        ----------
        x_i: torch.tensor
            tensor for anchor
        x_pos: torch.tensor
            tensor for positive anchor
        x_neg: torch.tensor
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
        b = False
        for x, y, z in zip(self.x, self.x_pos, self.x_neg):
            c, h, w = x.device, y.device, z.device
            if c != h or c != w or h != w:
                return False
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
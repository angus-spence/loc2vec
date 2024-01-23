from loc2vec.loc2vec_nn import Network
from loc2vec.utils import Config
from loc2vec.optim import batch_optimiser

import os
import random
from enum import Enum
from typing import Union
from itertools import groupby, chain
from dataclasses import dataclass

from tqdm import tqdm
import torch
import torchvision as tv

class Combs(Enum):
    ID = 1
    VALUES = 2

@dataclass
class Data_Loader:     
    """
    Object for loading data to tensor

    Parameters
    ----------
    x_path: [str, tuple, list]
        String or array-like object of strings of directory paths to x data
    x_pos_path: [str, tuple, list]
        String or array-like object of strings of directory paths to positive anchor data
    comb_filter: bool
        Comb input files for non-mataching rasters
    train_tensory_directory: str
        Path to directory where train tensors should be saved
    batch_size: int
        Batch size for tensors
    sample_limit: int
        Limit number of samples for model
    x_neg_path: [str, tuple, list]
        String or array-like object of strings for directory paths to negative anchor data
    shuffle: bool
        Boolean for if data indicies should be shuffeled
    """
    x_path: str
    x_pos_path: str
    comb_filter: bool = True
    batch_size: int = None
    sample_limit: int = None
    x_neg_path: str = None
    shuffle: bool = False
    paths: list = None
    in_channels: int = None
    _batch_index: int = 0
    _iter_index: int = 0
    _s: int = 0
    _e: int = 0

    def __post_init__(self):
        """
        Post-init for dataloader:
            - Identifies and assigns pytorch device
            - Evaluates data paths
            - Evaluates optimum batch size if self.batch_size not specified
            - 
        """
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            self.cuda = True
        else: self.device = torch.device('cpu')
        
        if self.x_neg_path: self.data_dirs = [self.x_path, self.x_pos_path, self.x_neg_path]
        else: self.data_dirs = [self.x_path, self.x_pos_path]
        
        print(f'   -> DEVICE: {self.device}')

        self.in_channels = self._image_shape()[0] * self._get_channels()

        if not self.batch_size:
            model = Network(in_channels=self.in_channels)
            print(f'IMAGE SHAPE: {self._image_shape()} CHANNELS: {self._get_channels()} SAMPLES: {self._get_samples()}') 
            self.batch_size = batch_optimiser(model, (self._image_shape()[0] * self._get_channels(), *self._image_shape()[1:]), self._get_samples(), num_iterations=20)
            self._e = self.batch_size
            del model

        self.batches = (len(self) - self._batch_dropout()) // self.batch_size

    def __len__(self):
        return len(self._get_data_files())//len(self.data_dirs)
    
    def __call__(self, index) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        """
        Returns tensor for (o) anchor, (+) anchor and (-) anchor for specific sample index
        """
        return tv.io.read_image(self._get_data_files()[0][index])[:3,:,:]

    def __iter__(self):
        return self

    def __next__(self):
        """
        Returns tensor of (o) anchor, (+) anchor and (-) anchor in batch iterator

        Returns
        -------
        anchors: tuple
            tuple of tensor objects for all anchors
        """
        if self._iter_index < len(self) // self.batch_size:
            self._iter_index += self.batch_size
            path = self._get_data_files()
            if not self.x_neg_path: x_neg = random.sample(path, len(path))[:len(self)]
            else: x_neg = path[len(self)*2:]
            x = path[:len(self)]
            x_pos = path[len(self):len(self)*2]

            x_out = x[self._s:self._e]
            x_pos_out = x_pos[self._s:self._e]
            x_neg_out = x_neg[self._s:self._e]
            self._s += self.batch_size
            self._e += self.batch_size

            return self._tensor_stack(x_out), self._tensor_stack(x_pos_out), self._tensor_stack(x_neg_out)
        else:
            self._iter_index = 0
            self._s = 0
            self._e = self.batch_size

            path = self._get_data_files()
            if self.comb_filter:
                path = [self.comb_filter(i) for i in path]
            if not self.x_neg_path: x_neg = random.sample(path, len(path))[:len(self)]
            else: x_neg = path[len(self)*2:] #TODO: Triplet Miner here?
            x = path[:len(self)]
            x_pos = path[len(self):len(self)*2]

            x_out = x[self._s:self._e]
            x_pos_out = x_pos[self._s:self._e]
            x_neg_out = x_neg[self._s:self._e]
            self._s += self.batch_size
            self._e += self.batch_size

            return self._tensor_stack(x_out), self._tensor_stack(x_pos_out), self._tensor_stack(x_neg_out)

    def __reverse__(self):
        self._batch_index -= self.batch_size
        return self

    def _batch_dropout(self):
        """
        Evaluates number of indicies to drop to ensure batch size is modulo of total indicies
        
        Returns
        -------
        batch_dropout: int
            Number of indicies to drop
        """
        return len(self) - (len(self) // self.batch_size)

    def _tensor_stack(self, batch: list) -> torch.Tensor:
        """
        Iterates over a list of PNG paths, converts images to a tensor and stacks to form (batch_size, channels, W, H) dimension Tensor

        Parameters
        ----------
        batch: list
            list of image paths in a batch to be converted to Tensor and then stacked

        Returns
        -------
        output: torch.Tensor
            Tensor of model input data; this can be for (o) anchor, (+) anchor or (-) anchor sets
        """
        batches = []
        for channel in batch:
            channels = []
            for img in channel:
                channels.append(tv.io.read_image(img)[:3,:,:].type(torch.float).to(self.device)) 
            t1 = torch.cat(channels)
            batches.append(t1)
        return torch.stack(batches)

    def _check_batch_size():
        return

    def _get_data_files(self) -> list:
        """
        Evaluate paths for all data inputs and returns to list

        Returns
        -------
        comp_f: list
            List of lists containing structured paths for all data inputs
        """
        if self.paths: return self.paths
        else:
            self._check_samples()
            comp_f = []
            for path_i in self.data_dirs:  
                _comp = []
                for root, dirs, files in os.walk(path_i):
                    if files: _comp.append(files)
                for j in tqdm(range(len(_comp[0])), desc=f"BUILDING PATHS FOR {str(path_i).upper()}"):
                    comp_f.append([os.path.join(path_i, os.listdir(path_i)[i], _comp[i][j]) for i in range(len(_comp))])
            self.paths = comp_f
            return comp_f

    def _comb_filter(self, anchor_files: list, comb_type: Combs) -> list:
        """
        Parse through input data files and identify non-common
        rasters. This can be done through a file ID or through
        raster channel values.

        Parameters
        ----------
        anchor_files: list
            List of files in anchor
        
        comb_type: Comb
            Method used to comb input files in data loader

        Returns
        -------
        anchor_files: list
            Filtered list of files in anchor
        """
        if comb_type == Combs.ID:
            cfg = Config()
            flt = []
            o, p, n = [], [], []
            for path in (cfg.x_path, cfg.x_pos_path, cfg.x_neg_path): 
                for root, dirs, files in os.walk(path):
                    flt = flt.append(chain.from_iterable(files))
            fmax = max(flt.count(i) for i in flt)
            fname = [i for i in flt if flt.count(i) < fmax]
            fcomb = []
            for channel in anchor_files:
                fcomb.append([i for i in channel if i not in fname])
            return fcomb
        elif comb_type == Combs.VALUES:
            # TODO: DO THIS BUT WILL BE A WHOLE NEW SCRIPT TO ACHEIVE
            #       WILL STAY WITH ID FOR NOW
            return 
        else: raise ValueError("comb_type out of range")

    def _force_cudnn_init(self):
        s = 32
        torch.nn.functional.conv2d(torch.zeros(s, s, s, s, device=self.device), torch.zeros(s, s, s, s, device=self.device))
        torch.cuda.empty_cache()

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
        for path in self.data_dirs:
            for root, dir, file in os.walk(path):
                for i in dir: 
                    t.append({dir[dir.index(i)]: len(os.listdir(os.path.join(path, i)))})
        print(t)
        return all(x == t[0] for x in t), t

    def _get_locs(self) -> list:
        """
        Return paths of directories for images
        """
        #TODO: THIS DOES NOT WORK IF THERE IS ONLY ONE CHANNEL
        #       NEED TO CHECK IF CHANGES MADE 15/11/23 : 22:13 WORK FOR MULTIPLE DIRS
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
        #TODO: THIS DOES NOT WORK IF THERE IS ONLY ONE CHANNEL
        #       PROBABLY BREAKS MOST THINGS IF THERE IS ONLY ONE CHANNEL    
        if self._check_paths():
            c = []
            for path_i in self.data_dirs:
                for root, dirs, files in os.walk(path_i):
                    if dirs: c.append(len(dirs))
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

    def _check_shape(self, x_i: torch.Tensor, x_pos: torch.Tensor, x_neg: torch.Tensor) -> bool:
        """
        Checks that all tensors have the same shape.

        Parameters
        ----------
        x_i: torch.tensor
            Tensor for anchor
        x_pos: torch.tensor
            Tensor for (+) anchor
        x_neg: torch.tensor
            Tensor for (-) anchor
        
        Returns
        -------
        bool
            True if all tensors have the same shape
        """
        for x, y, z in zip(x_i, x_pos, x_neg):
            c, h, w = x.shape, y.shape, z.shape
            if c != h or c != w or h != w:
                raise ValueError(f'All tensors must have the same shape. Got {c}, {h}, {w}.') 

    def _image_shape(self) -> tuple:
        """
        Returns
        -------
        image_shape: tuple
            tuple of D x W x H (channels, width, height)
        """
        return tuple(tv.io.read_image(self._get_data_files()[0][0])[:3,:,:].shape) 

    def _check_dtype(self, x_i: torch.Tensor, x_pos: torch.Tensor, x_neg: torch.Tensor) -> bool:
        """
        Checks that all tensors have the same dtype.

        Parameters
        ----------
        x_i: torch.tensor
            Tensor for anchor
        x_pos: torch.tensor
            Tensor for (+) anchor
        x_neg: torch.tensor
            Tensor for (-) anchor

        Returns
        -------
        bool
            True if all tensors have the same dtype
        """
        dt_c = True
        for x, y, z in zip(x_i, x_pos, x_neg):
            x1, x2, x3 = x.dtype, y.dtype, z.dtype
            if x1 != x2 or x3 != x1 or x2 != x3:
                dt_c = False
        if not dt_c:
            try:
                for x, y, z in zip(x_i, x_pos, x_neg):
                    x, y, z = x.type(torch.float), y.type(torch.float), z.type(torch.float)
            except:
                raise ValueError(f'All tensors must have the same dtype. Got {x1}, {x2}, {x3}.')
            
    def _check_device(self, x_i: torch.Tensor, x_pos: torch.Tensor, x_neg: torch.Tensor) -> bool:
        """
        Checks that all tensors are on the same device.

        Parameters
        ----------
        x_i: torch.tensor
            Tensor for anchor
        x_pos: torch.tensor
            Tensor for (+) anchor
        x_neg: torch.tensor
            Tensor for (-) anchor
        
        Returns
        -------
        bool
            True if all tensors are on the same device
        """
        b = False
        for x, y, z in zip(x_i, x_pos, x_neg):
            c, h, w = x.device, y.device, z.device
            if c != h or c != w or h != w:
                return False
        if not b:
            try:
                for x, y, z in zip(self.x, self.x_pos, self.x_neg):
                    x, y, z = x.to(self.device), y.to(self.device), z.to(self.device)
            except:
                raise ValueError(f'All tensors must be on the same device. Got {c}, {h}, {w}.')

    @staticmethod
    def _chunk(list, n):
        """
        """
        for i in range(0, len(list), n): 
            yield list[i:i + n]

    def _get_memory(self):
        return torch.cuda.memory_allocated(self.device)

    def _tensory_memory(self, tensor: torch.Tensor) -> float:
        """
        """
        return tensor.element_size() * tensor.nelement()
    
    def _theoretical_memory(self, dtype: torch, shape: tuple) -> int:
        """
        Evaluate the theoretical memory requirement of a tensor given it's dtype and shape

        Parameters
        ----------
        dtype: torch
            Data type of model input tensor
        shape: tuple
            Shape of the input tensor as a tuple
        
        
        Returns
        -------
        memory: int
            theoretical memory requirement in bytes
        """

        dtypes = {
            torch.float: 32,
            torch.float32: 32,
            torch.FloatTensor: 32,
            torch.float64: 64,
            torch.double: 64,
            torch.DoubleTensor: 64,
            torch.float16: 16,
            torch.half: 16,
            torch.HalfTensor: 16,
            torch.uint8: 8,
            torch.ByteTensor: 8
        }

        bm = torch.rand(1,1).type(dtype)
        bm = self._tensory_memory(bm)

        size = 1
        for ele in shape:
            size *= ele
        
        return size * (dtypes[dtype] * 8), bm * size
    
@dataclass
class SlimLoader:
    img_dir: str
    shuffle: bool
    batch_size: int
    device: str

    def __post_init__(self):
        self.channels, self.images, self.dimensions = self._input_spec()
        self.idx, self.s, self.e = 0, 0, self.batch_size

    def __len__(self):
        return self.images
    
    def __iter__(self):
        return self

    def __next__(self) -> torch.Tensor:
        """
        """
        if self.idx < len(self) // self.batch_size:
            self.idx += self.batch_size
            path = self._get_paths()
            x = path[:len(self)]
            x_out = x[self.s:self.e]
            self.s += self.batch_size 
            self.e += self.batch_size
        return self._batch_to_tensor(x_out)

    def _input_spec(self) -> Union[int, int, tuple]:
        """
        """
        channels = []
        no_files = []
        for root, dirs, files, in os.walk(self.img_dir):
            if dirs: channels.append(len(dirs))
            if files: no_files.append(len(files))
        return channels[0], no_files[0], tuple(tv.io.read_image(self._get_paths()[0][0])[:3,:,:].shape)
    
    def _get_paths(self):
        """
        """
        try:
            return self.paths
        except:
            file_names = []
            paths = []
            for root, dirs, files in os.walk(self.img_dir):
                if files: file_names.append(files)
            for file_idx in tqdm(range(len(file_names[0])), desc=f'BUILDING PATHS'):
                paths.append([os.path.join(self.img_dir, os.listdir(self.img_dir)[i], file_names[i][file_idx]) for i in range(len(file_names))])
            self.paths = paths
        return self.paths
    
    def _batch_to_tensor(self, batch: list) -> torch.Tensor:
        """
        """
        batches = []
        for channel in batch:
            channels = []
            for img in channel:
                channels.append(tv.io.read_image(img)[:3,:,:].type(torch.float).to(self.device))
            batch_tensor = torch.cat(channels)
            batches.append(batch_tensor)
        return torch.stack(batches).to(self.device)
from loc2vec.loc2vec_nn import Network, TripletLossFunction as tlf

import os
import random
from itertools import groupby
from dataclasses import dataclass

from tqdm import tqdm
from torch import nn
import torch
import torchvision as tv

@dataclass
class Data_Loader():     
    """
    Object for loading data to tensor

    Parameters
    ----------
    x_path: [str, tuple, list]
        String or array-like object of strings of directory paths to x data
    x_pos_path: [str, tuple, list]
        String or array-like object of strings of directory paths to positive anchor data
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
    batch_size: int = None
    sample_limit: int = None
    x_neg_path: str = None
    shuffle: bool = False
    paths: list = None
    in_channels: int = None
    _batch_index: int = 0
    _iter_index: int = 0


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
            print(f'IMAGE SHAPE: {self._image_shape()} CHANNELS: {self._get_channels()} SAMPLES: {self._get_samples()}')
            model = Network(in_channels=self.in_channels) 
            self.batch_size = self._optim_batch(model, (self._image_shape()[0] * self._get_channels(), *self._image_shape()[1:]), self._get_samples(), num_iterations=20)
            del model

        self.batches = (len(self) - self._batch_dropout()) // self.batch_size

    def __len__(self):
        return len(self._get_data_files())//len(self.data_dirs)
    
    def __call__(self, batch_index) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        """
        Returns tensor for (o) anchor, (+) anchor and (-) anchor for specific batch index
        """
    
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
            if not self.x_neg_path: x_neg = random.shuffle(path[:len(self)])
            else: x_neg = path[len(self)*2:]
            x_pos = path[:len(self)]
            x_neg = path[len(self):len(self)*2]
            return self._tensor_stack(x_pos), self._tensor_stack(x_neg), self._tensor_stack(x_neg)
        else:
            self._iter_index = 0
            raise StopIteration

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

    def _tensor_stack(self, files) -> torch.Tensor:
        """
        Iterates over a list of PNG paths, coverts and stacks Tensors

        Returns
        -------
        output: torch.Tensor
            Tensor of model input data; this can be for (o) anchor, (+) anchor or (-) anchor sets
        """
        data_tensors = []
        counter = 0
        for channel in files:
            for index in range(len(channel)):
                counter += 1
                print(f'LOADED: {counter}/{self.batch_size*self.in_channels} @ {round(torch.cuda.memory_allocated()*1e-9, 4)}GB', end='\r') 
                data_tensors.append(tv.io.read_image(channel[index])[:3,:,:].type(torch.float).to(self.device)) 
        return torch.stack(data_tensors)

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
            comp_f = []
            for path_i in self.data_dirs:  
                _comp = []
                for root, dirs, files in os.walk(path_i):
                    if files: _comp.append(files)
                for j in tqdm(range(len(_comp[0])), desc=f"BUILDING PATHS FOR {str(path_i).upper()}"):
                    comp_f.append([os.path.join(path_i,os.listdir(path_i)[i],_comp[i][j]) for i in range(len(_comp))])
            print(f'   -> INPUT SHAPE [{len(comp_f)}, {len(comp_f[0])}]')
            self.paths = comp_f
            return comp_f

    def _optim_batch(self, model: nn.Module, input_shape: tuple, samples: int, max_batch_size: int = None, num_iterations: int = 5, headroom_bias: int = None) -> int:
        """
        Evaluates optimum batch size if self.batch_size not specified
        
        Parameters
        ----------
        model: nn.Module
            Nerual network model
        input_shape: tuple[int, ...]
            Shape of single data index
        samples: int
            Number of samples
        max_batch_size: int = None
            Maximum batch to evaluate
        num_iterations: int
            Times to iterate through the model in evaluation
        headroom_bias: int
            byte headroom required
        
        Returns
        -------
        batch_size: int
            Optimum batch size
        """
        #TODO: THIS NEEDS SOME DEBUGGING -> USING A TRY IS NOT A GOOD IDEA HERE, IDEALLY WE CAN ISOLATE MEMORY
        #       EXCEPTIONS AS THIS ONLY WORKS IF MEMORY EXCEPTIONS ARE THE ONLY EXCEPTIONS
        #
        #       THIS IS SLIGHTLY BETTER NOW BUT STILL CRAP
        model.to(self.device)
        model.train(True)
        lf = tlf()
        optimiser = torch.optim.Adam(model.parameters())
        batch_size = 2
        while True:
            if max_batch_size is not None and batch_size >= max_batch_size:
                batch_size = max_batch_size
                break
            if batch_size >= samples:
                batch_size = batch_size // 2
                break
            try:
                for _ in tqdm(range(num_iterations), desc="Evaluating optimum batch size"):
                    anchor_i = torch.rand(*(batch_size, *input_shape), device=self.device, dtype=torch.float)
                    anchor_pos = torch.rand(*(batch_size, *input_shape), device=self.device, dtype=torch.float)
                    anchor_neg = torch.rand(*(batch_size, *input_shape), device=self.device, dtype=torch.float)
                    outputs = model(anchor_i)
                    loss = lf(outputs, model(anchor_pos), model(anchor_neg))
                    loss.backward()
                    optimiser.step()
                    optimiser.zero_grad()
                    batch_size *= 2
            except RuntimeError as e:
                if str(e)[:18] == "CUDA out of memory": 
                    batch_size //= 2
                    break
                else:
                    print(e)
                    quit()

        del model, optimiser
        torch.cuda.empty_cache()
        print(f'Optimum batch size: {batch_size}')
        return batch_size

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
        Return the W x H dimension of samples

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
    #self.data = torch.stack([torchvision.io.read_image(os.path.join(path, os.listdir(path)[i]))[:3, :, :] for i in range(len(os.listdir(path)))]).type(torch.float).to(device)
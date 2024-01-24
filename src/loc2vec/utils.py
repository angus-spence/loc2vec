import os
import tomllib
from dataclasses import dataclass
from typing import Union

import torch, torchvision
import numpy as np

@dataclass
class Channel_Validation:
    """
    Channel validation for data dirs
    """
    anchor_i_path: list
    anchor_p_path: list
    anchor_n_path: list

    def __post_init__(self):
        self.dirs = (self.anchor_i_path,
                     self.anchor_n_path,
                     self.anchor_p_path)
        


    def __len__(self):
        return

    def dimensions(self):
        no_channels = []
        no_images = []
        for path in (self.dirs):
            for root, dirs, files in os.walk(path):
                if dirs: no_channels.append(len(dirs))
                if files: no_images.append(len(files))
                if no_channels == 1:
                    smpl = os.path.join(path, dirs[0], files[0])
                else:
                    smpl = os.path.join(path, dirs[0], files[0][0])
        self.channels, self.samples, self.image_dimensions = no_channels, no_images, tuple(torchvision.io.read_image(smpl))
        return no_channels, no_images, tuple(torchvision.io.read_image(smpl))

    @property
    def channels(self):
        return self.dimensions[0]

    @property
    def samples(self):
        return self.dimensions[1]

    @property
    def image_dimensions(self):
        return self.dimensions[2]

    def squeeze(self, destructive: bool) -> Union[list, list, list]:
        """
        """



@dataclass
class Config:
    src: str = "./src/loc2vec/config.toml"

    def __post_init__(self):
        with open("./src/loc2vec/config.toml", "rb") as f:
            d = tomllib.load(f)
        self.epochs, self.lr, self.channels = d['training_parameters'].values()
        self.drive, self.anchor_i_path, self.anchor_pos_path, self.anchor_neg_path = d['paths'].values()

def visualise_tensor(tensor: torch.Tensor, ch:int=0, allkernels:bool=False, nrow:int=0, padding:int=1) -> np.ndarray:
    """
    Returns an array for visualising a tensor
    
    Parameters
    ----------
    tensor: torch.Tensor
        Tensor to visualise
    ch: int
        Number of channels
    allkernels: bool
        View all kernels
    nrow: int
        Number of rows in visualisation
    padding: ints
        Padding between kernels in visualisation
    """
    n,c,w,h = tensor.shape

    if allkernels: tensor = tensor.view(n*c, -1, w, h)
    elif c != 3: tensor = tensor[:,ch,:,:].unsqueeze(dim=1)

    rows = np.min((tensor.shape[0]) // nrow + 1, 64)
    grid = torchvision.utils.make_grid(tensor, nrow=nrow, normalize=True, padding=padding)
    return grid.numpy().transpose((1,2,0))

def embedding_sequencing(embeddings: torch.Tensor) -> np.ndarray:
    """
    Returns embedding stamps at each epoch to visualise embedding space over time
    
    Parameters
    ----------
    embeddings: torch.Tensor
        torch tensor of vector embeddings at given epoch
    """
    #TODO: IMPLEMENT THIS
    embs = embeddings.numpy()
    
def gpu_compute_memory(model: torch.nn.Module) -> float:
    """
    Returns the memory required by a model

    Parameters
    ----------
    model: torch.nn.Module
        Model to evaluate memory requirement
    """
    mem_params = sum([param.nelement()*param.element_size() for param in model.parameters()])
    mem_bufs = sum([buf.nelement()*buf.element_size() for buf in model.buffers()])
    return mem_params + mem_bufs

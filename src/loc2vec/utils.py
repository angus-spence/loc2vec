import tomllib
from dataclasses import dataclass

import torch, torchvision
import numpy as np

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

@dataclass
class Config:
    src: str = "./src/loc2vec/config.toml"

    def __post_init__(self):
        with open("./src/loc2vec/config.toml", "rb") as f:
            d = tomllib.load(f)
        self.epochs, self.lr, self.channels = d['training_parameters'].values()
        self.x_path, self.x_pos_path, self.x_neg_path = d['paths'].values()
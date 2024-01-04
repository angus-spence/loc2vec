from loc2vec.loc2vec_nn import Network, TripletLossFunction
from loc2vec.config import Params
from loc2vec.data_loader import Data_Loader

import random
from dataclasses import dataclass
from itertools import chain
from itertools import groupby
from typing import Any

import os
import torch
import torchvision as tv
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm

# THIS IMPLEMENTATION IS BAD -> WE DONT NEED TO COMPUTE ALL PAIR
# WISE DISTANCES BECAUSE THEY WONT ALL BE USED
# 
# WE SHOULD JUST IDENTIFY A 'HARD' TRIPLET N% OF THE TIME

@dataclass
class TripletMiner:
    """
    Mining of soft and hard triplets for re-training of model. Requires
    model weights to mine triplets.
    
    Parameters
    ----------
    model: torch.nn.Module
        triplet-loss network
    device: str
        device to use in embedding compute
    weights: str
        path to trained model to load weights from
    sh_ration: float
        ratio of soft / hard triplets to derive
    difficulty: float
        difficulty (0 - 1) of hard triplets
    """
    image_dir: str
    model: torch.nn.Module
    device: str    
    weights: str
    sh_ratio: float

    def __post_init__(self):
        self.model.to(device)
        self.model.load_state_dict(torch.load(self.weights, map_location=torch.device(self.device)))
        self.batch_size = self._optim_batch(self.model,
                                            (self.channels * self.dimension[0], self.dimension[1], self.dimension[2]),
                                            self.samples,
                                            128)

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.embeddings

    def _strp_ids(self, path: str, splitter: str, pos: int):
        """
        Extract list of file IDs from file name

        Parameters
        ----------
        path: str
            path of intput files
        splitter: str
            delineator in file name exposing id
        pos: int
            split index to use as id

        Returns
        -------
        ids: list
            list of identifiers
        """
        ids = []
        for root, dir, file in os.walk(path):
            if file and splitter: ids.append(str(file).strip(splitter)[pos])
            elif file: ids.append(file) 
        return ids

    def _check_channels(self) -> bool:
        """
        """

    def _get_paths(self): 
        """
        """
        try:
            return self.paths
        except:
            file_names = []
            paths = []
            for root, dirs, files in os.walk(self.image_dir):
                if files: file_names.append(files)
            for file_idx in tqdm(range(len(file_names[0])), desc=f'BUILDING PATHS'):
                paths.append([os.path.join(self.image_dir, os.listdir(self.image_dir)[i], file_names[i][file_idx]) for i in range(len(file_names))])
            self.paths = paths
        return self.paths

    @property
    def embeddings(self):
        return self.embs

    @embeddings.setter
    def embeddings(self):
        """
        """
        embs = []
        for channel in tqdm(self._get_paths(), desc=f'LOADING IMAGES TO TENRSOR'):
            for batch in (self.samples // self.batch_size) + 1:
                a, b = 0, self.batch_size
                tensor = torch.cat([tv.io.read_image(i)[:3,:,:].type(torch.float).to(self.device) for i in channel[a:b]])
                embs.append(self.model(tensor))
                a += self.batch_size
                b += self.batch_size
        self.embs = embs

    def _input_data_spec(self) -> int:
        """
        Returns
        -------
        no_channels: int
            number of input channels to the nn
        no_samples: int
            number of image samples
        """
        channels = []
        files = []
        for root, dirs, files in os.walk(self.image_dir):
            if dirs: channels.append(dirs)
            if files: files.append(len(files))
        self._ds = channels[0], files[0], tuple(tv.io.read_image(self._get_paths()[0][0])[:3,:,:].shape)

    @property
    def dimension(self):
        return self.dimension
    
    @dimension.setter
    def dimensions(self):
        return self._ds[2]

    @property
    def samples(self):
        return self.samples
    
    @samples.setter
    def samples(self):
        self.samples = self._ds[1] 

    @property
    def channels(self):
        return self.channels

    @channels.setter
    def channels(self):
        self.channels = self._ds()[0] 

    def _optim_batch(self, 
                     model: torch.nn.Module,
                     input_shape: tuple,
                     samples: int,
                     max_batch_size: int = None,
                     num_iterations: int = 6,
                     headroom_bias: int = None) -> int:
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
        self._force_cudnn_init()
        model.to(self.device)
        
        batch_size = 2
        while True:
            if max_batch_size is not None and batch_size >= max_batch_size:
                batch_size = max_batch_size
            if batch_size >= samples:
                batch_size = batch_size // 2
            try:
                for _ in tqdm(range(num_iterations), desc="EVALUATING OPTIMIUM BATCH SIZE"):
                    anchor_i = torch.rand(*(batch_size, *input_shape), device=self.device, dtype=torch.float)
                    outputs = model(anchor_i)
                    batch_size *= 2
            except RuntimeError as e:
                if str(e)[:18] == "CUDA out of memory": 
                    batch_size //= 2
                    break
                else:
                    print(e)
                    quit()

        del model
        torch.cuda.empty_cache()
        print(f'Optimum batch size: {batch_size}')
        return batch_size

    def _force_cudnn_init(self):
        s=32
        torch.nn.functional.conv2d(torch.zeros(s, s, s, s, device=self.device), torch.zeros(s, s, s, s, device=self.device))
        torch.cuda.empty_cache()        

    def _pairs_matrix(self):
        pairs = []
        for img in self._get_paths():
            pairs.append([])

    def _pairwise_distance_matrix(self):
        """
        """
        pwdm = []
        for img in self._evaluate_embeddings():
            pwdm.append(torch.stack(pwdm.append([F.pairwise_distance(img, i) for i in img])))
        self.pwdm = pwdm

    @staticmethod
    def var(sample) -> dict:
        """
        """
        vars = []
        for i in sample:
            var = (i - sum(sample) / len*sample) ** 2
            vars.append(var)
        return dict(zip((i for i in range(len(vars))), vars))

    @staticmethod
    def outliers_filter(sample: list, z) -> list:
        """
        """
        if len(sample) < 3:
            raise ValueError(f'Not enough data points')
        mean = sum(sample) / len(sample)
        std_dev = (sum((x - mean) ** 2 for x in sample) / len(sample)) ** 0.5
        vars = [sample.index(x) for x in sample if abs(x - mean) / std_dev < z]
        return vars

    def _hard(self, head):
        """
        """
        hard_triplet = []
        if not self.pwdm: self._pairwise_distance_matrix()
        for distances in self.pwdm:
            variances = self.var(distances)
            # NEED TO IMPLEMENT OUTLIER FILTER HERE
            sort = sorted(variances.items(), key=lambda item: item[1], reverse=True)
            top = [item[0] for item in sort[:head]]
            rndm = random.choice(top)
            hard_triplet.append(rndm)
        print(hard_triplet)
        return hard_triplet

    def _soft(self):
        pass

if __name__ == "__main__":
    if torch.cuda.is_available():
        device='cuda'
    else:
        device='cpu'
    miner = TripletMiner(image_dir=Params.X_PATH.value,
                         model=Network(in_channels=15), 
                         device=device, 
                         weights='src/loc2vec/loc2vec_model', 
                         sh_ratio=0.8)
    print(miner())
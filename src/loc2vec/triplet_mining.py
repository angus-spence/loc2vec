from loc2vec.loc2vec_nn import Network
from loc2vec.config import Params
from loc2vec.data_loader import SlimLoader
from loc2vec.optim import batch_optimiser, pca_dim_reduction
from loc2vec.loc2vec_run import evaluate_embeddings
import random
from dataclasses import dataclass
from itertools import chain
from itertools import groupby
from typing import Any

import os
import torch
import torchvision as tv
import torch.nn.functional as F

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
    embeddings: list = None
    batch_size: int = None
    dim_reduction: bool = True

    def __post_init__(self):
        self.channels, self.samples, self.dimension = self._input_data_spec()
        self.model = Network(in_channels=self.channels)
        if not self.batch_size: self.batch_size = batch_optimiser(
            model=self.model,
            device=self.device,
            input_shape=(*(self.channels * self.dimension[0]), self.dimension[1], self.dimension[2])
        )

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return 

    def _input_data_spec(self):
        """
        """
        no_channels = []
        no_files = []
        for root, dirs, files in os.walk(self.image_dir):
            if dirs: no_channels.append(len(dirs))
            if files: no_files.append(len(files))
            d_img = os.path.join(self.image_dir, dirs[0], files[0][0])
        return no_channels[0], no_files[0], tv.io.read_image(d_img)[:3,:,:].shape 

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

    def _get_embeddings(self):
        return evaluate_embeddings(img_dir=self.image_dir,
                                   batch_size=self.batch_size,
                                   device=self.device,
                                   to_csv=False)
         
    def _pairwise_distance_matrix(self) -> None:
        """
        """
        embs = self._get_embeddings()
        if self.dim_reduction:
            embs = pca_dim_reduction(embs,
                                     dims=120,
                                     device=self.device)
        pwdm = []
        for img in embs:
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
        if not self.pwdm: self._pairwise_distance_matrix()
        hard_triplet = []
        for distances in self.pwdm:
            variances = self.var(distances)
            # NEED TO IMPLEMENT OUTLIER FILTER HERE
            sort = sorted(variances.items(), key=lambda item: item[1], reverse=True)
            top = [item[0] for item in sort[:head]]
            rndm = random.choice(top)
            hard_triplet.append(rndm)
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
from loc2vec.loc2vec_nn import Network
from loc2vec.config import Params

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
        self.model = self.model.load_state_dict(torch.load(self.weights, map_location=torch.device(self.device))).to(self.device)

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self._evaluate_embeddings()

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

    def _evaluate_embeddings(self):
        """
        """
        embs = []
        for channel in tqdm(self._get_paths(), desc=f'EVALUATING EMBEDDINGS'):
            tensor = torch.cat([tv.io.read_image(i)[:3,:,:].type(torch.float).to(self.device) for i in channel])
            embs.append(self.model(tensor))
        return embs

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
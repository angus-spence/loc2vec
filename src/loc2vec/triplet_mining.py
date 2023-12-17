from typing import Any
from loc2vec.loc2vec_nn import Network
from loc2vec.data_loader import Data_Loader
from loc2vec.config import Params

from dataclasses import dataclass
from itertools import chain

import os
import torch
import torch.nn.functional as F
import torchvision as tv
from tqdm import tqdm

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
        difficulty of hard triplets
    """
    image_dir: str
    model: torch.nn.Module
    device: str    
    weights: str
    sh_ratio: float

    def __post_init__(self):
        self.model.load_state_dict(torch.load('src/loc2vec/loc2vec_model', map_location=torch.device(self.device)))


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

    def _path_stack(self):
        paths, stacker, channels = [], [], []
        for root, dir, file in os.walk(self.image_dir):
            if file: paths.append(file)
            if dir: channels.append(dir)
        channels = list(chain.from_iterable(channels))
        print(len(paths))
        for i in tqdm(range(len(paths))):
            stacker.append([os.path.join(self.image_dir, channels[j], paths[j][i]) for j in range(len(channels))])
        print(stacker)
        return stacker

    def _evaluate_embeddings(self):
        embs = []
        for channel in self._path_stack():
            img_tensor = []
            img_tensor.append([tv.io.read_image(img)[:3,:,:].type(torch.float).to(self.device) for img in channel])
            img_tensor = list(chain.from_iterable(img_tensor))
            tensor_out = torch.cat(img_tensor)
            print(tensor_out.shape)
            embs.append(self.model(tensor_out))
        return embs
    
    def _pairwise_distance_matrix(self):
        pwdm = []
        print(self.embs)
        for img_x in self.embs:
            pwdm = torch.stack(pwdm.append([F.pairwise_distance(img_x, i) for i in img_x]))
        return pwdm

if __name__ == "__main__":
    miner = TripletMiner(image_dir=r'C:\Users\aspence1\Documents\loc2vec_data\x_pos',
                         model=Network(in_channels=15), 
                         device='cpu', 
                         weights='src/loc2vec/loc2vec_model', 
                         sh_ratio=0.8)
    print(len(miner._evaluate_embeddings()))
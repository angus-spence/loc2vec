from loc2vec.loc2vec_nn import Network
from loc2vec.data_loader import Data_Loader
from loc2vec.config import Params

from dataclasses import dataclass

import torch

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
    model: torch.nn.Module
    device: str    
    weights: str
    sh_ratio: float

    def __post_init__(self):
        self.loader = Data_Loader(x_path=Params.X_PATH.value, x_pos_path=Params.X_POS_PATH.value, batch_size=2)
        self.model.load_state_dict(torch.load('src/loc2vec/loc2vec_model', map_location=torch.device(self.device)))

    def _evaluate_embeddings(self):
        
        for img in range(len(self.loader)):
            return

if __name__ == "__main__":

    miner = TripletMiner(model=Network(in_channels=15), 
                         device='cpu', 
                         weights='src/loc2vec/loc2vec_model', 
                         sh_ratio=0.8)
    miner._evaluate_embeddings()
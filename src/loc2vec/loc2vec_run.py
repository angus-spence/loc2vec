from loc2vec.loc2vec_nn import Network
from loc2vec.config import Params
from loc2vec.data_loader_slim import SlimLoader

import csv
import os

from tqdm import tqdm
import numpy as np
import torch

# MAKE A SLIM DATA LOADER!!

def evaluate_embeddings(img_dir: str, 
                        batch_size: int, 
                        device: str, 
                        to_csv: bool = True) -> list:
    """
    """
    loader = SlimLoader(
        img_dir=img_dir,
        shuffle=False,
        batch_size=8,
        device=device
    )
    model = Network(in_channels=15)
    model.load_state_dict(torch.load('src/loc2vec/loc2vec_model', 
                                     map_location=torch.device(device)))

    embs = []
    for batch in tqdm(range(len(loader)//batch_size)):
        x = next(loader)
        embs.append(model(x))

    if to_csv:
        with open('embs', 'w') as f:
            write = csv.writer(f)
            write.writerows(embs)

    return embs

if __name__ == "__main__":
    if torch.cuda.is_available():
        device = "cuda"
    else: 
        device = 'cpu'
    embs = evaluate_embeddings(Params.X_PATH.value,
                               batch_size=4,
                               device=device)
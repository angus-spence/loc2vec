from loc2vec.loc2vec_nn import Network, TripletLossFunction
from loc2vec.config import Params

import torch

model = Network(in_channels=12)
model.load_state_dict(torch.load('src/loc2vec/loc2vec_model', map_location=torch.device("cpu")))
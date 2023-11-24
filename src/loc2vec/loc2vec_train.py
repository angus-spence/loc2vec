from loc2vec.loc2vec_nn import Network, TripletLossFunction
from loc2vec.config import Params
from loc2vec.data_loader import Data_Loader

import torch
import numpy as np
from tqdm import tqdm

def train():
    data = loader.load_from_dirs()
    optimiser = torch.optim.Adam(model.parameters(), lr=Params.LEARNING_RATE.value)
    criterion = TripletLossFunction()

    for epoch in tqdm(range(Params.EPOCHS.value), desc='Epochs'):
        running_loss = []
        for step, (anchor, anchor_pos, anchor_neg) in enumerate(tqdm(data, desc="Training", leave=False)):
            x = anchor
            x_pos = anchor_pos
            x_neg = anchor_neg

            x_out = model(x)
            x_pos_out = model(x_pos)
            x_neg_out = model(x_neg)

            loss = criterion(x_out, x_pos_out, x_neg_out)
            loss.backward()
            optimiser.step()

            running_loss.append(loss.cpu().detach().numpy())
            print(f'Epoch: {epoch+1}/{Params.EPOCHS.value} - Loss: {round(np.mean(running_loss), 5)}')

if __name__ == "__main__":
    loader = Data_Loader(Params.X_PATH.value, x_pos_path=Params.X_POS_PATH.value)
    model = Network()
    batch_size = loader._optim_batch(model, (3*12, 500, 500), (128), 100000, headroom_bias=250000000)
    print(batch_size)
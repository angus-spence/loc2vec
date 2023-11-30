from loc2vec.loc2vec_nn import Network, Loc2vec, TripletLossFunction
from loc2vec.config import Params
from loc2vec.data_loader import Data_Loader

import torch
import numpy as np
from tqdm import tqdm

def train():
    loader = Data_Loader(Params.X_PATH.value, x_pos_path=Params.X_POS_PATH.value)
    device = loader.device
    #model = Network(in_channels=loader.in_channels).to(device)
    model = Loc2vec()
    optimiser = torch.optim.Adam(model.parameters(), lr=Params.LEARNING_RATE.value)
    criterion = TripletLossFunction().to(device)

    for epoch in tqdm(range(Params.EPOCHS.value), desc='Epochs'):
        running_loss = []
        
        for batch in range(loader.batches):
            o, plus, neg = next(loader)
            o, plus, neg = (model(o), model(plus), model(neg))

            loss = criterion(o, plus, neg)
            loss.backward()
            optimiser.step()

            del o, plus, neg

            running_loss.append(loss.cpu().detach().numpy())
            print(f'Epoch: {epoch+1}/{Params.EPOCHS.value} - Loss: {round(np.mean(running_loss), 3)}', end='\r')
    torch.save(model, "loc2vec_model")

if __name__ == "__main__":
    train()
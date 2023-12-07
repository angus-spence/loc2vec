from loc2vec.loc2vec_nn import Network, Loc2vec, TripletLossFunction
from loc2vec.config import Params
from loc2vec.data_loader import Data_Loader

import torch
import numpy as np
from tqdm import tqdm

def train(plot: bool = False):
    loader = Data_Loader(Params.X_PATH.value, x_pos_path=Params.X_POS_PATH.value)
    device = loader.device
    model = Network(in_channels=loader.in_channels)
    model.to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=Params.LEARNING_RATE.value)
    criterion = TripletLossFunction(margin=0.5).to(device)

    for epoch in tqdm(range(Params.EPOCHS.value), desc='Epochs'):
        running_loss = []
        running_points = []
        
        for batch in range(loader.batches):
            o, plus, neg = next(loader)
            #if Loc2vec: 
            #    o = o.view(-1, *loader._image_shape())
            #    plus = plus.view(-1, *loader._image_shape())
            #    neg = neg.view(-1, *loader._image_shape())
            
            o, plus, neg = (model(o), model(plus), model(neg))
            running_points.append(o, plus, neg)

            loss, loss_summary = criterion(o, plus, neg)
            loss.backward()
            optimiser.step()

            del o, plus, neg

            running_loss.append(loss.cpu().detach().numpy())
            print(f'Batch: {batch+1}/{loader.batches} - Running Loss: {round(float(np.mean(running_loss)), 3)}')
            print(loss_summary)
    
    torch.save(model, "loc2vec_model")

    if plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()

if __name__ == "__main__":
    train()
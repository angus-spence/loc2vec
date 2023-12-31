from loc2vec.loc2vec_nn import Network, Loc2vec, TripletLossFunction
from loc2vec.config import Params
from loc2vec.data_loader import Data_Loader

import torch
import numpy as np
from tqdm import tqdm

def train(logging: bool = True, plot: bool = False, reinforce: bool = True) -> None:
    """
    Training function for loc2vec Model which is saved after upon completion.

    Parameters
    ----------
    train_limit: int
        Maximum number of steps to train for
    
    logging: bool
        Bool for saving model analytics to a csv

    plot: bool
        Bool for plotting model analytics
    """
    loader = Data_Loader(x_path=Params.X_PATH.value, x_pos_path=Params.X_POS_PATH.value)
    device = loader.device
    model = Network(in_channels=loader.in_channels)
    model.to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-6)
    criterion = TripletLossFunction(margin=1).to(device)

    for epoch in range(Params.EPOCHS.value):
        running_loss = []
        running_points = []
        ap_log, an_log, mn_log = [], [], []
        
        batch_id = 0
        for batch in range(loader.batches):
            o, plus, neg = next(loader)
            #if Loc2vec: 
            #    o = o.view(-1, *loader._image_shape())
            #    plus = plus.view(-1, *loader._image_shape())
            #    neg = neg.view(-1, *loader._image_shape())
            o, plus, neg = (model(o), model(plus), model(neg))
            loss, loss_summary, ap, an, mn = criterion(o, plus, neg)
            loss.backward()
            optimiser.step()
            running_loss.append(loss.cpu().detach().numpy())
            ap_log.append(ap)
            an_log.append(an)
            mn_log.append(mn)

            # TODO: RUNNING THE SAME PNG THROUGH THE MODEL AND GETTING THE EMBEDDINGS, DEMONSTRATING
            #       HOW THE EMBEDDINGS CHANGES THROUGH THE EPOCHS -> WE HAVE RAM AVAILABLE FOR THIS
            
            print(f'Epoch: {epoch} Sample Set: {batch_id+1}/{loader.batches} - Running Loss: {round(float(np.mean(running_loss)), 3)}')
            print(loss_summary)
            batch_id += 1

    torch.save(model.state_dict(), "loc2vec_model")

    if logging:
        import pandas as pd

        df_loss = pd.DataFrame({'running_loss': running_loss,
                           'distance_ap': ap_log,
                           'distance_neg': an_log,
                           'distance_mn': mn_log}).to_csv('loc2vec_log.csv')
        df_emb = pd.DataFrame({
                           'point_o': [running_points[i][0] for i in range(len(running_points))],
                           'point_+': [running_points[i][1] for i in range(len(running_points))],
                           'point_-': [running_points[i][2] for i in range(len(running_points))]}).to_csv('embs.csv')
    
    if plot:
        import matplotlib.pyplot as plt

        iters = np.arange(0, len(running_loss), 1)
        
        fig, ax = plt.subplots(1, 2)
        ax[0].plot(iters, running_loss)
        ax[0].set_title("LOSS FUNCTION")
        #ax[0, 1].plot(iters, [running_points[i][0] for i in range(len(iters))])
        #ax[0, 1].plot(iters, [running_points[i][1] for i in range(len(iters))])
        #ax[0, 1].plot(iters, [running_points[i][2] for i in range(len(iters))])
        #ax[0, 1].set_title("")
        ax[1].plot(iters, ap_log)
        ax[1].plot(iters, an_log)
        ax[1].plot(iters, mn_log)
        ax[1].set_title("EUCLIDEAN DISTANCE BETWEEN TRIPLETS")

        plt.savefig("Loc2vec_plot", dpi=150)

    if reinforce:
        pass

if __name__ == "__main__":
    train(plot=True, logging=True)
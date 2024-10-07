from loc2vec.loc2vec_nn import Network, TripletLossFunction
from loc2vec.utils import Config
from loc2vec.data_loading import Tensor_Loader

import time
import logging
import uuid

import torch
import numpy as np

def train(batch_size = 16,
          margin: int = 1,
          logging: bool = True, 
          resnet: bool = False, 
          debug: bool = True, 
          plot: bool = False, 
          reinforce: bool = True,
          l_limit: float = 0.085,
          n = 25
          ) -> None:
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
    cfg = Config()
    loader = Tensor_Loader(anchor_i_path=cfg.anchor_i_path,
                           anchor_p_path=cfg.anchor_pos_path,
                           anchor_n_path=None,
                           batch_size=batch_size,
                           shuffle=False

    )
    device = loader.device
    model = Network(in_channels=loader.anchors[0].no_channels * 3, debug=debug, resnet=resnet)
    model.to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    criterion = TripletLossFunction(margin=margin).to(device)

    print("STARTING MODEL TRAINING")
    print(f"DEVICE: {str(device).upper()}")
    running_loss = []
    ap_log, an_log, mn_log = [], [], []
    batch_times = []
    btc = cfg.epochs * loader.batches
    for epoch in range(cfg.epochs):
        batch_id = 0
        for batch in range(loader.batches):
            ts = time.time()
            
            try: o, plus, neg, valid = next(loader)
            except RuntimeError as e:
                print(f"FILE READ ERROR AT BATCH: {batch_id}")
                continue
            
            optimiser.zero_grad()
            o, plus, neg = (model(o), model(plus), model(neg))
            loss, loss_summary, ap, an, mn = criterion(o, plus, neg)
            loss.backward()
            optimiser.step()
            running_loss.append(loss.cpu().detach().numpy())
            ap_log.append(ap)
            an_log.append(an)
            mn_log.append(mn)

            if len(running_loss) > n and sum(running_loss[-n:])/n < l_limit: break

            te = time.time()
            batch_times.append(te - ts)
            avg_batch_time = sum(batch_times)/len(batch_times)
            batch_id += 1
            pcom = round(((epoch*loader.batches + batch_id)/btc)*100, 2)
            eta = round(((btc - (epoch*loader.batches + batch_id)) * avg_batch_time)/60, 2)

            print(f'EPOCH: {epoch} BATCH: {batch_id+1}/{loader.batches} - RUNNING_LOSS: {round(float(np.mean(running_loss)), 3)}')
            print(loss_summary)
            print(f'{pcom}% COMPLETE | <{eta} MINUTES TO COMPLETE')

    print("SAVING MODEL")
    torch.save(model.state_dict(), "loc2vec_model_3_channel_140324")

    if logging:
        import pandas as pd

        df_loss = pd.DataFrame({'running_loss': running_loss,
                           'distance_ap': ap_log,
                           'distance_neg': an_log,
                           'distance_mn': mn_log}).to_csv('loc2vec_log.csv')
    
    if plot:
        print("PLOTTING")
        import matplotlib.pyplot as plt

        iters = np.arange(0, len(running_loss), 1)
        
        fig, ax = plt.subplots(1, 2, figsize=(16,9))
        ax[0].plot(iters, running_loss, linewidth=0.3, alpha=0.6)
        ax[0].set_title("LOSS FUNCTION")
        ax[1].plot(iters, ap_log, linewidth=0.3, alpha=0.6)
        ax[1].plot(iters, an_log, linewidth=0.3, alpha=0.6)
        ax[1].plot(iters, mn_log, linewidth=0.3, alpha=0.6)
        ax[1].set_title("EUCLIDEAN DISTANCE BETWEEN TRIPLETS")

        fig.set_dpi(150)
        plt.savefig(f"Loc2vec_3_channel_loss_{uuid.uuid4().hex}", dpi=150)
        
    if reinforce:
        pass

if __name__ == "__main__":
    logging.basicConfig(filename='tv.io.read_image.log', level=logging.ERROR)
    train(resnet=True, debug=False, batch_size=16)
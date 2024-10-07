from loc2vec.loc2vec_nn import Network
from loc2vec.utils import Config
from loc2vec.data_loading import Tensor_Loader

import csv
import os
import datetime
from itertools import chain

from tqdm import tqdm
import torch

def evaluate_embeddings(img_dir: str, 
                        batch_size: int, 
                        device: str, 
                        to_csv: bool = True) -> list:
    """
    """

    loader = Tensor_Loader(
        anchor_i_path=cfg.anchor_i_path,
        anchor_p_path=cfg.anchor_pos_path,
        anchor_n_path=None,
        batch_size=batch_size
    )
    ids = list(chain.from_iterable([f for (r, d, f) in os.walk(loader.anchors[0].channels[0].path) if f]))
    ids = [i.removesuffix(".png").removeprefix("output_") for i in ids]
    model = Network(in_channels=3, debug=False, resnet=True)
    model.to(device)
    model.load_state_dict(torch.load('loc2vec_model_3_channel_120324', 
                                     map_location=torch.device(device)))

    itr = 0
    for batch in tqdm(range(len(loader)//batch_size)):
        accident_ids = ids[itr:itr+batch_size]
        try: i, p, n, valid = next(loader)
        except: continue
        embs = model(i)
        embs = torch.flatten(embs, start_dim=1)
        embs = embs.cpu().detach().numpy().tolist()

        if to_csv:
            with open(os.path.join(cfg.output_path, f'embs_{str(datetime.date.today())}.csv'), 'a', newline='') as f:
                write = csv.writer(f)
                for row in embs:   
                    row.insert(0, accident_ids[embs.index(row)])
                    write.writerow(row)
        itr += batch_size

if __name__ == "__main__":
    cfg = Config()
    if torch.cuda.is_available(): device = "cuda"
    else: device = "cpu"
    embs = evaluate_embeddings(cfg.anchor_i_path,
                               batch_size=64,
                               device=device)
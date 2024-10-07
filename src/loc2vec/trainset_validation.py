from loc2vec.utils import Config, visualise_tensor
from loc2vec.data_loading import Tensor_Loader

import os
import numpy as np
from itertools import chain

import torch
from torch.nn import functional as F
import matplotlib.pyplot as plt

def list_files(dir):
    fn = []
    for r, d, f in os.walk(dir):
        if f: fn.append(f)
    return list(chain.from_iterable(fn))

def remove_uncommon(dir1, dir2):
    fn1 = set(list_files(dir1))
    fn2 = set(list_files(dir2))
    ftd = list(fn1.symmetric_difference(fn2))
    flen1, flen2 = (len(fn1), len(fn2))
    print(flen1, flen2)
    diff = max([flen1, flen2]) - min([flen1, flen2])
    print(f"DELETING {len(ftd)} FILES WHERE DIFFERENCE IS {diff}. DO YOU WANT TO CONTINNUE:")
    c = str(input("Y = YES / ELSE = NO: ")).lower()
    if c == "y":
        delA = [os.path.join(dir1, "all_layers", i) for i in ftd]
        delB = [os.path.join(dir2, "all_layers", i) for i in ftd]
        dels = 0
        for path in delA:
            try:
                os.remove(path)
                dels += 1
                print(f"DELETED {dels}/{len(ftd)} FILES")
            except:
                continue
        for path in delB:
            try:
                os.remove(path)
                dels += 1
                print(f"DELETED {dels}/{len(ftd)} FILES")
            except:
                continue
    else:
        print("DID NOTHING")
        return

def check_bias(dir):
    paths = []
    for r, d, f in os.walk(dir):
        for file in f:
            paths.append(os.path.join(r, file))
    fsize = [os.path.getsize(f) for f in paths]
    plt.hist(fsize, bins=500, color=(210/255, 35/255, 40/255, 1))
    plt.title("DISTRIBUTION OF URBAN DENSITY IN IMAGE SAMPLES")
    plt.xlabel("File Size Inferring Urban Density")
    plt.ylabel("Count")
    plt.show()

def test():
    loader = Tensor_Loader(cfg.anchor_i_path,
                           cfg.anchor_pos_path,
                           batch_size=16)
    a, ap, an, valid = next(loader)
    apnvis = (visualise_tensor(a), visualise_tensor(ap), visualise_tensor(an))
    fig, axs = plt.subplots(1, 4)
    axs[0].matshow(apnvis[0])
    axs[1].matshow(apnvis[1])
    axs[2].matshow(apnvis[2])
    axs[3].matshow(torch.flatten(F.cosine_similarity(a, ap, dim=3), start_dim=1).cpu().numpy())
    plt.show()
    print(a.shape)
    print(F.cosine_similarity(a, ap, dim=2).mean())
    print(F.cosine_similarity(a, an, dim=2).mean())
    print(F.cosine_similarity(a, ap, dim=3).mean())
    print(F.cosine_similarity(a, an, dim=3).mean())

if __name__ == "__main__":
    cfg = Config()
    test()
    quit()
    dir1 = cfg.anchor_i_path
    dir2 = cfg.anchor_pos_path
    remove_uncommon(dir1, dir2)
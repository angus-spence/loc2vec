from loc2vec.utils import Config
from loc2vec.optim import batch_optimiser

import os
import shutil
from itertools import chain
from dataclasses import dataclass

import torch
import torchvision as tv
from tqdm import tqdm

@dataclass
class Tensor_Loader:
    anchor_i_path: str
    anchor_p_path: str
    anchor_n_path: str = None
    batch_size: int = None
    shuffle: bool = False

    def __post_init__(self) -> None:
        """
        """
        if torch.cuda.is_available(): self.device = torch.device('cuda')
        else: self.device = torch.device('cpu')

        self.paths = (self.anchor_i_path, self.anchor_p_path, self.anchor_n_path)
        self.achors = [Anchor(i) for i in self.paths if i is not None]

        if not self.batch_size:
            

@dataclass
class Anchor:
    path: str

    def __post_init__(self):
        self._check_path()
        self.no_channels = []
        self.no_samples = []
        for path, channels, samples in os.walk(self.path):
            if channels: self.no_channels = len(channels)
            if samples: self.no_samples.append(len(samples))

        self.channels = [Channel(os.path.join(self.path, i)) for i in os.listdir(self.path)]

    def __len__(self) -> int:
        return self.channels

    def _get_paths(self) -> list:
        try: return self.fns
        except:
            paths = []
            fns = []
            for root, dirs, files in os.walk(self.path):
                if files: fns.append(files)
            for j in tqdm(range(len(fns[0])), desc=f'BUILDINGS PATHS FOR {str(self.path).upper()}'):
                paths.append([os.path.join(self.path, os.listdir(self.path)[i], fns[i][j]) for i in range(len(fns))])
            self.fns = list(chain.from_iterable(paths))
        return self.fns

    def _get_common_ids(self) -> list:
        fns = [s for (p, c, s) in os.walk(self.path) if s]
        if self.no_channels == 1: return fns
        else: return list(chain.from_iterable(fns))

    def _check_samples(self) -> bool:
        return all(x == self.no_samples[0] for x in self.no_samples)
    
    def _check_path(self) -> None:
        if os.path.exists(self.path) == False:
            raise ValueError(f'Path {self.path} does not exist')

@dataclass
class Channel(list):
    path: str

    def __post_init__(self):
        self._check_path()
        self.samples = [i for i in os.listdir(self.path)]

    def squeeze(self, common: list, destructive: bool) -> None:
        rm = [i for i in self.samples if i not in common]
        kp = [i for i in self.samples if i not in rm]
        if destructive:
            (os.remove(os.path.join(self.path, i)) for i in rm)
        else:
            p = os.path.join(self.path, f'clean')
            os.mkdir(p)
            for file in tqdm(kp, desc="   -> MOVING FILES TO NEW DIR"):
                shutil.copy(os.path.join(self.path, file), os.path.join(p, file))
            self.path = p

    def _check_path(self) -> None:
        if os.path.exists(self.path) == False:
            raise ValueError(f'Path "{self.path}" does not exist')

if __name__ == "__main__":
    cfg = Config()
    l = Tensor_Loader(
        anchor_i_path=cfg.anchor_i_path,
        anchor_p_path=cfg.anchor_pos_path,
        anchor_n_path=None
    )
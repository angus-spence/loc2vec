from loc2vec.loc2vec_nn import Network
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
    _itridx: int = 0
    _s: int = 0
    _e: int = 0

    def __post_init__(self) -> None:
        """
        """
        if torch.cuda.is_available(): self.device = torch.device('cuda')
        else: self.device = torch.device('cpu')

        self.paths = (self.anchor_i_path, self.anchor_p_path, self.anchor_n_path)
        self.anchors = [Anchor(i) for i in self.paths if i is not None]
        self.dim = self.anchors[0].channels[0]._get_dimension()

        if not self.batch_size:
            model = Network(in_channels=self.anchors[0].no_channels * 3)
            self.batch_size = batch_optimiser(
                model=model,
                device=self.device,
                input_shape=(self.dim[0] * self.anchors[0].no_channels, *self.dim[1:]),
                no_samples=self.anchors[0].channels[0].no_samples,
                no_iterations=10,
                max_batch_size=128
            )
            del model
        self._e = self.batch_size

    def __len__(self):
        return self.anchors[0].channels[0].no_samples
    
    def __call__(self, index):
        return tv.io.read_image(self.anchors[0].channels[0].paths[index])
    
    def __inter__(self):
        return self
    
    def __next__(self):
        if self._itridx < len(self) // self.batch_size:
            self._itridx += self.batch_size
            i, p, n = self._channel_call(self._s, self._e)
            self._s += self.batch_size
            self._e += self.batch_size
            return i, p, n
        else:
            self._itridx, self._s, self._e = 0, 0, self.batch_size
            i, p, n = self._channel_call(self._s, self._e)
            self._s += self.batch_size
            self._e += self.batch_size
            return i, p, n

    def _channel_call(self, start: int, end: int):
        batch = []
        for a in self.anchors:
            for c in a.channels:
                batch.append(c._get_paths()[start:end])
        print(batch)
        
        for anchor in batch:
            for channel in batch:
                channel_tensor = []
                for img in channel:
                    channel_tensor.append(tv.io.read_image(img)[:3,:,:].type(torch.float).to(self.device))
                batch.append(torch.cat(channel_tensor))
        return batch
    
    def __reverse__(self):
        self._itridx -= self.batch_size
        return self

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
        self.no_samples = len(self.samples)
        self.paths = self._get_paths()

    def squeeze(self, common: list, destructive: bool) -> None:
        print("   -> IDENTIFYING COMMON IDS")
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

    def _get_paths(self) -> list:
        return [os.path.join(self.path, os.listdir(self.path)[i]) for i in range(len(os.listdir(self.path)))]

    def _check_path(self) -> None:
        if os.path.exists(self.path) == False:
            raise ValueError(f'Path "{self.path}" does not exist')
        
    def _get_dimension(self):
        return tuple(tv.io.read_image(os.path.join(self.path, os.listdir(self.path)[0]))[:3,:,:].shape)

if __name__ == "__main__":
    cfg = Config()
    l = Tensor_Loader(
        anchor_i_path=cfg.anchor_i_path,
        anchor_p_path=cfg.anchor_pos_path,
        anchor_n_path=None,
        batch_size=128
    )
    print(l.__next__())
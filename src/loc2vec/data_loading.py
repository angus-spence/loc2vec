from loc2vec.loc2vec_nn import Network
from loc2vec.utils import Config, visualise_tensor
from loc2vec.optim import batch_optimiser

import random
import os
import shutil
from itertools import chain
from dataclasses import dataclass

import torch
import torchvision as tv
import matplotlib.pyplot as plt
from tqdm import tqdm

@dataclass
class Tensor_Loader:
    anchor_i_path: str
    anchor_p_path: str
    anchor_n_path: str = None
    batch_size: int = None
    shuffle: bool = False
    plot: bool = False
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
        self.batches = (len(self) - self._batch_dropout()) // self.batch_size

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
            call = self._channel_call(self._s, self._e)
            i, p, n, valid = (*call[0], call[1]) 
            self._s += self.batch_size
            self._e += self.batch_size
            if self.plot: self.visualise_anchors(i, p, n)
            return i, p, n, valid
        else:
            self._itridx, self._s, self._e = 0, 0, self.batch_size
            call = self._channel_call(self._s, self._e)
            i, p, n, valid = (*call[0], call[1]) 
            self._s += self.batch_size
            self._e += self.batch_size
            if self.plot: self.visualise_anchors(i, p, n)
            return i, p, n, valid

    def __reverse__(self):
        self._itridx -= self.batch_size
        return self

    def visualise_anchors(self,
             anchor_i: torch.Tensor, 
             anchor_p: torch.Tensor, 
             anchor_n: torch.Tensor) -> None:
        enu = [anchor_i, anchor_p, anchor_n]
        fig, ax = plt.subplots(1, 3)
        ax[0].matshow(visualise_tensor(anchor_i))
        ax[0].set_title("ANCHOR", fontsize=10)
        ax[0].axis('off')
        ax[1].matshow(visualise_tensor(anchor_p))
        ax[1].set_title("ANCHOR | POSITIVE", fontsize=10)
        ax[1].axis('off')
        ax[2].matshow(visualise_tensor(anchor_n))
        ax[2].set_title("ANCHOR | NEGATIVE", fontsize=10)
        ax[2].axis('off')
        fig.set_dpi(200)
        plt.show()

    def _batch_dropout(self):
        return len(self) // self.batch_size

    def _channel_call(self, start: int, end: int) -> torch.Tensor:
        batch = []
        for a in self.anchors:
            for c in a.channels:
                batch.append(c.paths[start:end])
        if self.anchor_n_path == None: 
            for c in self.anchors[0].channels:
                r = random.randint(0, self.anchors[0].channels[0].no_samples - self.batch_size)
                batch.append(c.paths[r:r+self.batch_size])
        return self._stack_to_tensor(batch)

    def _stack_to_tensor(self, stack) -> torch.Tensor:
        valid = self._batch_validation(stack)
        if not valid:
            print("INVALID FILE TYPE -> MOVING TO NEXT BATCH")
            return [[], [], []], valid 
        out = []
        tstack = [] 
        for anchor in stack:
            if self.anchors[0].no_channels == 1:
                tstack = torch.stack([tv.io.read_image(anchor[i])[:3,:,:].type(torch.float).to(self.device) for i in range(self.batch_size)])
                out.append(tstack)
            else:
                for channel in anchor:
                    print(channel)
                    tstack.append(torch.cat([tv.io.read_image(channel[i])[:3,:,:].type(torch.float).to(self.device) for i in range(self.batch_size)]))
                t = torch.stack(tstack)
                out.append(t)
        return out, valid
    
    def _batch_validation(self, stack) -> bool:
        for anchor in stack:
            return all([os.path.splitext(i)[-1] == '.png' for i in anchor])

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

    def _common_inputs(self) -> list:
        return

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
        self = self._get_paths()

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
        f = os.listdir(self.path)
        self.paths = [os.path.join(self.path, i) for i in tqdm(f)]
        return self.paths

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
        batch_size=8,
        plot=True 
    )
    l.__next__()
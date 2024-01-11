from loc2vec.config import Params

from typing import Union
from dataclasses import dataclass
import os

from tqdm import tqdm
import torch
import torchvision as tv

@dataclass
class SlimLoader:
    img_dir: str
    shuffle: bool
    batch_size: int
    device: str

    def __post_init__(self):
        self.channels, self.images, self.dimensions = self._input_spec()
        self.idx, self.s, self.e = (0, 0, self.batch_size)

    def __len__(self):
        return self.images
    
    def __iter__(self):
        return self

    def __next__(self) -> torch.Tensor:
        """
        """
        if self.idx < len(self) // self.batch_size:
            self.idx += self.batch_size
            path = self._get_paths()
            x = path[:len(self)]
            x_out = x[self.s, self.e]
            self.s += self.batch_size 
            self.e += self.batch_size
        return self._batch_to_tensor(x_out)

    def _input_spec(self) -> Union[int, int, tuple]:
        """
        """
        channels = []
        no_files = []
        for root, dirs, files, in os.walk(self.img_dir):
            if dirs: channels.append(len(dirs))
            if files: no_files.append(len(files))
        return channels[0], no_files[0], tuple(tv.io.read_image(self._get_paths()[0][0])[:3,:,:].shape)
    
    def _get_paths(self):
        """
        """
        try:
            return self.paths
        except:
            file_names = []
            paths = []
            for root, dirs, files in os.walk(self.img_dir):
                if files: file_names.append(files)
            for file_idx in tqdm(range(len(file_names[0])), desc=f'BUILDING PATHS'):
                paths.append([os.path.join(self.img_dir, os.listdir(self.img_dir)[i], file_names[i][file_idx]) for i in range(len(file_names))])
            self.paths = paths
        return self.paths
    
    def _batch_to_tensor(self, batch: list) -> torch.Tensor:
        """
        """
        batches = []
        for channel in batch:
            channels = []
            for img in channel:
                channels.append(tv.io.read_image(img)[:3,:,:].type(torch.float).to(self.device))
            batch_tensor = torch.cat(channels)
            batches.append(batch_tensor)
        return torch.stack(batches)
    
if __name__ == "__main__":
    loader = SlimLoader(
        img_dir=Params.X_PATH.value,
        shuffle=False,
        batch_size=4,
        device='cpu'
    )
    print(loader._input_spec())
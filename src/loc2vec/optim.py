from loc2vec.loc2vec_nn import TripletLossFunction

import numpy as np
import torch
from tqdm import tqdm

def batch_optimiser(self,
                    model: torch.nn.Module,
                    device: str,
                    input_shape: tuple,
                    no_samples: int,
                    no_iterations: int,
                    max_batch_size: int,
                    ) -> int:
    """
    """
    model.to(device)
    model.train(True)
    lf = TripletLossFunction
    optimiser = torch.optim.Adam(model.parameters())
    batch_size = 2
    while True:
        if max_batch_size is not None and batch_size >= max_batch_size:
            batch_size = max_batch_size
            break
        if batch_size >= no_samples:
            batch_size = batch_size // 2
            break
        try:
            for _ in tqdm(range(no_iterations), desc="Evaluating optimum batch size"):
                anchor_i = torch.rand(*(batch_size, *input_shape), device=self.device, dtype=torch.float)
                anchor_pos = torch.rand(*(batch_size, *input_shape), device=self.device, dtype=torch.float)
                anchor_neg = torch.rand(*(batch_size, *input_shape), device=self.device, dtype=torch.float)
                outputs = model(anchor_i)
                loss, loss_summary, ap, an, mn = lf(outputs, model(anchor_pos), model(anchor_neg))
                del loss_summary, ap, an, mn
                loss.backward()
                optimiser.step()
                optimiser.zero_grad()
                batch_size *= 2
        except RuntimeError as e:
            if str(e)[:18] == "CUDA out of memory": 
                batch_size //= 2
                break
            else:
                print(e)
                quit()

    del model, optimiser
    torch.cuda.empty_cache()
    print(f'Optimum batch size: {batch_size}')
    return batch_size

def pca_dim_reduction(x: torch.Tensor, dims: int, device: str) -> torch.Tensor:
    """
    """
    if device == "cuda":
        x = x.cpu().detach().numpy()
    else:
        x = x.detach().numpy()
    cov_matrix = np.cov(x)
    values, vectors = np.linalg.eig(cov_matrix)
    return torch.Tensor([x.dot(vectors.T[i]) for i in range(dims)]).to(device)
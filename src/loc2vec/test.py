

import numpy as np
import torch
import matplotlib.pyplot as plt

m = torch.nn.Conv3d(12, 32, 4, stride=2, padding=(1, 1, 1), bias=False)

x = torch.randn(12, 3, 128, 128)
y = m(x)

print(y.size())

fig, ax = plt.subplots(1, 2, figsize=(12, 9))
ax[0].imshow(x[0].reshape(128, 128, 3).detach().numpy(), cmap='viridis')
ax[0].set_title('$x$')
ax[1].imshow(y[0].reshape(64, 64, 1).detach().numpy(), cmap='viridis')
ax[1].set_title('$Conv3d(x^2)$')

plt.show()
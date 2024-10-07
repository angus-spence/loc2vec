import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import resnet50, resnet18, ResNet50_Weights, ResNet18_Weights

class Network(torch.nn.Module):
    def __init__(self, in_channels, debug=True, resnet: bool=False, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.debug = debug
        self.resnet = resnet
        if debug:
            self.dropout = nn.Dropout2d(0.5)
            self.relu = nn.ReLU()
            self.leak_relu = nn.LeakyReLU()
            self.relu_p = nn.PReLU()
            self.pool = nn.MaxPool2d(2, stride=2, padding=0)
            self.conv1 = nn.Conv2d(in_channels, 64, 1, stride=1, padding=1)
            self.conv2 = nn.Conv2d(64, 64, 1, stride=1, padding=1)
            self.conv3 = nn.Conv2d(64, 128, 1, stride=1, padding=1)
            self.conv4 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
            self.conv5 = nn.Conv2d(256, 256, 3, stride=2, padding=1)
            self.conv6 = nn.Conv2d(256, 128, 3, stride=2, padding=1)
            self.conv7 = nn.Conv2d(128, 64, 3, stride=2, padding=1)
            self.conv8 = nn.Conv2d(64, 32, 3, stride=2, padding=1)
            self.conv9 = nn.Conv2d(32, 16, 3, stride=2, padding=1)
            self.conv10 = nn.Conv2d(16, 8, 3, stride=2, padding=1)
            self.fc1 = nn.Linear(4, 1)
        elif resnet:
            self.model = resnet50(weights=ResNet50_Weights.DEFAULT)
            self.model.avgpool = nn.AvgPool2d(kernel_size=2, stride=1, padding=0)
            self.model.fc = nn.Linear(100352, 16)
        else:
            self.model = nn.Sequential(
            nn.Dropout2d(0.5),
            nn.Conv2d(in_channels, 64, 1, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, 1, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, stride=2, padding=0),
            nn.Conv2d(128, 256, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2, padding=0),
            nn.Conv2d(256, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 64, 3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(64),
            nn.Linear(4, 4)
        )

    def forward(self, x: torch.Tensor):
        """
        Forward pass for network

        Parameters
        ----------
        x: torch.Tensor
            Tensor for model input
        
        Returns
        -------
        x: torch.Tensor
            Tensor for model output
        """
        if self.debug:
            x = self.dropout(x)
            x = self.conv1(x)
            x = self.leak_relu(x)
            x = self.conv2(x)
            x = self.relu(x)
            x = self.conv3(x)
            x = self.pool(x)
            x = self.leak_relu(x)
            x = self.conv4(x)
            x = self.pool(x)
            x = self.relu(x)
            x = self.conv5(x)
            x = self.pool(x)
            x = self.leak_relu(x)
            x = self.conv6(x)
            x = self.pool(x)
            x = self.relu_p(x)
            x = self.conv7(x)
            x = self.pool(x)
            x = self.leak_relu(x)
            x = self.conv8(x)
            x = self.pool(x)
            x = self.relu(x)
            x = self.conv9(x)
            x = self.pool(x)
            x = self.relu(x)
            x = self.fc1(x)
            return x
        elif self.resnet: 
            x = self.model(x)
            return x
        else:
            return self.model(x)

class TripletLossFunction(nn.Module):
    """
    Evaluates triplet loss
    
    Parameters
    ----------
    margin: float
        margin for use in loss function
    """
    def __init__(self, margin=1.0):
        super(TripletLossFunction, self).__init__()
        self.margin = margin
    
    def forward(self, anchor_i: torch.Tensor, anchor_p: torch.Tensor, anchor_n: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for nn.Module

        Parameters
        ----------
        anchor_i: torch.Tensor
            Tensor for the x[i] sample
        anchor_p: torch.Tensor
            Tensor for (+) anchor for x[i]
        anchor_n: torch.Tensor
            Tensor for (-) anchor for x[i]
        
        Returns
        -------
        loss: torch.Tensor
            Tensor output from the Triplet Loss Funciton
        """
        distance_i_p = F.pairwise_distance(anchor_i, anchor_p)
        distance_i_n = F.pairwise_distance(anchor_i, anchor_n)
        distance_p_n = F.pairwise_distance(anchor_p, anchor_n)
        distance_min_neg = torch.min(distance_i_n, distance_p_n)
        losses = F.relu(distance_i_p - distance_min_neg + self.margin)
        
        np_losses = losses.cpu().data.numpy()
        np_distance_a_pos = np.mean(distance_i_p.cpu().data.numpy())
        np_distance_a_neg = np.mean(distance_i_n.cpu().data.numpy())
        np_min_neg_dist = np.mean(distance_min_neg.cpu().data.numpy())

        loss_log = f'MAX LOSS: {round(float(np.max(np_losses)),3)} | MEAN LOSS: {round(float(np.mean(np_losses)),3)} | (o)/(+) DIST: {round(float(np.mean(np_distance_a_pos)),3)} | (o)/(-) DIST: {round(float(np.mean(np_min_neg_dist)),3)}'

        return losses.mean(), loss_log, np_distance_a_pos, np_distance_a_neg, np_min_neg_dist
import torch
import torch.nn as nn


class BaseFeatureNetwork(nn.Module):
    def __init__(self):
        super().__init__()
    
    @property
    def embedding_dim(self):
        raise NotImplementedError
    
    @torch.no_grad()
    def encode_image(self, x):
        raise NotImplementedError
    
    def forward(self, x):
        return x
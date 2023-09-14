"""
@author Bryce Forrest
"""

import torch
import torch.nn as nn


class GMF(nn.Module):
    """
    Generalized matrix factorization. Uses PyTorch framework to generate latent feature matrices

    Xiangnan He, Lizi Liao, Hanwang Zhang, Liqiang Nie, Xia Hu, and Tat-Seng Chua. 2017. Neural Collaborative Filtering
    """

    def __init__(self, user_dim, item_dim, output_dim=1):
        super(GMF, self).__init__()
        self.name='gmf'
        self.u_embed = nn.Linear(user_dim, 16, dtype=torch.float32)
        self.v_embed = nn.Linear(item_dim, 16, dtype=torch.float32)

        self.layers = nn.Identity()
        self.dropout = nn.Dropout(0.1)
        
        self.output = nn.Sequential(
            nn.Linear(16, 1, dtype=torch.float32),
            nn.Sigmoid()
        )

    def forward(self, M_u, M_v):
        u_latent = self.u_embed(M_u)
        v_latent = self.v_embed(M_v)

        x = u_latent * v_latent
        
        x = self.layers(x)
        x = self.dropout(x)

        return self.output(x)

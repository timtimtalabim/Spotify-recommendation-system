"""
@author Bryce Forrest
"""

import torch
import torch.nn as nn

class MLP(nn.Module):
    """
    Multi-layer perceptron for recommendation

    Xiangnan He, Lizi Liao, Hanwang Zhang, Liqiang Nie, Xia Hu, and Tat-Seng Chua. 2017. Neural Collaborative Filtering
    """
    def __init__(self, user_dim, item_dim, output_dim=1):
        super(MLP, self).__init__()

        self.name='mlp'
        self.u_embed = nn.Linear(user_dim, 128, dtype=torch.float32)
        self.v_embed = nn.Linear(item_dim, 128, dtype=torch.float32)
        
        
        self.layers = nn.Sequential(
            nn.Linear(256, 128, dtype=torch.float32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64, dtype=torch.float32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32, dtype=torch.float32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32,16, dtype=torch.float32),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        
        self.output = nn.Sequential(
            nn.Linear(16, output_dim, dtype=torch.float32),
            nn.Sigmoid()
        )       
        
    def forward(self, M_u, M_v):
        u_latent = self.u_embed(M_u)
        v_latent = self.v_embed(M_v)
        if u_latent.shape != v_latent.shape:
            u_latent = u_latent.expand(v_latent.shape[0],-1)
        x = torch.concat((u_latent, v_latent),axis=1)
        
        x = self.layers(x)

        return self.output(x)
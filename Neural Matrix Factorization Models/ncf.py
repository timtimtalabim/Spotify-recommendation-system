"""
@author Bryce Forrest
"""

import torch
import torch.nn as nn 
from gmf import GMF
from mlp import MLP

class NCF(nn.Module):
    """
    Concatenation of GMF and NLP for recommendation
    
    Xiangnan He, Lizi Liao, Hanwang Zhang, Liqiang Nie, Xia Hu, and Tat-Seng Chua. 2017. Neural Collaborative Filtering
    """
    def __init__(self, user_dim, item_dim, gmf=None, mlp=None, alpha=0.5):
        super(NCF, self).__init__()
        self.name='nmf' if gmf is None else 'nmf_pt'
        self.alpha=alpha
        self.activation = {}
        self.gmf = GMF(user_dim, item_dim, output_dim=1) if gmf is None else gmf
        self.mlp = MLP(user_dim, item_dim, output_dim=1) if mlp is None else mlp
    
        self.hook_1 = self.gmf.layers.register_forward_hook(self.getActivation('gmf'))
        self.hook_2 = self.mlp.layers.register_forward_hook(self.getActivation('mlp'))
    
        self.layer = nn.Sequential(
            nn.Linear(32, 1, dtype=torch.float32),
            nn.Sigmoid()
        )        
    
    def forward(self, M_u, M_v):
        self.gmf(M_u, M_v)
        self.mlp(M_u, M_v)

        x = torch.concat((
            self.alpha*self.activation['gmf'], 
            (1-self.alpha)*self.activation['mlp']),axis=1
        )
        
        x = self.layer(x)

        return x
    
    def getActivation(self, name):
        # the hook signature
        def hook(model, input, output):
            self.activation[name] = output
        return hook
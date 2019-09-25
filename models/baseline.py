import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.non_local import SpatialNL
#from non_local import SpatialNL


class model(nn.Module):
    def __init__(self):
        super(model, self).__init__()

        self.Encoder = nn.Sequential(
            nn.Conv1d(6, 10, 15, 3),
            nn.BatchNorm1d(10),
            nn.Dropout(),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=5)
        )
        
        self.non_local = SpatialNL(10,10)

        self.FC_layer = nn.Sequential(
            nn.Linear(650, 50),
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(50, 1)
        )
        

    def forward(self, vital):
        latent = self.Encoder(vital)
        non_local = self.non_local(latent)
        flat_bottle = latent.flatten(start_dim=1)
        output = self.FC_layer(flat_bottle)
        prob = F.sigmoid(output/2.0)
        return prob

if __name__ == '__main__':
    m = model()
    latent = m.Encoder(torch.from_numpy(np.random.rand(100,6,3000)).float())
    non_local = m.non_local(latent)
    print(non_local.shape)
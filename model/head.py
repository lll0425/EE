import torch.nn as nn

class TxHead(nn.Module):
    def __init__(self, emb_dim: int = 512, num_classes: int = 36, hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.PReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden, num_classes),
        )
    def forward(self, emb):
        return self.net(emb)
import torch
import torch.nn as nn

NUM_FEATURES = 91


class MLPPolicy(nn.Module):
    """Scores each candidate branching variable independently from its 91-dim
    feature vector; shared weights across candidates within a node (a node's
    branching decision is argmax over these per-candidate scores)."""

    def __init__(self, num_features=NUM_FEATURES, hidden_dims=(128, 128, 64)):
        super().__init__()
        dims = [num_features, *hidden_dims, 1]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

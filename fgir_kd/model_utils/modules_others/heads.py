from torch import nn
from einops.layers.torch import Reduce, Rearrange


class Head(nn.Module):
    def __init__(self, classifier, hidden_size, num_classes, shape_format='bsd'):
        super().__init__()

        if classifier == 'cls':
            self.head = nn.Linear(hidden_size, num_classes)
        else:
            self.head_pool = Reduce('b s d -> b d', 'mean')
            self.head = nn.Linear(hidden_size, num_classes)

        if shape_format == 'bdhw':
            self.rearrange = Rearrange('b d h w -> b (h w) d')
        elif shape_format == 'bhwd':
            self.rearrange = Rearrange('b h w d -> b (h w) d')

    def forward(self, x):
        # x shape: B (batch size), S (sequence length), D (hidden dim size) or B, D, H, W
        if hasattr(self, 'rearrange'):
            x = self.rearrange(x)

        if hasattr(self, 'head_pool'):
            x = self.head_pool(x)
            x = self.head(x)
        elif hasattr(self, 'head'):
            x = self.head(x[:, 0, :])

        return x

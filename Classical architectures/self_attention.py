# https://peterbloem.nl/blog/transformers

# in pytorch, complete self-attention

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, k, heads=4, mask=False):
        super().__init__()
        assert k % heads == 0, "Embedding size must be divisible by number of heads"
        self.k, self.heads, self.mask = k, heads, mask
        # these compute the queries, keys and values for all heads
        self.tokeys = nn.Linear(k, k, bias=False)
        self.toqueries = nn.Linear(k, k, bias=False)
        self.tovalues = nn.Linear(k, k, bias=False)

        # this will be applied after the multi-head self attention operation
        self.unifyheads = nn.Linear(k, k)

    def forward(self, x):
        b, t, k = x.size()
        h = self.heads
        queries = self.toqueries(x)
        keys    = self.tokeys(x)
        values  = self.tovalues(x)

        s = k // h # split size
        # split each head into h pieces / chunks
        keys    = keys.view(b, t, h, s)
        queries = queries.view(b, t, h, s)
        values  = values.view(b, t, h, s)

        # fold heads into the batch dimension
        keys    = keys.transpose(1, 2).contiguous().view(b * h, t, s)
        queries = queries.transpose(1, 2).contiguous().view(b * h, t, s)
        values  = values.transpose(1, 2).contiguous().view(b * h, t, s)

        # get dot product of queries and keys for each head, and scale
        dot = torch.bmm(queries, keys.transpose(1, 2))
        # dot has size (b*h, t, t) containing raw weights

        # scale the dot products by the dimensionality
        dot = dot / np.sqrt(k)

        # normalize
        dot = F.softmax(dot, dim=2)
        # dot now contains row-wise normalized weights

        # apply the self attention to the values
        out = torch.bmm(dot, values).view(b, h, t, s)
        # swap h, t back, unify heads
        out = out.transpose(1, 2).contiguous().view(b, t, s*k)
        return self.unifyheads(out)


# the transformer block
class TransformerBlock(nn.Module):
    def __init__(self, k, heads):
        super().__init__()
        self.attention = SelfAttention(k, heads=heads)
        self.norm1 = nn.LayerNorm(k)  # ???
        self.norm2 = nn.LayerNorm(k)

        self.ff = nn.Sequential(
            nn.Linear(k, 4*k),
            nn.ReLU(),
            nn.Linear(4*k, k))

    def forward(self, x):
        attended = self.attention(x)
        x = self.norm1(attended + x)
        feedforward = self.ff(x)
        return self.norm2(feedforward + x)
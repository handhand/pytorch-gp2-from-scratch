import torch
import torch.nn as nn
from MyGPT2.gelu import Gelu

# ref: https://learning.oreilly.com/library/view/build-a-large/9781633437166/OEBPS/Text/chapter-4.html#p109
class FeedForward(nn.Module):
    """
    The linear layer shape is (feature, output_dim)
    The input of attention is (batch, token_num, dim)
    According to https://pytorch.org/docs/stable/generated/torch.nn.Linear.html, feature=dim then will work, 
    and output is (batch, token_num, output_dim)
    
    So the final output of this module is same with input
    """
    def __init__(self, emb_dim):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 4),
            Gelu(),
            nn.Linear(emb_dim * 4, emb_dim),
        )

    def forward(self, x):
        return self.layer(x)
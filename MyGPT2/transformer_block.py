import torch.nn as nn
from MyGPT2.layer_norm import LayerNorm
from MyGPT2.multihead_attention import MultiheadAttention
from MyGPT2.feed_forward import FeedForward

# ref: https://learning.oreilly.com/library/view/build-a-large/9781633437166/OEBPS/Text/chapter-4.html#p143
class TransformerBlock(nn.Module):
    '''
    Layer norm -> mha -> dropout -> shortcut -> layer norm -> feed forward -> dropout -> shortcut
    '''
    def __init__(self, cfg):
        super().__init__()
        emb_dim = cfg["emb_dim"]
        self.layer_norm1 = LayerNorm(emb_dim) # use our custom LayerNorm, NOT nn.LayerNorm
        heads = cfg["n_heads"]
        emb_dim = cfg["emb_dim"]
        assert (emb_dim % heads) == 0, 'Embedding必须可以被head整除，用来计算head的dimension'
        self.mha = MultiheadAttention(
            input_dim=emb_dim,
            heads = heads,
            head_dim = int(emb_dim / heads), # 注意类型必须是int
            drop_rate=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"],
            context_length=cfg["context_length"]
        )
        self.drop_out = nn.Dropout(cfg["drop_rate"])
        self.layer_norm2 = LayerNorm(emb_dim)
        self.feed_forward = FeedForward(emb_dim=emb_dim)

    def forward(self, x):
        shortcut = x
        x = self.layer_norm1(x)
        x = self.mha(x)
        x = self.drop_out(x)
        x = shortcut + x

        shortcut = x
        x = self.layer_norm2(x)
        x = self.feed_forward(x)
        x = self.drop_out(x)
        x = shortcut + x
        return x
import torch
import torch.nn as nn
from MyGPT2.transformer_block import TransformerBlock
from MyGPT2.layer_norm import LayerNorm

# ref: https://learning.oreilly.com/library/view/build-a-large/9781633437166/OEBPS/Text/chapter-4.html#p171
class GPT2Model(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, x):
        _, seq_length = x.shape
        
        pos_in = torch.arange(seq_length, device=x.device) # shape: [seq_length]
        pos_embeds = self.pos_emb(pos_in) # batch dim will be broadcasted

        tok_embeds = self.tok_emb(x)

        x = pos_embeds + tok_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        return self.out_head(x)
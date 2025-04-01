import torch

class LayerNorm(torch.nn.Module):
    """
    normalization一般没有trainable参数;
    embedding_dim用在对normalized之后的数值进行scale和shift
    """
    def __init__(self, embedding_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = torch.nn.Parameter(torch.ones(embedding_dim)) # 初始值为[1, 1, 1, ... 1], 即对norm值没影响
        self.shift = torch.nn.Parameter(torch.zeros(embedding_dim)) # 初始值为[0, 0, 0, ... 0], 即对norm值没影响

    def forward(self, x):
        # 注意这里dim = -1，即计算平均值的是embedding的维度，
        # 是一个token自己的各个embedding之间进行normalization，而不是batch normalization
        mean = x.mean(dim = -1, keepdim = True) # shape = [batch, token, 1], x - mean时1可以broadcast
        var = x.var(dim = -1, keepdim = True)
        x = (x - mean)/torch.sqrt(var + self.eps)
        return x * self.scale + self.shift
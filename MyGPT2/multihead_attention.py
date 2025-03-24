import torch
import torch.nn as nn

class MultiheadAttention(torch.nn.Module):
    '''
    https://learning.oreilly.com/library/view/build-a-large/9781633437166/OEBPS/Text/chapter-3.html#p284
    输出和输入的shape应该相同，这样才可以多个叠在一起
    shape = [batch, token, embedding = (head数量 * head_dim)]

    context_length在这里不影响参数数量, 主要用于生成 attention mask
    '''

    def __init__(self, input_dim, heads, head_dim, drop_rate, qkv_bias, context_length):
        super().__init__()
        # heads * head_dim 必须等于 input_dim
        self.heads = heads
        self.head_dim = head_dim
        # print(input_dim)
        # print(heads)
        # print(head_dim)
        self.q_proj = nn.Linear(input_dim, heads * head_dim, bias=qkv_bias) # typically embedding = input_dim = heads*head_dim = output_dim
        self.k_proj = nn.Linear(input_dim, heads * head_dim, bias=qkv_bias)
        self.v_proj = nn.Linear(input_dim, heads * head_dim, bias=qkv_bias)
        self.dropout = nn.Dropout(drop_rate)

        # something like
        # [[0,1,1],
        #  [0,0,1],
        #   0,0,0]]
        self.register_buffer(
            "mask",
            torch.triu(
                torch.ones(context_length, context_length),
                diagonal=1
            )
        )
        self.proj = nn.Linear(heads * head_dim, heads * head_dim)

    def forward(self, x):
        batch, tokens, embedding = x.shape
        q = self.q_proj(x) # x shape为[batch, token, embedding], Linear作用于最后一维embedding
        k = self.k_proj(x)
        v = self.v_proj(x)
        # embedding 拆成 heads 和 head_dim
        q = q.view(batch, tokens, self.heads, self.head_dim)
        k = k.view(batch, tokens, self.heads, self.head_dim)
        v = v.view(batch, tokens, self.heads, self.head_dim)
        # 再转换成多头的形式, shape -> [batch, heads, tokens, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # 计算各个头的注意力
        # [batch, heads, tokens, head_dim] @ [batch, heads, head_dim, tokens] = [batch, heads, tokens, tokens]
        att_scores = q @ k.transpose(2, 3)

        # Causal Attention Mask:
        # 不可以注意未来的出现的字，所以要把未来的字的score设置为负无穷
        # apply后类似这样，再算softmax, -inf的地方就为0了
        # [[12.123,  -inf,   -inf...],
        #  [2.123,  4.123,  -inf...],
        #   3.123,  6.123,  7.123]]
        mask = self.mask.bool()[:tokens, :tokens]
        att_scores.masked_fill_(mask, -torch.inf)

        att_weights = torch.softmax(att_scores/(self.head_dim ** 0.5), dim=-1) # scaled dot product
        att_weights = self.dropout(att_weights)
        # 最终类似这样, 每一行表示某个token对其他token的注意力, 比如第一个token只会注意自己, 不能注意之后的词:
        # [[1.0,    0,      0],
        #  [0.1,    0.9,    0],
        #   0.1,    0.2,    0.7]]

        # [batch, heads, tokens, tokens] @ [batch, heads, tokens, head_dim] = [batch, heads, tokens, head_dim]
        context_vec = att_weights @ v

        # 重新转换成一个大matrix，传给下一个Module
        context_vec = context_vec.transpose(1, 2)
        context_vec = context_vec.contiguous().view(batch, tokens, self.heads * self.head_dim)

        # 最后再来个projection
        context_vec = self.proj(context_vec)

        return context_vec

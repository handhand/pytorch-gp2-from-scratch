import torch

def text_to_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    # 加多一个batch的维度，用于兼容LLM的输入shape
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor


def ids_to_text(ids, tokenizer):
    return tokenizer.decode(ids.squeeze(0).tolist())


def generate_text_simple(
        model,
        idx,
        max_new_tokens,
        context_size):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        next_id = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True) # shape is [batch, 1], so that can concat with previous ids
        idx = torch.cat((idx, next_id), dim=-1)
    return idx
    
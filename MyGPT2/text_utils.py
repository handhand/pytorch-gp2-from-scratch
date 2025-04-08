import torch

def text_to_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    # 加多一个batch的维度，用于兼容LLM的输入shape
    return encoded_tensor

def ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0) # remove batch dimension
    return tokenizer.decode(flat.tolist())


def generate_text_simple(
        model,
        idx,
        max_new_tokens,
        context_size,
        end_of_text_id = 50256):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        next_id = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True) # shape is [batch, 1], so that can concat with previous ids
        idx = torch.cat((idx, next_id), dim=-1)
        if next_id == end_of_text_id:
            break
    return idx
    
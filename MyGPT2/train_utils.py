import torch
import numpy as np

# https://learning.oreilly.com/library/view/build-a-large/9781633437166/OEBPS/Text/chapter-5.html#p148
def train_model_simple(model, train_loader, val_loader,
                       optimizer, device, num_epochs,
                       eval_freq, eval_iter, start_context, tokenizer):
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(
                input_batch, target_batch, model, device
            )
            loss.backward()
            optimizer.step()
            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, "
                      f"Val loss {val_loss:.3f}"
                )

        # generate_and_print_sample(
        #     model, tokenizer, device, start_context
        # )
    return train_losses, val_losses, track_tokens_seen


def calc_loss_batch(input_batch, target_batch, model, device):
    '''
    given the input and target, return the loss
    '''
    input_batch.to(device)
    target_batch.to(device)
    # ouput(logits) shape:  [batch, token, vocab]
    # target shape: [batch, token]
    
    # cross_entropy需要的input shape为[batch_size, num_classes] (类别在第二维)
    # target的shape为[batch_size]
    # 所以这里做了转换符合shape
    # 另外reduction默认为mean，即返回的loss是一个标量
    logits = model(input_batch) 
    return torch.nn.functional.cross_entropy(
        input=logits.flatten(0,1), # shape为 [batch*token, vocab_size(类别数量)]
        target=target_batch.flatten() # shape为 [batch*token]
    )


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches = eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches = eval_iter)
    model.train()
    return train_loss, val_loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    '''
    load the data in data_loader, and calculate the average loss.
    use calc_loss_batch internally
    '''
    total_loss = 0.
    if len(data_loader) == 0:
        raise "no data in loader"
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(len(data_loader), num_batches)

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i > num_batches:
            break
        loss = calc_loss_batch(input_batch, target_batch, model, device)
        total_loss += loss.item()
    return total_loss / num_batches


def assign(left, right):
    '''
    right是numpy array；
    保证right的shape和left相等，然后将right作为pytorch parameter返回
    '''
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, "
                          "Right: {right.shape}"
        )
    return torch.nn.Parameter(torch.tensor(right))


def load_weights_into_gpt(gpt, params):
    '''
    加载官方的gpt参数到我们的模型中
    gpt: pytorch模型
    params: 从官方下载的参数，是一个字典，key是不同的层，value是层的参数
    '''
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params["wpe"])
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params["wte"])

    for b in range(len(params["blocks"])):
        q_w, k_w, v_w = np.split((params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
        gpt.trf_blocks[b].mha.q_proj.weight = assign(gpt.trf_blocks[b].mha.q_proj.weight, q_w.T)
        gpt.trf_blocks[b].mha.k_proj.weight = assign(gpt.trf_blocks[b].mha.k_proj.weight, k_w.T)
        gpt.trf_blocks[b].mha.v_proj.weight = assign(gpt.trf_blocks[b].mha.v_proj.weight, v_w.T)

        q_b, k_b, v_b = np.split((params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
        gpt.trf_blocks[b].mha.q_proj.bias = assign(gpt.trf_blocks[b].mha.q_proj.bias, q_b)
        gpt.trf_blocks[b].mha.k_proj.bias = assign(gpt.trf_blocks[b].mha.k_proj.bias, k_b)
        gpt.trf_blocks[b].mha.v_proj.bias = assign(gpt.trf_blocks[b].mha.v_proj.bias, v_b)

        gpt.trf_blocks[b].mha.proj.weight = assign(
            gpt.trf_blocks[b].mha.proj.weight, 
            params["blocks"][b]["attn"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].mha.proj.bias = assign(
            gpt.trf_blocks[b].mha.proj.bias, 
            params["blocks"][b]["attn"]["c_proj"]["b"])
         
        # 下边feed forward layers取0和2的是因为 feedforward是 sequential，
        # 第0，2是Linear，第1层是Gelu
        gpt.trf_blocks[b].feed_forward.layer[0].weight = assign(
            gpt.trf_blocks[b].feed_forward.layer[0].weight, 
            params["blocks"][b]["mlp"]["c_fc"]["w"].T)
        gpt.trf_blocks[b].feed_forward.layer[0].bias = assign(
            gpt.trf_blocks[b].feed_forward.layer[0].bias, 
            params["blocks"][b]["mlp"]["c_fc"]["b"])
        gpt.trf_blocks[b].feed_forward.layer[2].weight = assign(
            gpt.trf_blocks[b].feed_forward.layer[2].weight, 
            params["blocks"][b]["mlp"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].feed_forward.layer[2].bias = assign(
            gpt.trf_blocks[b].feed_forward.layer[2].bias, 
            params["blocks"][b]["mlp"]["c_proj"]["b"])
        
        gpt.trf_blocks[b].layer_norm1.scale = assign(
            gpt.trf_blocks[b].layer_norm1.scale, 
            params["blocks"][b]["ln_1"]["g"])
        gpt.trf_blocks[b].layer_norm1.shift = assign(
            gpt.trf_blocks[b].layer_norm1.shift, 
            params["blocks"][b]["ln_1"]["b"])
        gpt.trf_blocks[b].layer_norm2.scale = assign(
            gpt.trf_blocks[b].layer_norm2.scale, 
            params["blocks"][b]["ln_2"]["g"])
        gpt.trf_blocks[b].layer_norm2.shift = assign(
            gpt.trf_blocks[b].layer_norm2.shift, 
            params["blocks"][b]["ln_2"]["b"])
    
    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])
    
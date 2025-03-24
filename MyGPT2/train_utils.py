import torch

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


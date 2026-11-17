"""Training and validation helpers; no notebook global dependencies."""
import torch
from tqdm.auto import tqdm
from .transformer_model import create_mask


def run_epoch(model, dataloader, loss_fn, device, optimizer=None, pad_idx=1):
    """Run one training or validation epoch; return average batch loss."""
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    if len(dataloader) == 0:
        raise ValueError('Dataloader is empty')
    with torch.enable_grad() if training else torch.no_grad():
        for src, tgt in tqdm(dataloader, total=len(dataloader), desc='Train' if training else 'Validate'):
            src, tgt = src.to(device), tgt.to(device)
            tgt_input, tgt_out = tgt[:, :-1], tgt[:, 1:]
            src_mask, tgt_mask, src_pad, tgt_pad = create_mask(src, tgt_input, pad_idx)
            logits = model(src, tgt_input, src_mask, tgt_mask, src_pad, tgt_pad, src_pad)
            loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item()
    return total_loss / len(dataloader)


def train_epoch(model, optimizer, dataloader, loss_fn, device, pad_idx=1):
    return run_epoch(model, dataloader, loss_fn, device, optimizer, pad_idx)


def evaluate(model, dataloader, loss_fn, device, pad_idx=1):
    return run_epoch(model, dataloader, loss_fn, device, None, pad_idx)

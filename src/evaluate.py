"""Greedy EN -> FR translation and sentence-level BLEU evaluation."""
import torch
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from .transformer_model import generate_square_subsequent_mask

@torch.no_grad()
def greedy_decode(model, src, src_mask, max_len, start_symbol, eos_idx=3):
    device = src.device
    memory = model.encode(src, src_mask)
    ys = torch.full((1, 1), start_symbol, dtype=torch.long, device=device)
    for _ in range(max_len - 1):
        tgt_mask = generate_square_subsequent_mask(ys.size(1), device=device)
        out = model.decode(ys, memory, tgt_mask)
        next_word = model.generator(out[:, -1, :]).argmax(dim=-1).item()
        ys = torch.cat([ys, torch.tensor([[next_word]], device=device)], dim=1)
        if next_word == eos_idx:
            break
    return ys

@torch.no_grad()
def translate(model, src_sentence, text_transform, vocab_transform, device,
              src_language='en', tgt_language='fr', bos_idx=2, eos_idx=3):
    model.eval()
    src = text_transform[src_language](src_sentence).unsqueeze(0).to(device)
    src_mask = torch.zeros((src.shape[1], src.shape[1]), dtype=torch.bool, device=device)
    tokens = greedy_decode(model, src, src_mask, src.shape[1] + 5, bos_idx, eos_idx)
    decoded = vocab_transform[tgt_language].lookup_tokens(tokens[0].tolist())
    return ' '.join(token for token in decoded if token not in ('<bos>', '<eos>', '<pad>'))

def mean_sentence_bleu(model, test_frame, text_transform, vocab_transform, device):
    """Mean smoothed sentence BLEU on a dataframe; scale 0..1."""
    smoothing = SmoothingFunction().method1
    scores = []
    for _, row in test_frame.iterrows():
        predicted = translate(model, row['English words/sentences'],
                              text_transform, vocab_transform, device)
        scores.append(sentence_bleu([row['French words/sentences'].split()],
                                    predicted.split(), smoothing_function=smoothing))
    return sum(scores) / len(scores) if scores else float('nan')

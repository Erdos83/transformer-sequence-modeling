"""Sequence-to-sequence Transformer architecture from the original notebook."""
import math
import torch
from torch import Tensor
from torch import nn
from torch.nn import Transformer


def generate_square_subsequent_mask(size, device=None):
    """Causal additive attention mask (0 for visible, -inf for future)."""
    return torch.triu(torch.full((size, size), float('-inf'), device=device), diagonal=1)


def create_mask(src, tgt, pad_idx=1):
    """Batch-first attention and padding masks."""
    src_len, tgt_len = src.shape[1], tgt.shape[1]
    src_mask = torch.zeros((src_len, src_len), device=src.device, dtype=torch.bool)
    tgt_mask = generate_square_subsequent_mask(tgt_len, device=tgt.device)
    return src_mask, tgt_mask, src.eq(pad_idx), tgt.eq(pad_idx)

class PositionalEncoding(nn.Module):
  def __init__(self, d_model, dropout, max_len=5000):
    """
    :param max_len: Input length sequence.
    :param d_model: Embedding dimension.
    :param dropout: Dropout value (default=0.1)
    """
    super(PositionalEncoding, self).__init__()
    self.dropout = nn.Dropout(p=dropout)

    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    pe = pe.unsqueeze(0)
    self.register_buffer('pe', pe)

  def forward(self, x):
    """
    Inputs of forward function
    :param x: the sequence fed to the positional encoder model (required).
    Shape:
        x: [sequence length, batch size, embed dim]
        output: [sequence length, batch size, embed dim]
    """

    x = x + self.pe[:, :x.size(1)]
    return self.dropout(x)

class TokenEmbedding(nn.Module):
  def __init__(self, vocab_size: int, emb_size):
    """
    :param vocab_size: Size of the vocabulary.
    :param emb_size: Embedding dimension.
    """
    super(TokenEmbedding, self).__init__()
    self.embedding = nn.Embedding(vocab_size, emb_size)
    self.emb_size = emb_size

  def forward(self, tokens: Tensor):
    """
    :param tokens: Tensor of token indices.
    :return: Tensor of embeddings.
    """
    return self.embedding(tokens.long()) * math.sqrt(self.emb_size)

class TransformerSeq(nn.Module):
  """
  Transformer for sequence-to-sequence tasks.
  """
  def __init__(
    self,
    num_encoder_layers: int,
    num_decoder_layers: int,
    emb_size: int,
    nhead: int,
    src_vocab_size: int,
    tgt_vocab_size: int,
    dim_feedforward: int = 512,
    dropout: float = 0.2
):
    super(TransformerSeq, self).__init__()
    self.transformer = Transformer(
        d_model=emb_size,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        batch_first=True
    )
    self.generator = nn.Linear(emb_size, tgt_vocab_size)
    self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
    self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
    self.positional_encoding = PositionalEncoding(
        emb_size, dropout=dropout)

  def forward(self,
            src: Tensor,
            trg: Tensor,
            src_mask: Tensor,
            tgt_mask: Tensor,
            src_padding_mask: Tensor,
            tgt_padding_mask: Tensor,
            memory_key_padding_mask: Tensor):
    src_emb = self.positional_encoding(self.src_tok_emb(src))
    tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))
    outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, None,
                            src_padding_mask, tgt_padding_mask, memory_key_padding_mask)
    return self.generator(outs)

  def encode(self, src: Tensor, src_mask: Tensor):
    """
    Encode the source sequence.
    :param src: Source sequence tensor.
    :param src_mask: Source sequence mask.
    :return: Encoded source sequence tensor.
    """
    return self.transformer.encoder(self.positional_encoding(
                        self.src_tok_emb(src)), src_mask)

  def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
    """
    Decode the target sequence.
    :param tgt: Target sequence tensor.
    :param memory: Encoded source sequence tensor.
    :param tgt_mask: Target sequence mask.
    :return: Decoded target sequence tensor.
    """
    return self.transformer.decoder(self.positional_encoding(
                      self.tgt_tok_emb(tgt)), memory,
                      tgt_mask)
"""Dataset, tokenization, vocabulary, and batch preparation for EN -> FR."""
import ast
import json
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

SRC_LANGUAGE, TGT_LANGUAGE = 'en', 'fr'
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
SPECIAL_SYMBOLS = ['<unk>', '<pad>', '<bos>', '<eos>']

# Custom Dataset class.
class TranslationDataset(Dataset):
  """
  Custom Dataset class for translation data.
  """
  def __init__(self, csv):
    """
    :param csv: Pandas dataframe containing translation data.
    """
    self.csv = csv

  def __len__(self):
    """
    :return: Length of the dataset.
    """
    return len(self.csv)

  def __getitem__(self, idx):
    """
    :param idx: Index of the sample to retrieve.
    :return: Tuple containing English and French translation.
    """
    return(
      self.csv['English words/sentences'].iloc[idx],
      self.csv['French words/sentences'].iloc[idx]
    )

def load_translation_csv(path):
    """Read a CSV with 'translation' mappings or prepared English/French columns.

    Uses literal parsing rather than executing untrusted CSV contents with eval().
    """
    df = pd.read_csv(path)
    if 'translation' in df.columns:
        def parse(value):
            if isinstance(value, dict):
                return value
            try:
                return json.loads(value)
            except (ValueError, TypeError):
                return ast.literal_eval(value)
        translations = df['translation'].map(parse)
        df['English words/sentences'] = translations.map(lambda x: x['en'])
        df['French words/sentences'] = translations.map(lambda x: x['fr'])
        df = df.drop(columns=['translation'])
    required = {'English words/sentences', 'French words/sentences'}
    if not required.issubset(df.columns):
        raise ValueError(f'Missing required columns: {required - set(df.columns)}')
    return df


def build_text_pipeline(train_dataset):
    """Build tokenizers, vocabularies, text transforms and collate function."""
    token_transform = {
        SRC_LANGUAGE: get_tokenizer('spacy', language='en_core_web_sm'),
        TGT_LANGUAGE: get_tokenizer('spacy', language='fr_core_news_sm'),
    }
    vocab_transform = {}
    for language, index in [(SRC_LANGUAGE, 0), (TGT_LANGUAGE, 1)]:
        def yield_tokens():
            for pair in train_dataset:
                yield token_transform[language](pair[index])
        vocab = build_vocab_from_iterator(yield_tokens(), min_freq=1,
                                          specials=SPECIAL_SYMBOLS, special_first=True)
        vocab.set_default_index(UNK_IDX)
        vocab_transform[language] = vocab

    def tensor_transform(token_ids):
        return torch.cat((torch.tensor([BOS_IDX]), torch.tensor(token_ids),
                          torch.tensor([EOS_IDX])))

    text_transform = {
        language: lambda sentence, language=language: tensor_transform(
            vocab_transform[language](token_transform[language](sentence)))
        for language in (SRC_LANGUAGE, TGT_LANGUAGE)
    }

    def collate_fn(batch):
        src_batch = [text_transform[SRC_LANGUAGE](src.rstrip('\n')) for src, _ in batch]
        tgt_batch = [text_transform[TGT_LANGUAGE](tgt.rstrip('\n')) for _, tgt in batch]
        return (pad_sequence(src_batch, padding_value=PAD_IDX, batch_first=True),
                pad_sequence(tgt_batch, padding_value=PAD_IDX, batch_first=True))
    return token_transform, vocab_transform, text_transform, collate_fn

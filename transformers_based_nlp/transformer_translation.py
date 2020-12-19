import torch
import torch.nn as nn
import torch.optim as optim
import spacy
from torch.utils.tensorboard import SummaryWriter
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator
import os
import pytorch_lightning as pl

try:
    spacy_eng = spacy.load("en")
    spacy_ger = spacy.load("de")

except Exception as e:
    os.system("python3 -m spacy download en")
    os.system("python3 -m spacy download de")
    spacy_eng = spacy.load("en")
    spacy_ger = spacy.load("de")


def tokenizer_ger(text):
    return [tok.text for tok in spacy_ger.tokenizer(text)]


def tokenizer_eng(text):
    return [tok.text for tok in spacy_eng.tokenizer(text)]


print("=>>> token done")

german = Field(
    tokenize=tokenizer_ger,
    lower=True,
    init_token="<sos>",
    eos_token="<eos>",
)

english = Field(
    tokenize=tokenizer_eng,
    lower=True,
    init_token="<sos>",
    eos_token="<eos>",
)

print("=>>> field done")

train_data, valid_data, test_data = Multi30k.splits(
    exts=(".de", ".en"), fields=(german, english)
)

german.build_vocab(train_data, max_size=10000, min_freq=2)
english.build_vocab(train_data, max_size=10000, min_freq=2)


class Transformer(pl.LightningModule):
    def __init__(
        self,
        embedding_size,
        src_vocab_size,
        trg_vocab_size,
        src_pad_idx,
        num_heads,
        num_encoder_layers,
        num_decoder_layers,
        forward_expansion,
        dropout,
        max_len,
        device,
    ):
        super(Transformer, self).__init__()
        self.src_word_embedding = nn.Embedding(src_vocab_size, embedding_size)
        self.src_position_embedding = nn.Embedding(max_len, embedding_size)
        self.trg_word_embedding = nn.Embedding(trg_vocab_size, embedding_size)
        self.trg_position_embedding = nn.Embedding(max_len, embedding_size)
        self.device = device

        self.transformer = nn.Transformer(
            embedding_size,
            num_heads,
            num_encoder_layers,
            num_decoder_layers,
            forward_expansion,
            dropout,
        )

        self.fc_out = nn.Linear(embedding_size, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.src_pad_idx = src_pad_idx

    def make_src_mask(self, src):
        # src_shape: (src_len, N)
        # pytorch wants it in the format of (N, src_len), so do a transpose
        src_mask = src.transpose(0, 1) == self.src_pad_idx
        return src_mask

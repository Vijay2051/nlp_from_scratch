import torch
import torch.nn as nn
from torch.nn.modules import dropout
import torch.optim as optim
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator, TabularDataset
import numpy as np
import spacy
import os
import random

# constants
FOLDER_PATH = "input/translation"

try:
    spacy_eng = spacy.load("en")
    spacy_ger = spacy.load("de")

except Exception as e:
    os.system("python3 -m spacy download en")
    os.system("python3 -m spacy download de")
    spacy_eng = spacy.load("en")
    spacy_ger = spacy.load("de")


# spacy tokenizer
def tokenizer_ger(text):
    return [tok.text for tok in spacy_ger.tokenizer(text)]


def tokenizer_eng(text):
    return [tok.text for tok in spacy_eng.tokenizer(text)]


german = Field(
    tokenize=tokenizer_ger,
    sequential=True,
    use_vocab=True,
    lower=True,
    init_token="<sos>",
    eos_token="<eos>",
)

english = Field(
    tokenize=tokenizer_eng,
    sequential=True,
    use_vocab=True,
    lower=True,
    init_token="<sos>",
    eos_token="<eos>",
)

# remove three double quotes for the large en de dataset
"""

# load the data
# create fields in json format for the tabular data
fields = {"english": ("eng", english), "german": ("ger", german)}

# split the dataset into train and test from the dataset
train_data, validation_data = TabularDataset.splits(
    path= FOLDER_PATH,
    train="en_de_train.csv",
    validation="en_de_val.csv",
    format="csv",
    fields=fields)

"""

train_data, val_data, test_data = Multi30k.splits(
    exts=(".de", ".en"), fields=(german, english)
)

# build the vocab
english.build_vocab(train_data, max_size=10000, min_freq=2)
german.build_vocab(train_data, max_size=10000, min_freq=2)


class Encoder(nn.Module):
    """
    : input_size => vocab_size(mostly the vocab size of the german lang)
    : embedding_size => size of (each word is mapped to some D dimensional space)
    : hidden_size => outputs from each state of lstm, hidden the size of the encoder and decoder are the same
    : num_layers => num layers in lstm stack
    : dropout => dropout layer
    """

    def __init__(self, input_size, embedding_size, hidden_size, num_layers, dropout, p):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(p)
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=p)

    def forward(self, x):
        # x.shape: seq_len, N
        embedding = self.dropout(self.embedding(x))
        # after embedding = seq_len, N, embedding_size
        outputs, (hidden, cell) = self.rnn(embedding)
        return hidden, cell


class Decoder(nn.Module):
    """
    : input size is gonna be the size of the english vocab
    : output size is gonna be the same as the input size
    : hidden_size of the encoder and decoder are the same
    """

    def __init__(
        self, input_size, embedding_size, hidden_size, output_size, num_layers, p
    ):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(p)
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=p)
        self.fc = nn.Linear(
            hidden_size, output_size
        )  # the output size is the same as the input size

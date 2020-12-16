import os
import random

import numpy as np
import spacy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.modules import dropout
from torch.utils.tensorboard import SummaryWriter
from torchtext.data import BucketIterator, Field, TabularDataset
from torchtext.datasets import Multi30k

from utils import bleu, load_checkpoint, save_checkpoint, translate_sentence

# constants
FOLDER_PATH = "../input/translation"

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

# remove three double quotes for the large en de dataset

# load the data
# create fields in json format for the tabular data
fields = [("eng", english), ("ger", german)]

# split the dataset into train and test from the dataset
train_data, validation_data = TabularDataset.splits(
    path=FOLDER_PATH,
    train="en_de_train.csv",
    validation="en_de_val.csv",
    format="csv",
    fields=fields,
)

print("=>>>>>>>>>>> split complete")

# build the vocab
english.build_vocab(train_data, max_size=10000, min_freq=2)
german.build_vocab(train_data, max_size=10000, min_freq=2)

print("=>>>>>>>>>>> vocab building complete")


class Encoder(nn.Module):
    """
    : input_size => vocab_size(mostly the vocab size of the german lang)
    : embedding_size => size of (each word is mapped to some D dimensional space)
    : hidden_size => outputs from each state of lstm, hidden the size of the encoder and decoder are the same
    : num_layers => num layers in lstm stack
    : dropout => dropout layer
    """

    def __init__(self, input_size, embedding_size, hidden_size, num_layers, p):
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
    : decoder takes hidden and the cell which are the context vectors coming out of the encoder
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

    def forward(self, x, hidden, cell):
        """
        : since we have sent the whole sequence of sentence into the encoder (seq_len, N),
            but the decoder will predict the values word by word.
            So, the input shape of the x for the decoder should be (1, N) => one word at a time,
            and the the output from this word form the sequence for the next word.
        : The above method could be easily done by unsqueezing the the x and adding a single dimension to it
        """
        x = x.unsqueeze(0)
        embedding = self.dropout(self.embedding(x))
        # x.shape => (1, N, embedding_size)
        outputs, (hidden, cell) = self.rnn(embedding, (hidden, cell))
        # x.shape => (1, N, hidden_size)
        predictions = self.fc(outputs)
        # preds.shape => (1, N, len_of_vocab(output_size))
        """
            : we dont want that 1 at the 0 th dimension of the predictions, so squeeze them out
        """
        predictions = predictions.squeeze(0)
        return predictions, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, source, target, teacher_force_ratio=0.5):
        """
        : we gonna send in the german sentence as well as the correct sentence
        : teacher_force_ratio => 0.5(half) some times you have to leave it to the rnn for guessing
            the next word, and sometimes it should be eaxctly the same word in the target so: 0.5
        : we gonna send the inputs straight into the encoder and decoder which gonna return hidden
            and cell which gonna be inputs for the decoder
        """
        batch_size = source.shape[1]
        target_len = target.shape[0]
        target_vocab_size = len(english.vocab)

        hidden, cell = self.encoder(source)

        outputs = torch.zeros(target_len, batch_size, target_vocab_size).to(device)

        # grab the start token
        x = target[0]

        for t in range(1, target_len):
            output, hidden, cell = self.decoder(x, hidden, cell)
            # each hidden and cell gonna be next input for the decoder
            # output_size = batch_size, target_vocab_size (N, eng_vocab_size)
            outputs[t] = output  # we are adding each output along their dimension
            best_guess = output.argmax(1)

            x = target[t] if random.random() < teacher_force_ratio else best_guess

        return outputs


# set the hyper parameters
num_epochs = 100
learning_rate = 0.001
batch_size = 64

# MOdel hyper params
load_model = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_size_encoder = len(german.vocab)
input_size_decoder = len(english.vocab)
output_size = len(english.vocab)
encoder_embedding_size = 300
decoder_embedding_size = 300
hidden_size = 1024
num_layers = 2
enc_dropout = 0.5
dec_dropout = 0.5

# Tensorboard to get nice loss plot
writer = SummaryWriter(f"runs/loss_plot")
step = 0

train_iterator, valid_iterator = BucketIterator.splits(
    (train_data, validation_data),
    batch_size=batch_size,
    sort_within_batch=True,
    sort_key=lambda x: len(x.src),
    device=device,
)

# encoder_network
encoder_net = Encoder(
    input_size_encoder, encoder_embedding_size, hidden_size, num_layers, enc_dropout
).to(device)

# decoder_network
decoder_net = Decoder(
    input_size_decoder,
    decoder_embedding_size,
    hidden_size,
    output_size,
    num_layers,
    dec_dropout,
).to(device)

# Model and Optimizer
model = Seq2Seq(encoder_net, decoder_net).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

pad_idx = english.vocab.stoi["<pad>"]
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

if load_model:
    load_checkpoint(torch.load("my_checkpoint.pth.tar"), model, optimizer)

# german sentence
sentence = "Sindhu ist so eine Schlampe, wie du weißt"

for epoch in range(num_epochs):
    print(f"Epoch [{epoch}/{num_epochs}]")

    checkpoint = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
    save_checkpoint(checkpoint)

    model.eval()
    translated_sentence = translate_sentence(
        model, sentence, german, english, device, max_length=50
    )

    print(f"translated_sentence: {translated_sentence}")

    model.train()

    for batch_idx, batch in enumerate(train_iterator):
        src = batch.src.to(device)
        target = batch.trg.to(device)

        output = model(src, target)

        """
            : output.shape => (trg_len, batch_size_N, output_dim)
            : we have to reshape it into two dim for cross entropy loss
            : [1:] => we dont have to send the start token into the model
            : we dont have the output dim for the target so only (-1) is enough
            : we also know that lstm are famous for their exploding and vanishing gradients => LOL, ROFL
                so do the gradient clipping to make sure that gradients are in the healthy condition
        """
        output = output[1:].reshape(-1, output.shape[2])
        target = target[1:].reshape(-1)

        optimizer.zero_grad()
        loss = criterion(output, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()
        writer.add_scalar("Training loss", loss, global_step=step)
        step += 1

score = bleu(test_data[1:100], model, german, english, device)
print(f"Bleu score {score*100:.2f}")

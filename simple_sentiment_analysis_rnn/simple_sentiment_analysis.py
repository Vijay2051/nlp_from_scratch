import os
import sys

import spacy
import torch
import torch.nn as nn
import pandas as pd
import torch.optim as optim
from sklearn import model_selection
from torchtext.data import BucketIterator, Field, LabelField, TabularDataset

import config
import models

SEED = config.SEED

# set the manual seed with torch
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

try:
    spacy_eng = spacy.load("en")
    spacy_ger = spacy.load("de")

except Exception as e:
    os.system("python3 -m spacy download en")
    os.system("python3 -m spacy download de")
    spacy_eng = spacy.load("en")
    spacy_ger = spacy.load("de")

def tokenize(text):
    return [tok.text for tok in spacy_eng.tokenizer(text)]

review = Field(sequential=True, use_vocab=True, lower=True, tokenize=tokenize)
sentiment = LabelField(dtype = torch.float)


"""
    *** use this code to generate train and test csv for imdb
"""
# df = pd.read_csv("../input/imdb.csv")
# df.columns = ["review", "sentiment"]
# df["sentiment"] = df["sentiment"].apply(lambda x: 0 if x == "positive" else 1)
# train, test = model_selection.train_test_split(df, test_size=0.2)
# train.to_csv("../input/train_imdb.csv", index=False)
# test.to_csv("../input/test_imdb.csv", index=False)

fields = {"review": ("rev", review), "sentiment": ("sen", sentiment)}

# dataset
train_data, test_data = TabularDataset.splits(
    path="../input/",
    train="train_imdb.csv",
    test="test_imdb.csv",
    format="csv",
    fields=fields,
    skip_header = False
)

# build the vocab
review.build_vocab(train_data, max_size = 10000, min_freq=2)
sentiment.build_vocab(train_data)

# iterator
train_iterator, test_iterator = BucketIterator.splits(
    (train_data, test_data), batch_size=64, device=config.DEVICE, sort_within_batch=False, sort_key=lambda x: len(x.rev)
)

for batch in test_iterator:
    print(batch)

INPUT_DIM = len(review.vocab)
EMBEDDING_DIM = 100
HIDDEN_DIM = 256
OUTPUT_DIM = 1

model = models.RNN(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM)
optimizer = optim.SGD(model.parameters(), lr=1e-3)
criterion = nn.BCEWithLogitsLoss()


model = model.to(config.DEVICE)
criterion = criterion.to(config.DEVICE)

def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """

    #round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float() #convert into float for division 
    acc = correct.sum() / len(correct)
    return acc

def train(model, iterator, optimizer, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.train()
    
    for batch in iterator:
        
        optimizer.zero_grad()
                
        predictions = model(batch.rev).squeeze(1)
        
        loss = criterion(predictions, batch.sen)
        
        acc = binary_accuracy(predictions, batch.sen)
        
        loss.backward()
        
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate(model, iterator, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.eval()
    
    with torch.no_grad():
    
        for batch in iterator:

            predictions = model(batch.rev).squeeze(1)
            
            loss = criterion(predictions, batch.sen)
            
            acc = binary_accuracy(predictions, batch.sen)

            epoch_loss += loss.item()
            epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

N_EPOCHS = 200
for epoch in range(N_EPOCHS):
    
    train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
    valid_loss, valid_acc = evaluate(model, test_iterator, criterion)
    
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')
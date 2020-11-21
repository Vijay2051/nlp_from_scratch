from sys import maxsize
import spacy
from torchtext.datasets import Multi30k
import os
from torchtext.data import Field, BucketIterator

try:
    spacy_eng = spacy.load("en")
    spacy_ger = spacy.load("de")

except Exception as e:
    os.system("python3 -m spacy download en")
    os.system("python3 -m spacy download de")
    spacy_eng = spacy.load("en")
    spacy_ger = spacy.load("de")


def tokenize_eng(text):
    return [tok.text for tok in spacy_eng.tokenizer(text)]


def tokenize_ger(text):
    return [tok.text for tok in spacy_ger.tokenizer(text)]


english = Field(sequential=True, use_vocab=True, lower=True, tokenize=tokenize_eng)
german = Field(sequential=True, use_vocab=True, lower=True, tokenize=tokenize_ger)

train_data, validation_data, test_data = Multi30k.splits(
    exts=(".de", ".en"), fields=(german, english)
)

# build the vocab for german and english

english.build_vocab(train_data, max_size=10000, min_freq=2)
german.build_vocab(train_data, max_size=10000, min_freq=2)

train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data, validation_data, test_data), batch_size=64, device="cpu"
)

sent = []
for batch in train_iterator:
    print(batch.trg)
    # sent.append(int(batch.trg[0, :][0]))

# for i in sent:
#     print(english.vocab.itos[i])
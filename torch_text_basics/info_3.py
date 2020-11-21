from operator import truediv
import spacy
import pandas as pd
from torchtext.data import Field, BucketIterator, TabularDataset
import os
from sklearn.model_selection import train_test_split

try:
    spacy_eng = spacy.load("en")
    spacy_ger = spacy.load("de")

except Exception as e:
    os.system("python3 -m spacy download en")
    os.system("python3 -m spacy download de")
    spacy_eng = spacy.load("en")
    spacy_ger = spacy.load("de")

"""

*** use this code to save the files in csv and json format ***

eng_text = open("../input/wmt_en.txt", encoding="utf-8").read().split("\n")
ger_text = open("../input/wmt_de.txt", encoding="utf-8").read().split("\n")

raw_data = {"german": [line for line in ger_text[:10000]],
            "english": [line for line in eng_text[:10000]]}

df = pd.DataFrame(raw_data, columns=["german", "english"])

train, test = train_test_split(df, test_size=0.2)

# save it in json
train.to_json("../input/torch_text_data_3/train_wmt_10000.json", orient="records",  lines=True)
test.to_json("../input/torch_text_data_3/test_wmt_10000.json", orient="records", lines=True)

# save it in csv
train.to_csv("../input/torch_text_data_3/train_wmt_10000.csv", index=False)
test.to_csv("../input/torch_text_data_3/test_wmt_10000.csv", index=False)

"""

def tokenize_eng(text):
    return [tok.text for tok in spacy_eng.tokenizer(text)]

def tokenize_ger(text):
    return [tok.text for tok in spacy_ger.tokenizer(text)]

english = Field(sequential=True, use_vocab=True, tokenize=tokenize_eng, lower=True)
german = Field(sequential=True, use_vocab=True, tokenize=tokenize_ger, lower=True)

# create fields in json format for the tabular data
fields = {"english": ("eng", english), "german": ("ger", german)}

# split into train and test from the dataset
train_data, test_data = TabularDataset.splits(path="../input/torch_text_data_3", train="train_wmt_10000.json", test="test_wmt_10000.json", format="json", fields=fields)

# build the vocab 
english.build_vocab(train_data, max_size=10000, min_freq=2)
german.build_vocab(train_data, max_size=10000, min_freq=2)

# buidl the iterator
train_iterator, test_iterator = BucketIterator.splits((train_data, test_data), batch_size=32, device="cuda")

for batch in train_iterator:
    print(batch)
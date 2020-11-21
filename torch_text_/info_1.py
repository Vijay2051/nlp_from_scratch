# steps

# 1. specify how the pre processing should be done
# 2. Use the dataset to load the data --> Tabular Data (json, csv, tsv)
# 3. Construct the iterator to do batching and padding -BucketIterator

from torchtext.data import Field, TabularDataset, BucketIterator
import spacy
import os

try:
    spacy_en = spacy.load("en")
except Exception as e:
    os.system("python3 -m spacy download en")
    spacy_en = spacy.load("en")

# tokenize the quote from the train.csv
# tokenize = lambda x: x.split() #this is a bad tokenizer ;-)
def tokenize(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]


quote = Field(sequential=True, use_vocab=True, tokenize=tokenize, lower=True)
score = Field(sequential=False, use_vocab=False)

# mentioning a tuple inside the dict: this will be usefull while mentioning the batch
# it can be mentioned as batch.q for quote and batch.s for score
fields = {"quote": ("q", quote), "score": ("s", score)}


# dataset
train_data, test_data = TabularDataset.splits(
    path="../input/torch_text_data",
    train="train.csv",
    test="test.csv",
    format="csv",
    fields=fields,
)


print(train_data[0].__dict__.keys())
print(train_data[0].__dict__.values())

# build the vocabulary from the quote
quote.build_vocab(train_data,
                  max_size=10000, 
                  min_freq=1, 
                #   vectors="glove.6B.100d" #this is of 1gb
                  )

# iterator
train_iterator, test_iterator = BucketIterator.splits(
    (train_data, test_data), batch_size=2, device="cpu"
)

for batch in train_iterator:
    print(batch.q)

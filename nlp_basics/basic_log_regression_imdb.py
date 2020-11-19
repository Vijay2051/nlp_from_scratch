import pandas as pd

from nltk.tokenize import word_tokenize
from sklearn import linear_model
from sklearn import metrics
from sklearn import model_selection
from sklearn.feature_extraction.text import CountVectorizer

if __name__ == "__main__":
    # read the csv of imdb dataset
    df = pd.read_csv("../input/imdb.csv")
    print(df.head())
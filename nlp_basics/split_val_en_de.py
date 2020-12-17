import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("../input/translation/en_de_val.csv")

val_train, val_test = train_test_split(df, test_size=0.2, random_state=42)
val_train.to_csv("../input/translation/en_de_val_train.csv", index=False)
val_test.to_csv("../input/translation/en_de_val_test.csv", index=False)
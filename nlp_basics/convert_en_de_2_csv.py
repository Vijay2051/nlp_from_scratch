import pandas as pd

de = open("../input/translation/wmt_de.txt").readlines()
en = open("../input/translation/wmt_en.txt").readlines()

en_de_df_train = pd.DataFrame(data=zip(en[:4000000], de[:4000000]), columns=["english", "german"])
en_de_df_val = pd.DataFrame(data=zip(en[4000000:], de[4000000:]), columns=["english", "german"])

en_de_df_train.to_csv("../input/translation/en_de_train.csv", index=False)
en_de_df_val.to_csv("../input/translation/en_de_val.csv", index=False)

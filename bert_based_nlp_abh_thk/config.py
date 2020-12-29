from transformers import BertTokenizer
import torch

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
max_len = BertTokenizer.max_model_input_sizes["bert-base-uncased"]
train_batch_size = 8
valid_batch_size = 4
epochs = 10
training_file = "../input/imdb.csv"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

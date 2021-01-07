from transformers import BertTokenizer
import torch

TOKENIZER = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
MAX_LEN = 64
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 4
EPOCHS = 10
TRAINING_FILE = "../input/imdb.csv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "../input/abh_sent_model_torch/model.bin"
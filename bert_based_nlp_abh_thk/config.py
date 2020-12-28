from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
max_len = BertTokenizer.max_model_input_sizes["bert-base-uncased"]
train_batch_size = 8
valid_batch_size = 4
epochs = 10
accumulation = 2

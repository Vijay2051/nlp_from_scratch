import transformers
import torch.nn as nn

class BertBaseUncased(nn.Module):
    def __init__(self) -> None:
        super(BertBaseUncased, self).__init__()
        self.bert = transformers.BertModel.from_pretrained("bert-base-uncased")
        self.bert_drop = nn.Dropout(0.3)
        self.out = nn.Linear(768, 1)

    def forward(self, ids, mask, token_type_ids):
        out1, out2 = self.bert(ids, attention_mask=mask, )
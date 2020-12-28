import transformers
import torch.nn as nn


class BertBaseUncased(nn.Module):
    def __init__(self) -> None:
        super(BertBaseUncased, self).__init__()
        self.bert = transformers.BertModel.from_pretrained("bert-base-uncased")
        self.bert_drop = nn.Dropout(0.3)
        self.out = nn.Linear(768, 1)

    def forward(self, ids, mask, token_type_ids):
        out1, out2 = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids)
        """
            :: out1 => sequence of hidden states for each token for all batches
                        if you have 512 tokens, then you'll have 512 vectors of size 768 for each batch
            :: out2 => contains the last layer hidden for the first cls token of the sequence
        """
        bert_drop = self.bert_drop(out2)
        bert_out = self.out(bert_drop)
        return bert_out

import torch.nn as nn
import torch

class RNN(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim):
        super(RNN, self).__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, text):
        # text = [sent_len, batch_size]
        embedded = self.embedding(text)
        # embedded = [sent_len, batch_size, embedding_dim]
        # print(embedded.shape)

        output, hidden = self.rnn(embedded)
        # output = [sent_len, batch_size, hidden_dim]
        # hiden = [1, batch_size, hidden_dim]
        # print(output.shape, hidden.shape)

        assert torch.equal(output[-1, :, :], hidden.squeeze(0))
        return self.fc(hidden.squeeze(0))
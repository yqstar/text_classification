import torch
from torch import nn


# Define Network
class Rnn(nn.Module):
    def __init__(self):
        super(Rnn, self).__init__()
        self.embeddings = nn.Embedding(num_embeddings=4411, embedding_dim=64)
        self.rnn = nn.RNN(64, 128, batch_first=True)
        self.linear = nn.Linear(128, 2)
        # self.model = nn.Sequential()

    def forward(self, inputs):
        print("inputs shape is {}".format(inputs.shape))
        emb = self.embeddings(inputs)
        # print("inputs shape is {}".format(inputs.shape))
        hidden_state, out_state = self.rnn(emb)
        # hn：（num_layers*directions，batch_size，hidden_size）
        # output：(seq_len，batch_size，hidden_size * directions)
        # print("hidden_state shape is {}".format(hidden_state.shape))
        # print("out_state shape is {}".format(out_state.shape))
        out_state = torch.squeeze(out_state, dim=0)
        target = self.linear(out_state)
        return target




import torch
from torch import nn


class BookingNN(nn.Module):
    def __init__(self,
                 vocab_size,
                 emb_dim=500,
                 rnn_dim=500,
                 num_layers=2,
                 dropout=0.3,
                 rnn_dropout=0.3,
                 tie_weight=False):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        """
        cat_dims = [int(train_test[col].nunique()) for col in categorical_cols]
        emb_dims = [(x, min(50, (x + 1) // 2)) for x in cat_dims]
        """
        self.emb_layers = nn.ModuleList([nn.Embedding(x, y) for x, y in emb_dims])
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.dense_n = nn.Linear(1, 5)
        self.rnn = nn.LSTM(input_size=emb_dim + sum([d[1] for d in emb_dims]) + 5,
                           hidden_size=rnn_dim,
                           num_layers=num_layers,
                           batch_first=True,
                           dropout=rnn_dropout,
                           bidirectional=True)
        self.dense = nn.Linear(rnn_dim * 2, vocab_size)

        if tie_weight:
            self.dense.weight = self.embedding.weight

        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.rnn_dim = rnn_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.rnn_dropout = rnn_dropout
        self.tie_weight = tie_weight

    def forward(self, x_seq, x_cat, x_num, h0=None):
        out_c = [emb_layer(x_cat[:, i]) for i, emb_layer in enumerate(self.emb_layers)]
        out_c = torch.cat(out_c + [self.dense_n(x_num)], axis=1)
        out_c = out_c.repeat(x_seq.shape[1], 1, 1)
        out_c = out_c.view(out_c.shape[1], out_c.shape[0], out_c.shape[2])
        out_s, hidden = self.rnn(torch.cat([self.drop(self.embedding(x_seq)), out_c], axis=2), h0)
        out_s = self.dense(self.drop(out_s.reshape(out_s.size(0) * out_s.size(1), out_s.size(2))))
        return out_s, hidden

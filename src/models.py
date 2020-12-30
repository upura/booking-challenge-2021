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
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.rnn = nn.RNN(input_size=emb_dim,
                          hidden_size=rnn_dim,
                          num_layers=num_layers,
                          dropout=rnn_dropout)
        self.dense = nn.Linear(rnn_dim, vocab_size)

        if tie_weight:
            self.dense.weight = self.embedding.weight

        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.rnn_dim = rnn_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.rnn_dropout = rnn_dropout
        self.tie_weight = tie_weight

    def forward(self, x, h0=None):
        out, hidden = self.rnn(self.drop(self.embedding(x)), h0)
        out = self.dense(self.drop(out.view(out.size(0) * out.size(1), out.size(2))))
        return out, hidden

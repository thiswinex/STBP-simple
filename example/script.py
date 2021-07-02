import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import layers


def lstm2snn(rnn):
    nip = rnn.input_size
    nhid = rnn.hidden_size
    nlayers = rnn.num_layers
    bias = rnn.bias
    dropout = rnn.dropout
    batch_first = rnn.batch_first

    snn = nn.Sequential()
    if batch_first: # (batch, seq, nip) -> (batch, nip, seq)
        snn.add_module("transpose", layers.Transpose(0, 2, 1))
    else:           # (seq, batch, nip) -> (batch, nip, seq)
        snn.add_module("transpose", layers.Transpose(1, 2, 0))
    snn.add_module("td linear 0", layers.tdLayer(nn.Linear(nip, nhid, bias=bias)))
    snn.add_module("spike 0", layers.LIFSpike())
    if dropout:
        snn.add_module("dropout 0", nn.Dropout(dropout))
    for index in range(1, nlayers):
        snn.add_module("td linear %d" % index, layers.tdLayer(nn.Linear(nhid, nhid, bias=bias)))
        snn.add_module("spike %d" % index, layers.LIFSpike())
        if dropout and index != nlayers - 1:
            snn.add_module("dropout %d" % index, nn.Dropout(dropout))
    snn.add_module("rate coding", layers.RateCoding())

    return snn



class LSTMModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, ntoken, ninp, nhid, nlayers, dropout=0.5):
        super(LSTMModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.rnn = nn.LSTM(ninp, nhid, nlayers, dropout=dropout)
        self.decoder = nn.Linear(nhid, ntoken)

        self.init_weights()

        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden):
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output)
        return decoded, hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                weight.new_zeros(self.nlayers, bsz, self.nhid))


if __name__ == '__main__':
    lstm = LSTMModel(10, 512, 256, 5)
    print(lstm.rnn)
    snn = lstm2snn(lstm.rnn)
    print(snn)
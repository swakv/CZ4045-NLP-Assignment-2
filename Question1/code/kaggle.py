# Q1_data
import os
from io import open
import torch

# Q1_model
import math
import torch.nn as nn
import torch.nn.functional as F

#Q1_main
# import argparse
import time
import torch.onnx
import numpy as np

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self):
        self.dictionary = Dictionary()
        self.train = self.tokenize('../wikitext-2/wiki.train.tokens')
        self.valid = self.tokenize('../wikitext-2/wiki.valid.tokens')
        self.test = self.tokenize( '../wikitext-2/wiki.test.tokens')

    def tokenize(self, path):
        """Tokenizes a text file."""
        
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r', encoding="utf8") as f:
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r', encoding="utf8") as f:
            idss = []
            for line in f:
                words = line.split() + ['<eos>']
                ids = []
                for word in words:
                    ids.append(self.dictionary.word2idx[word])
                idss.append(torch.tensor(ids).type(torch.int64))
            # print("idss",len(idss))
            ids = torch.cat(idss)
            # print("ids",len(ids))

        return ids




class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, tie_weights=False):
        super(RNNModel, self).__init__()
        self.ntoken = ntoken # VOCAB SIZE
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)
            # print(rnn_type)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError( """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout) # input size, hidden dimension, no of hidden layers size
        self.decoder = nn.Linear(nhid, ntoken) # hidden dimension, vocab size

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.weight)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, input, hidden):
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output)
        decoded = decoded.view(-1, self.ntoken)
        return F.log_softmax(decoded, dim=1), hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                    weight.new_zeros(self.nlayers, bsz, self.nhid))
        else:
            return weight.new_zeros(self.nlayers, bsz, self.nhid)

# Temporarily leave PositionalEncoding module here. Will be moved somewhere else.
class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerModel(nn.Module):
    """Container module with an encoder, a recurrent or transformer module, and a decoder."""

    def _init_(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self)._init_()
        try:
            from torch.nn import TransformerEncoder, TransformerEncoderLayer
        except:
            raise ImportError('TransformerEncoder module does not exist in PyTorch 1.1 or lower.')
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, ntoken)

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.weight)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, src, has_mask=True):
        if has_mask:
            device = src.device
            if self.src_mask is None or self.src_mask.size(0) != len(src):
                mask = self._generate_square_subsequent_mask(len(src)).to(device)
                self.src_mask = mask
        else:
            self.src_mask = None

        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        output = self.decoder(output)
        return F.log_softmax(output, dim=-1)



class FNNModel(nn.Module):

    def __init__(self, ntoken, ninp, nhid):
        super(FNNModel, self).__init__()

        self.embeddings = nn.Embedding(ntoken, ninp)
        
        self.linear = nn.Linear(ninp, nhid)
        self.linear2 = nn.Linear(nhid, ntoken) 

        self.ntoken = ntoken 
        self.init_weights()
        self.nhid = nhid

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.embeddings.weight, -initrange, initrange)
        nn.init.zeros_(self.linear2.weight)
        nn.init.uniform_(self.linear2.weight, -initrange, initrange)

    def forward(self, input):
        embeds = self.embeddings(input)
        out = F.tanh(self.linear(embeds))
        log_probs = F.log_softmax(self.linear2(out), dim=1)
        return log_probs

class FNNModelSharing(nn.Module):

    def __init__(self, ntoken, ninp, nhid):
        super(FNNModelSharing, self).__init__()

        self.embeddings = nn.Embedding(ntoken, ninp)
        
        self.linear = nn.Linear(ninp, nhid)
        self.linear2 = nn.Linear(nhid, ntoken) 
        
        self.linear2.weight.data = self.embeddings.weight.data

        self.ntoken = ntoken 
        self.init_weights()
        self.nhid = nhid

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.embeddings.weight, -initrange, initrange)
        nn.init.zeros_(self.linear2.weight)
        nn.init.uniform_(self.linear2.weight, -initrange, initrange)

    def forward(self, input):
        embeds = self.embeddings(input)
        out = F.tanh(self.linear(embeds))
        
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs


data = 'wikitext-2'
model_name = 'FNNS'
emsize = 200
nhid = 200
nlayers = 2
lr = 0.01
clip = 0.25
epochs = 6
batch_size = 20
bptt = 8
dropout = 0.2
seed = 1111
tied = True
save = 'model_kag.pt'
onnx_export = ''
nhead = 2
dry_run = False
cuda_var = True
log_interval = 100

# Set the random seed manually for reproducibility.
torch.manual_seed(seed)
if torch.cuda.is_available():
    if not cuda_var:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

device = torch.device("cuda" if cuda_var else "cpu")

###############################################################################
# Load data
###############################################################################

corpus = Corpus()
# print("final train",len(corpus.train))

# Starting from sequential data, batchify arranges the dataset into columns.
# For instance, with the alphabet as the sequence and batch size 4, we'd get
# ┌ a g m s ┐
# │ b h n t │
# │ c i o u │
# │ d j p v │
# │ e k q w │
# └ f l r x ┘.
# These columns are treated as independent by the model, which means that the
# dependence of e. g. 'g' on 'f' can not be learned, but allows more efficient
# batch processing.

def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)

def ngram(data, no):
    ngrams_tuple = torch.split(data,8)
    return ngrams_tuple


eval_batch_size = 10
if model_name  != 'FNN' and model_name != 'FNNS':
    train_data = batchify(corpus.train, batch_size)
    val_data = batchify(corpus.valid, eval_batch_size)
    test_data = batchify(corpus.test, eval_batch_size)
else: 
    train_data = ngram(corpus.train, 8)
    val_data = ngram(corpus.valid, 8)
    test_data = ngram(corpus.test, 8)
###############################################################################
# Build the model
###############################################################################

ntokens = len(corpus.dictionary)
if model_name  == 'Transformer':
    model = TransformerModel(ntokens, emsize, nhead, nhid, nlayers, dropout).to(device)
elif  model_name  == 'FNN':
    model = FNNModel(ntokens, emsize, nhid)
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
elif model_name  == 'FNNS':
    model = FNNModelSharing(ntokens, emsize, nhid)
    # optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    optimizer = torch.optim.SGD(model.parameters(), lr, momentum=0.9)
else:
    model = RNNModel(model_name, ntokens, emsize, nhid, nlayers, dropout, tied).to(device)
  

criterion = nn.NLLLoss()

###############################################################################
# Training code
###############################################################################

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""

    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


# get_batch subdivides the source data into chunks of length bptt.
# If source is equal to the example output of the batchify function, with
# a bptt-limit of 2, we'd get the following two Variables for i = 0:
# ┌ a g m s ┐ ┌ b h n t ┐
# └ b h n t ┘ └ c i o u ┘
# Note that despite the name of the function, the subdivison of data is not
# done along the batch dimension (i.e. dimension 1), since that was handled
# by the batchify function. The chunks are along dimension 0, corresponding
# to the seq_len dimension in the LSTM.

def get_batch(source, i):
    # print("i " ,i)
    seq_len = min(bptt, len(source) - 1 - i)
    # print("SEQLEN",seq_len)
    data = source[i:i+seq_len]
    # print("DATA SHAPE ",data.shape)
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target



def evaluate(data_source):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.
    ntokens = len(corpus.dictionary)
    if model_name  != 'Transformer' and model_name  != 'FNN' and model_name  != 'FNNS':
        hidden = model.init_hidden(eval_batch_size)
    
    with torch.no_grad():
        if model_name  != 'FNN' and model_name  != 'FNNS':
            for i in range(0, data_source.size(0) - 1, bptt):
                data, targets = get_batch(data_source, i)
                if model_name  == 'Transformer' or model_name  == 'FNN' or model_name  == 'FNNS':
                    output = model(data)
                    output = output.view(-1, ntokens)
                else:
                    output, hidden = model(data, hidden)
                    hidden = repackage_hidden(hidden)
                total_loss += len(data) * criterion(output, targets).item()
        else: #for FNN
            for i in range(len(data_source)):
                if i == len(data_source)-1:
                    target_n = 0
                else:
                    target_n = np.asarray(data_source[i+1][0])
                    
                data = np.asarray(data_source[i])
                data = torch.LongTensor(list(data))
                targets = np.asarray(data_source[i][1:])
                targets = np.append(targets, target_n)
                targets = torch.LongTensor(list(targets))

                output = model(data)
                output = output.view(-1, ntokens)
                total_loss += len(data) * criterion(output, targets).item()

    return total_loss / (len(data_source) - 1)

def train():
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0.
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    # print("--------", model)
    if model_name  != 'Transformer' and model_name  != 'FNN' and model_name  != 'FNNS':
        hidden = model.init_hidden(batch_size)
    if model_name  != 'FNN' and model_name  != 'FNNS':
        for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
            # print("batch issssss",batch)
            data, targets = get_batch(train_data, i)
            # Starting each batch, we detach the hidden state from how it was previously produced.
            # If we didn't, the model would try backpropagating all the way to start of the dataset.
            model.zero_grad()
            if model_name  == 'Transformer' or model_name  == 'FNN' or model_name == 'FNNS':
                optimizer.zero_grad()
                output = model(data)
                output = output.view(-1, ntokens)
            else:
                hidden = repackage_hidden(hidden)
                output, hidden = model(data, hidden)
            loss = criterion(output, targets)
            loss.backward()
            # optimizer.step()


            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            # if model != 'Transformer' and model != 'FNN':
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            for p in model.parameters():
                p.data.add_(p.grad, alpha=-lr)

            total_loss += loss.item()

            if batch % log_interval == 0 and batch > 0:
                cur_loss = total_loss / log_interval
                elapsed = time.time() - start_time
                print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.3f} | ms/batch {:5.2f} | '
                        'loss {:5.2f} | ppl {:8.2f}'.format(
                    epoch, batch, len(train_data) // bptt, lr,
                    elapsed * 1000 / log_interval, 
                    cur_loss, 
                    np.exp(cur_loss)))
                total_loss = 0
                start_time = time.time()
            if dry_run:
                break
    else: #for FNN
        print("here")
        for i in range(len(train_data)):
            if i == len(train_data)-1:
                target_n = 0
            else:
                target_n = np.asarray(train_data[i+1][0])
                
            data = np.asarray(train_data[i])
            
            # print("thisthisthisthis", list(data))
            data = torch.LongTensor(list(data))

            targets = np.asarray(train_data[i][1:])
            targets = np.append(targets, target_n)
            targets = torch.LongTensor(list(targets))

            model.zero_grad()
            optimizer.zero_grad()
            output = model(data)
            output = output.view(-1, ntokens)

            loss = criterion(output, targets)
            loss.backward()
            optimizer.step()

            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            for p in model.parameters():
                p.data.add_(p.grad, alpha=-lr)

            total_loss += loss.item()

            print("i is ", i)
            if i % log_interval == 0 and i > 0:
                print("hereeeeee")
                cur_loss = total_loss / log_interval
                elapsed = time.time() - start_time
                print('| epoch {:3d} | seq number {:3d} | lr {:02.3f} | '
                        'loss {:5.2f} | ppl {:8.2f}'.format(
                    epoch, i, lr,
                    cur_loss, 
                    np.exp(cur_loss)))
                total_loss = 0
                start_time = time.time()
            print("not in if")
            if dry_run:
                print("dry")
                break



def export_onnx(path, batch_size, seq_len):
    print('The model is also exported in ONNX format at {}'.
          format(os.path.realpath(onnx_export)))
    model.eval()
    dummy_input = torch.LongTensor(seq_len * batch_size).zero_().view(-1, batch_size).to(device)
    
    hidden = model.init_hidden(batch_size)
    torch.onnx.export(model, (dummy_input, hidden), path)


# Loop over epochs.
lr = lr
best_val_loss = None

# At any point you can hit Ctrl + C to break out of training early.
try:
    for epoch in range(1, epochs+1):
        epoch_start_time = time.time()
        train()
        val_loss = evaluate(val_data)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                           val_loss, math.exp(val_loss)))
        print('-' * 89)
        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            with open(save, 'wb') as f:
                torch.save(model, f)
            best_val_loss = val_loss
        else:
            # Anneal the learning rate if no improvement has been seen in the validation dataset.
            lr /= 4.0
except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

# Load the best saved model.
with open(save, 'rb') as f:
    model = torch.load(f)
    # after load the rnn params are not a continuous chunk of memory
    # this makes them a continuous chunk, and will speed up forward pass
    # Currently, only rnn model supports flatten_parameters function.
    if model_name  in ['RNN_TANH', 'RNN_RELU', 'LSTM', 'GRU']:
        model.rnn.flatten_parameters()

# Run on test data.
test_loss = evaluate(test_data)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
print('=' * 89)

if len(onnx_export) > 0:
    # Export the model in ONNX format.
    export_onnx(onnx_export, batch_size=1, seq_len=bptt)
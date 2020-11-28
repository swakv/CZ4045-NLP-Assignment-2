###############################################################################
# Language Modeling on Wikitext-2
#
# This file generates new sentences sampled from the language model
#
###############################################################################

import argparse

import torch

import Q1_data

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 Language Model')

# Model parameters.
parser.add_argument('--data', type=str, default='./data/wikitext-2',
                    help='location of the data corpus')
parser.add_argument('--checkpoint', type=str, default='../models/modelFNN.pt',
                    help='model checkpoint to use')
parser.add_argument('--outf', type=str, default='../generated/generated.txt',
                    help='output file for generated text')
parser.add_argument('--words', type=int, default='1000',
                    help='number of words to generate')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--temperature', type=float, default=1.0,
                    help='temperature - higher will increase diversity')
parser.add_argument('--log-interval', type=int, default=100,
                    help='reporting interval')
args = parser.parse_args()



class FNNModel(nn.Module):

    def __init__(self, ntoken, ninp, nhid):
        super(FNNModel, self).__init__()

        self.embeddings = nn.Embedding(ntoken, ninp)
        self.linear = nn.Linear(7*ninp, nhid)
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
        embeds = embeds.view(1,-1)
        embeds = embeds.resize_((200,1400))
        out = self.linear(embeds)
        out = F.tanh(self.linear(embeds))
        log_probs = F.log_softmax(self.linear2(out))
        return log_probs

class FNNModelSharing(nn.Module):

    def __init__(self, ntoken, ninp, nhid):
        super(FNNModelSharing, self).__init__()

        self.embeddings = nn.Embedding(ntoken, ninp)
       
        self.linear = nn.Linear(7*ninp, nhid)
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
        embeds = embeds.view(1,-1)
        embeds = embeds.resize_((200,1400))
        out = F.tanh(self.linear(embeds))
       
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs


# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

device = torch.device("cuda" if args.cuda else "cpu")
# print(device)

if args.temperature < 1e-3:
    parser.error("--temperature has to be greater or equal 1e-3")

with open(args.checkpoint, 'rb') as f:
    model = torch.load(f, map_location='cpu').to(device)

# print(model)
model.eval()


corpus = Q1_data.Corpus(args.data)
ntokens = len(corpus.dictionary)

# print(model)
# is_transformer_model = hasattr(model, 'model_type') and (model.model_type == 'Transformer' or model.model_type == 'FNN')
is_transformer_model = True
if not is_transformer_model:
    hidden = model.init_hidden(1)
input = torch.randint(ntokens, (1, 7), dtype=torch.long).to(device)

with open(args.outf, 'w') as outf:
    with torch.no_grad():  # no tracking history
        for i in range(args.words):
            if is_transformer_model:
                output = model(input)
                word_weights = output[-1].squeeze().div(args.temperature).exp().cpu()
                word_idx = torch.multinomial(word_weights, 1)[0]
                word_tensor = torch.Tensor([[word_idx]]).long().to(device)
                input = torch.cat((input, word_tensor), dim=1)
                # input = torch.cat([input, word_tensor], 0)
            else:
                output, hidden = model(input, hidden)
                word_weights = output.squeeze().div(args.temperature).exp().cpu()
                word_idx = torch.multinomial(word_weights, 1)[0]
                input.fill_(word_idx)

            word = corpus.dictionary.idx2word[word_idx]

            outf.write(word + ('\n' if i % 20 == 19 else ' '))

            if i % args.log_interval == 0:
                print('| Generated {}/{} words'.format(i, args.words))
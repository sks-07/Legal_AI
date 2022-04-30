#import numpy as np


import torch 
import torch.nn as nn
import json
import os
import nltk
import torch.nn.functional as F
#import argparse

class WordEncoder(nn.Module):
    def __init__(self,args) -> None:
        super(WordEncoder,self).__init__()
        self.input_size = args.input_dim
        self.hidden_dim = args.hidden_dim
        self.gru_layer=nn.GRU(input_size=self.input_size,
                              hidden_size=self.hidden_dim,
                              bidirectional=True)

    def forward(self,inp):
        output,_ = self.gru_layer(inp)
        return output

class WordAttn(nn.Module):
    """Word Level Attention for Sentences"""

    def __init__(self, input_dim, device):
        super(WordAttn, self).__init__()
        self.input_dim = input_dim
        self.device = device
        self.linear = nn.Linear(in_features=self.input_dim,
                                out_features=self.input_dim,
                                bias=True)
        self.tanh = nn.Tanh()
        self.context = nn.Parameter(torch.Tensor(self.input_dim, 1))
        self._create_weights(mean=0.0, std=0.05)

    def _create_weights(self, mean=0.0, std=0.05):
        self.context.data.normal_(mean, std)

    def forward(self, ip):
        output = self.linear(ip)
        # [batch_size * num_sentences, sentence_length, 2 * hidden_dim]
        output = self.tanh(output)
        output = torch.matmul(output, self.context)
        # [batch_size * num_sentences, sentence_length, 1]

        output = torch.squeeze(output, dim=-1)
        # [batch_size * num_sentences, sentence_length]

        attn_weights = F.softmax(output, dim=1)
        # [batch_size * num_sentences , sentence_length]

        sent_embeds = self.element_wise_multiply(ip, attn_weights)

        return sent_embeds

    def element_wise_multiply(self, ip, attn_weights):
        sent_embeds = torch.tensor([])
        sent_embeds = sent_embeds.to(self.device)
        for sentence, weights in zip(ip, attn_weights):
            weights = weights.view(1, -1)
            sentence = torch.squeeze(sentence, dim=0)
            # [1, sentence_length, 2 * hidden_dim]
            # -> [sentence_length, 2 * hidden_dim]

            sent_embed = torch.matmul(weights, sentence)
            # [1, 2 * hidden_dim]

            sent_embeds = torch.cat((sent_embeds, sent_embed), dim=0)
            # [batch_size * num_sentences, 2 * hidden_dim]

        return sent_embeds

class argParser():
    def __init__(self,path):
        self.args=self.__arg_parser(path)
        self.input_dim=self.args['INPUT_SIZE']
        self.hidden_dim=self.args['HIDDEN_SIZE']

    def __arg_parser(self,path):
        with open(path,'r') as f:
            args=json.load(f)
        return args

def main():
    my_arg=argParser('args.json')
    wd = WordEncoder(my_arg)

    path=r"D:\Thesis\rerun\wor2vec\fold_0\vocab_200.json"
    with open(path,'r') as f:
        wdv=json.load(f)

       
    with open(r'D:\Thesis\Legal_AI\script\legal.1.1\facts\1440.txt', 'r',encoding="utf8") as f:
        context=nltk.tokenize.sent_tokenize(f.read())
        
    

    x=torch.Tensor(wdv['delhi']).reshape(1,1,len(wdv['delhi']))
    y=torch.Tensor(wdv['high']).reshape(1,1,len(wdv['high']))
    out=torch.cat((x,y),dim=0)
    ip=wd.forward(out)

    # batch_size = ip.size(dim=0)
    
    # num_sentences = ip.size(dim=1)
    # sentence_length = ip.size(dim=2)
    # input_size = ip.size(dim=3)
    # ip = ip.view(batch_size*num_sentences, sentence_length, input_size)

    att=WordAttn(my_arg.input_dim,device='cpu')
    att.forward(ip)
    #print(wd.forward(x))


main()
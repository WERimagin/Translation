#https://www.pytry3g.com/entry/pytorch-seq2seq

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from func import constants
from func.utils import Word2Id,make_tensor,make_vec,make_vec_c,to_var
import numpy as np
from model.transformer.module import Attention,MultiHeadAttention,EncoderLayer,DecoderLayer

#(batch_size,seq_len,dim)
def position_encoder(batch_size,seq_len,dim,device):
    output=[[[np.sin(pos/(10000**(2*i/dim))) if i%2==0 else np.cos(pos/(10000*(2*i/dim))) for i in range(dim)] \
                                                                                    for pos in range(seq_len)] \
                                                                                    for _ in range(batch_size)]
    output=torch.tensor(output,dtype=torch.float).to(device)
    return output

class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()

        self.hidden_size=args.hidden_size
        self.device=args.device

        self.word_embed=nn.Embedding(args.vocab_size, args.hidden_size,padding_idx=constants.PAD)
        self.layers=nn.ModuleList([EncoderLayer(args) for _ in range(args.n_layers)])

    #input:(batch,seq_len)
    def forward(self,input):
        batch_size,seq_len=input.size()

        pad_mask=torch.eq(input,constants.PAD)#(batch,seq_len)
        pad_mask=pad_mask.view(batch_size,1,seq_len).repeat(1,seq_len,1).to(self.device)#(batch,seq_len,seq_len)

        output=self.word_embed(input)#(batch,seq_len,embed_size)
        output=torch.add(output,position_encoder(batch_size,seq_len,self.hidden_size,self.device))

        #6層分
        for layer in self.layers:
            output=layer(output)

        return output


class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder,self).__init__()
        self.hidden_size=args.hidden_size
        self.device=args.device

        self.word_embed=nn.Embedding(args.vocab_size, args.hidden_size, padding_idx=constants.PAD)
        self.layers=nn.ModuleList([DecoderLayer(args) for _ in range(args.n_layers)])
        self.out=nn.Linear(args.hidden_size,args.vocab_size)

    #input:(batch,seq_len)
    #encoder_output:(batch,encoder_seq_len,hidden_size)
    def forward(self,input,encoder_output):
        input=input[:,:-1]
        batch_size,seq_len=input.size()
        encoder_seq_len=encoder_output.size(1)

        pad_mask=torch.eq(input,constants.PAD)#(batch,seq_len)
        pad_mask=pad_mask.view(batch_size,1,seq_len).repeat(1,seq_len,1).to(self.device)#(batch,seq_len,seq_len)

        output=self.word_embed(input)#(batch,seq_len,embed_size)
        output=torch.add(output,position_encoder(batch_size,seq_len,self.hidden_size,self.device))

        #6層分のlayerを通す
        for layer in self.layers:
            output=layer(output,encoder_output)

        output=self.out(output)
        #output=torch.softmax(output,dim=-1)

        return output

class Transformer(nn.Module):
    def __init__(self,args):
        super(Transformer, self).__init__()
        args.n_layers=6
        args.head_num=8
        args.hidden_size=512

        self.encoder=Encoder(args)
        self.decoder=Decoder(args)


    #input_words:(batch,seq_len)
    def forward(self, input_word,output_word,train=True,beam=False):

        encoder_output=self.encoder(input_word)#(batch,seq_len,hidden_size)
        output=self.decoder(output_word,encoder_output)
        return output

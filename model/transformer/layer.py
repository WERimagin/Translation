import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.transformer.module import MultiHeadAttention,FeedForward

class EncoderLayer(nn.Module):
    def __init__(self,args):
        super(EncoderLayer,self).__init__()

        self.hidden_size=args.hidden_size

        self.self_attention=MultiHeadAttention(args)
        self.ff1=nn.Linear(self.hidden_size,self.hidden_size)
        self.ff2=nn.Linear(self.hidden_size,self.hidden_size)
        self.norm=nn.LayerNorm(self.hidden_size)
        self.dropout=nn.Dropout(args.dropout)
        self.feedforward=FeedForward(args)

    #input:(batch,seq_len,dim)
    def forward(self,input,self_attention_mask,non_pad_mask):
        #MultiHead
        output=self.self_attention(input,input,input,self_attention_mask)#(batch,seq_len,dim)
        output*=non_pad_mask

        ##FF
        output=self.feedforward(output)
        output*=non_pad_mask

        return output

class DecoderLayer(nn.Module):
    def __init__(self,args):
        super(DecoderLayer,self).__init__()

        self.hidden_size=args.hidden_size

        self.self_attention=MultiHeadAttention(args)
        self.enc_dec_attention=MultiHeadAttention(args)
        self.ff1=nn.Linear(self.hidden_size,self.hidden_size)
        self.ff2=nn.Linear(self.hidden_size,self.hidden_size)
        self.norm=nn.LayerNorm(self.hidden_size)
        self.dropout=nn.Dropout(args.dropout)
        self.feedforward=FeedForward(args)

    #input:(batch,seq_len,dim)
    def forward(self,input,encoder_output,self_attention_mask,enc_dec_attention_mask,non_pad_mask):

        output=self.self_attention(input,input,input,self_attention_mask)#(batch,seq_len,dim)
        output*=non_pad_mask

        output=self.enc_dec_attention(encoder_output,output,encoder_output,enc_dec_attention_mask)#(batch,seq_len,dim)
        output*=non_pad_mask

        output=self.feedforward(output)
        output*=non_pad_mask

        return output

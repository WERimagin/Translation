#https://www.pytry3g.com/entry/pytorch-seq2seq

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from func import constants
from func.utils import Word2Id,make_tensor,make_vec,make_vec_c,to_var
import numpy as np
from model.transformer.module import MultiHeadAttention,EncoderLayer,DecoderLayer

#PADのところは0,それ以外は1
#(batch,seq_len)
def get_non_pad_mask(seq):
    return seq.ne(constants.PAD).type(torch.float).unsqueeze(-1)#(batch,seq_len,1)

#(batch,k_seq_len)
#(batch,q_seq_len)
def get_attn_key_pad_mask(seq_k, seq_q):
    ''' For masking out the padding part of key sequence. '''

    # Expand to fit the shape of key query attention matrix.
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(constants.PAD)#(batch,k_seq_len)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1) #(batch,q_seq_len,k_seq_len) # b x lq x lk

    return padding_mask

def get_subsequent_mask(seq,device):
    ''' For masking out the subsequent info. '''

    sz_b, len_s = seq.size()
    subsequent_mask = torch.triu(
        torch.ones((len_s, len_s), device=device, dtype=torch.uint8), diagonal=1)
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)#(batch,seq_len,seq_len)# b x ls x ls

    return subsequent_mask

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

        #self-attenのマスク
        slf_attn_mask = get_attn_key_pad_mask(input,input)#(batch,q_seq_len,k_seq_len)
        #反対,padではないところが1となる。
        non_pad_mask = get_non_pad_mask(input)#(batch,seq_len,1)

        output=self.word_embed(input)#(batch,seq_len,embed_size)
        output=output+position_encoder(batch_size,seq_len,self.hidden_size,self.device)

        #6層分
        for layer in self.layers:
            output=layer(
                output,
                self_attention_mask=slf_attn_mask,
                non_pad_mask=non_pad_mask)

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
    #src_input:(batch,src_seq_len)
    #src_output:(batch,encoder_seq_len,hidden_size)
    def forward(self,input,src_input,src_output):
        input=input[:,:-1]

        batch_size,seq_len=input.size()
        enc_seq_len=src_output.size(1)

        # -- Prepare masks
        non_pad_mask = get_non_pad_mask(input)#(batch,seq_len,1)

        #subseq_mask
        slf_attn_mask_subseq = get_subsequent_mask(input,self.device)#(batch,q_seq_len,q_seq_len)
        #self_pad_mask
        slf_attn_mask_keypad = get_attn_key_pad_mask(input,input)#(batch,q_seq_len,q_seq_len)
        #subseq_maskとpad_maskを足して2を1に直す
        slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)#(batch,q_seq_len,q_seq_len)

        #dec_enc_pad_mask
        dec_enc_attn_mask = get_attn_key_pad_mask(src_input,input)#(batch,q_seq_len,k__seq_len)

        output=self.word_embed(input)#(batch,seq_len,embed_size)
        output=torch.add(output,position_encoder(batch_size,seq_len,self.hidden_size,self.device))

        #6層分のlayerを通す
        for layer in self.layers:
            output=layer(
                output,
                src_output,
                self_attention_mask=slf_attn_mask,
                enc_dec_attention_mask=dec_enc_attn_mask,
                non_pad_mask=non_pad_mask)

        output=self.out(output)

        return output

class Transformer(nn.Module):
    def __init__(self,args):
        super(Transformer, self).__init__()
        args.n_layers=6
        args.head_num=8
        args.hidden_size=512
        args.dropout=0.1

        self.encoder=Encoder(args)
        self.decoder=Decoder(args)

    #input_words:(batch,seq_len)
    def forward(self,src,tgt,train=True,beam=False):
        src_output=self.encoder(src)#(batch,seq_len,hidden_size)
        output=self.decoder(tgt,src,src_output)#(batch,seq_len,vocab_size)
        return output

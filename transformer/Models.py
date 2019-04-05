#モデルの定義
#encoder,decoderの定義

''' Define the Transformer model '''
import torch
import torch.nn as nn
import numpy as np
import transformer.Constants as Constants
from transformer.Layers import EncoderLayer, DecoderLayer

__author__ = "Yu-Hsiang Huang"

#PADのところは0,それ以外は1
#(batch,seq_len)
def get_non_pad_mask(seq):
    assert seq.dim() == 2
    return seq.ne(Constants.PAD).type(torch.float).unsqueeze(-1)#(batch,seq_len,1)

#n_position:seq_len
#d_hid:次元数
def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    ''' Sinusoid position encoding table '''

    #sin,cosの中身を計算
    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)


    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.

    return torch.FloatTensor(sinusoid_table)#(seq_len,hidden_size)

#(batch,k_seq_len)
#(batch,q_seq_len)
def get_attn_key_pad_mask(seq_k, seq_q):
    ''' For masking out the padding part of key sequence. '''

    # Expand to fit the shape of key query attention matrix.
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(Constants.PAD)#(batch,k_seq_len)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1) #(batch,q_seq_len,k_seq_len) # b x lq x lk

    return padding_mask

def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''

    sz_b, len_s = seq.size()
    subsequent_mask = torch.triu(
        torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=1)
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)  #(batch,seq_len,seq_len)# b x ls x ls

    return subsequent_mask

#(batch_size,seq_len,dim)
def position_encoder(batch_size,seq_len,dim,device):
    output=[[[np.sin(pos/(10000**(2*i/dim))) if i%2==0 else np.cos(pos/(10000*(2*i/dim))) for i in range(dim)] \
                                                                                    for pos in range(seq_len)] \
                                                                                    for _ in range(batch_size)]
    output=torch.tensor(output,dtype=torch.float).to(device)
    return output

class Encoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self,
            n_src_vocab, len_max_seq, d_word_vec,
            n_layers, n_head, d_k, d_v,
            d_model, d_inner, dropout=0.1):

        super().__init__()

        n_position = len_max_seq + 1

        self.src_word_emb = nn.Embedding(
            n_src_vocab, d_word_vec, padding_idx=Constants.PAD)

        self.position_enc = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(n_position, d_word_vec, padding_idx=0),
            freeze=True)

        #同じlayerをn_layers(6個)用意
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

    def forward(self, src_seq, src_pos, return_attns=False):

        enc_slf_attn_list = []

        # -- Prepare masks
        #self-attenのマスク
        slf_attn_mask = get_attn_key_pad_mask(seq_k=src_seq, seq_q=src_seq)#(batch,q_seq_len,k_seq_len)
        #反対,padではないところが1となる。
        non_pad_mask = get_non_pad_mask(src_seq)#(batch,seq_len,1)

        # -- Forward
        #positionをたす
        #enc_output = self.src_word_emb(src_seq) + position_encoder()
        enc_output = self.src_word_emb(src_seq)
        enc_output=enc_output+position_encoder(enc_output.size(0),enc_output.size(1),enc_output.size(2),enc_output.device)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]

        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output,

class Decoder(nn.Module):
    ''' A decoder model with self attention mechanism. '''

    def __init__(
            self,
            n_tgt_vocab, len_max_seq, d_word_vec,
            n_layers, n_head, d_k, d_v,
            d_model, d_inner, dropout=0.1):

        super().__init__()
        n_position = len_max_seq + 1

        self.tgt_word_emb = nn.Embedding(
            n_tgt_vocab, d_word_vec, padding_idx=Constants.PAD)

        self.position_enc = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(n_position, d_word_vec, padding_idx=0),
            freeze=True)

        #同じlayerを6個用意
        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

    def forward(self, tgt_seq, tgt_pos, src_seq, enc_output, return_attns=False):

        dec_slf_attn_list, dec_enc_attn_list = [], []

        # -- Prepare masks
        non_pad_mask = get_non_pad_mask(tgt_seq)#(batch,seq_len,1)

        #subseq_mask
        slf_attn_mask_subseq = get_subsequent_mask(tgt_seq)#(batch,q_seq_len,q_seq_len)
        #self_pad_mask
        slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=tgt_seq, seq_q=tgt_seq)#(batch,q_seq_len,q_seq_len)
        slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)#(batch,q_seq_len,q_seq_len)

        #dec_enc_pad_mask
        dec_enc_attn_mask = get_attn_key_pad_mask(seq_k=src_seq, seq_q=tgt_seq)#(batch,q_seq_len,k__seq_len)

        # -- Forward
        #dec_output = self.tgt_word_emb(tgt_seq) + self.position_enc(tgt_pos)
        dec_output = self.tgt_word_emb(tgt_seq)
        dec_output=dec_output+position_encoder(dec_output.size(0),dec_output.size(1),dec_output.size(2),dec_output.device)

        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
                dec_output, enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask,
                dec_enc_attn_mask=dec_enc_attn_mask)

            if return_attns:
                dec_slf_attn_list += [dec_slf_attn]
                dec_enc_attn_list += [dec_enc_attn]

        if return_attns:
            return dec_output, dec_slf_attn_list, dec_enc_attn_list
        return dec_output,

#モデルの本体
class Transformer(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''

    def __init__(self,args):

        super().__init__()

        n_src_vocab=args.vocab_size
        n_tgt_vocab=args.vocab_size
        len_max_seq=args.src_length
        d_word_vec=512#単語の次元数
        d_model=512#隠れそうの次元数
        d_inner=2048#position-wise-feed-forward
        n_layers=6
        n_head=8
        d_k=64#keyの次元数
        d_v=64
        dropout=args.dropout
        tgt_emb_prj_weight_sharing=True
        emb_src_tgt_weight_sharing=True

        self.encoder = Encoder(
            n_src_vocab=n_src_vocab, len_max_seq=len_max_seq,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            dropout=dropout)

        self.decoder = Decoder(
            n_tgt_vocab=n_tgt_vocab, len_max_seq=len_max_seq,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            dropout=dropout)

        self.tgt_word_prj = nn.Linear(d_model, n_tgt_vocab, bias=False)
        nn.init.xavier_normal_(self.tgt_word_prj.weight)

        assert d_model == d_word_vec, \
        'To facilitate the residual connections, \
         the dimensions of all module outputs shall be the same.'

        if tgt_emb_prj_weight_sharing:
            # Share the weight matrix between target word embedding & the final logit dense layer
            self.tgt_word_prj.weight = self.decoder.tgt_word_emb.weight
            self.x_logit_scale = (d_model ** -0.5)
        else:
            self.x_logit_scale = 1.

        if emb_src_tgt_weight_sharing:
            # Share the weight matrix between source & target word embeddings
            assert n_src_vocab == n_tgt_vocab, \
            "To share word embedding table, the vocabulary size of src/tgt shall be the same."
            self.encoder.src_word_emb.weight = self.decoder.tgt_word_emb.weight

    def forward(self, src_seq, tgt_seq,train=True):
        src_pos=None
        tgt_pos=None

        tgt_seq=tgt_seq[:, :-1]

        enc_output, *_ = self.encoder(src_seq, src_pos)
        dec_output, *_ = self.decoder(tgt_seq, tgt_pos, src_seq, enc_output)
        seq_logit = self.tgt_word_prj(dec_output) * self.x_logit_scale


        return seq_logit
        #return seq_logit.view(-1, seq_logit.size(2))#(batch*seq_len,vocab_size)

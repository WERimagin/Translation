import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

#scaled-dot-product-attention:パラメーターを使用しないから計算が早い
#key:(batch,seq_len,dim_k)
#query:(batch,seq_len,dim_k)
#value:(batch,seq_len,dim_v)
#pad_mask:(batch,seq_len,seq_len)
class ScaledDotProductAttention(nn.Module):
    def __init__(self,args):
        super(ScaledDotProductAttention,self).__init__()
        self.hidden_size=args.hidden_size
        self.scaler=float(self.hidden_size**0.5)

        self.dropout=nn.Dropout(args.dropout)

    def forward(self,key,query,value,mask):
        batch_size,seq_len,_=key.size()

        key=key.transpose(1,2)#(batch,dim_k,seq_len)
        qk=torch.bmm(query,key)#(batch,seq_len,seq_len)
        qk=qk/self.scaler#(batch,seq_len,seq_len)

        if mask is not None:
            #mask=mask.view(1,seq_len,seq_len).repeat(batch_size,1,1)
            qk=qk.masked_fill(mask,-np.inf)


        #print(qk.size(),qk)
        #qk=nn.Dropout(qk)
        #qk=torch.softmax(qk,dim=-1)#(batch,seq_len,seq_len)
        qk=torch.softmax(qk,dim=-1)
        qk=self.dropout(qk)
        output=torch.bmm(qk,value)#(batch,seq_len,dim_v)

        return output

class MultiHeadAttention(nn.Module):
    def __init__(self,args):
        super(MultiHeadAttention,self).__init__()
        self.hidden_size=args.hidden_size
        self.head_num=args.head_num
        self.head_dim=int(self.hidden_size/self.head_num)

        self.Wk=nn.Linear(self.hidden_size,self.hidden_size)
        self.Wq=nn.Linear(self.hidden_size,self.hidden_size)
        self.Wv=nn.Linear(self.hidden_size,self.hidden_size)

        self.Wconcat=nn.Linear(self.hidden_size,self.hidden_size)

        self.Attention=ScaledDotProductAttention(args)
        self.dropout=nn.Dropout(args.dropout)
        self.norm=nn.LayerNorm(self.hidden_size)

    #key:(batch,seq_len,dim)
    #query:(batch,seq_len,dim)
    #value:(batch,seq_len,dim)
    #mask:(batch,seq_len,seq_len)
    def forward(self,key,query,value,mask=None):
        batch,k_seq_len,dim=key.size()
        _,q_seq_len,_=query.size()
        _,v_seq_len,_=value.size()
        residual=query

        head_k=self.Wk(key)#(batch,seq_len,dim)
        head_q=self.Wq(query)#(batch,seq_len,dim)
        head_v=self.Wv(value)#(batch,seq_len,dim)

        #(batch*head_num,seq_len,head_dim)
        head_k=head_k.view(batch,k_seq_len,self.head_num,self.head_dim).permute(2,0,1,3).contiguous().view(batch*self.head_num,k_seq_len,self.head_dim)
        head_q=head_q.view(batch,q_seq_len,self.head_num,self.head_dim).permute(2,0,1,3).contiguous().view(batch*self.head_num,q_seq_len,self.head_dim)
        head_v=head_v.view(batch,v_seq_len,self.head_num,self.head_dim).permute(2,0,1,3).contiguous().view(batch*self.head_num,v_seq_len,self.head_dim)

        mask=mask.repeat(self.head_num,1,1)#(batch*head_dim,q_seq_len,k_seq_len)
        output=self.Attention(head_k,head_q,head_v,mask)#(batch*head_num,q_seq_len,head_dim)

        #(batch,seq_len,dim)
        output=output.view(batch,self.head_num,q_seq_len,self.head_dim).permute(1,2,0,3).contiguous().view(batch,q_seq_len,dim)

        #(batch,seq_len,dim)
        output=self.Wconcat(output)
        output=self.dropout(output)

        output=self.norm(output+residual)

        return output

class FeedForward(nn.Module):
    def __init__(self,args):
        super(FeedForward,self).__init__()

        self.hidden_size=args.hidden_size

        self.ff1=nn.Linear(self.hidden_size,self.hidden_size)
        self.ff2=nn.Linear(self.hidden_size,self.hidden_size)
        self.norm=nn.LayerNorm(self.hidden_size)
        self.dropout=nn.Dropout(args.dropout)

    def forward(self,input):
        residual=input

        output=self.ff2(F.relu(self.ff1(input)))
        output=self.dropout(output)
        output=self.norm(output+residual)

        return output

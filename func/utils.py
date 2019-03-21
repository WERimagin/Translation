import random
import numpy as np
import torch
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from func import constants
import datetime
import argparse
import json
import platform


#使わない
#文のリスト(入力と出力の二つ)を取って、id化した物を返す
#ここでは非numpy,かつサイズがバラバラ->make_vectorでバッチごとに　揃える
class Word2Id:
    def __init__(self,enc_sentences,dec_sentences):
        self.words=["<pad>"]#素性として使うwordのリスト
        self.word2id={}#word->idの変換辞書
        self.id2word={}#id->wordの変換辞書
        self.enc_sentences=enc_sentences
        self.dec_sentences=dec_sentences
        self.vocab_size=0

    def __call__(self):
        #wordsの作成
        print(self.dec_sentences[0])
        sentences=self.enc_sentences+self.dec_sentences
        for sentence in tqdm(sentences):
            for word in sentence:
                if word not in self.words:
                    self.words.append(word)
        self.vocab_size=len(self.words)
        #word2idの作成
        #id2wordの作成
        for i,word in enumerate(self.words):
            self.word2id[word]=i
            self.id2word[i]=word
        #sentence->ids
        enc_id_sentences=[]
        dec_id_sentences=[]
        for sentence in self.enc_sentences:
            sentence=[self.word2id[word] for word in sentence]
            enc_id_sentences.append(sentence)
        for sentence in self.dec_sentences:
            sentence=[self.word2id[word] for word in sentence]
            dec_id_sentences.append(sentence)
        return enc_id_sentences,dec_id_sentences

#batchのidを返す
class BatchMaker:
    def __init__(self,data_size,batch_size,shuffle=True):
        self.data_size=data_size
        self.batch_size=batch_size
        self.data=list(range(self.data_size))
        self.shuffle=shuffle
    def __call__(self):
        if self.shuffle:
            random.shuffle(self.data)
        batches=[]
        batch=[]
        for i in range(self.data_size):
            batch.append(self.data[i])
            if len(batch)==self.batch_size:
                batches.append(batch)
                batch=[]
        if len(batch)>0:
            batches.append(batch)
        return batches

def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

#渡されたデータをpytorchのためにto_varで変換する
def make_tensor(id_number):
    return to_var(torch.from_numpy(np.array(id_number,dtype="long")))

#渡されたデータをpytorchのためにto_varで変換する
def make_vec(sentences):
    maxsize=max([len(sentence) for sentence in sentences])
    sentences_cp=[]
    for sentence in sentences:
        sentences_cp.append(sentence+[constants.PAD]*(maxsize-len(sentence)))
    return to_var(torch.from_numpy(np.array(sentences_cp,dtype="long")))

def make_vec_c(sentences):
    sent_maxsize=max([len(sentence) for sentence in sentences])
    char_maxsize=max([len(word) for sentence in sentences for word in sentence])
    sentence_ex=np.zeros((len(sentences),sent_maxsize,char_maxsize),dtype="long")
    for i,sentence in enumerate(sentences):
        for j,word in enumerate(sentence):
            for k,char in enumerate(word):
                sentence_ex[i,j,k]=char
    return to_var(torch.from_numpy(sentence_ex))

def logger(args,text):
    print(text)
    #サーバーの時のみ、logを記録
    if args.system=="Linux":
        with open("log.txt","a")as f:
            f.write("{}\t{}\n".format(str(datetime.datetime.today()).replace(" ","-"),text))

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_epoch", type=int, default="0", help="input model epoch")
    parser.add_argument("--cuda_number", type=int, default="0", help="specify cuda number")
    parser.add_argument("--train_batch_size", type=int, default="32", help="input train batch size")
    parser.add_argument("--test_batch_size", type=int, default="64", help="input test batch size")

    ##データのオプション
    parser.add_argument("--src_length", type=int, default="30", help="input train batch size")
    parser.add_argument("--tgt_length", type=int, default="30", help="input test batch size")
    parser.add_argument("--model_version", type=int, default="1", help="input test batch size")
    parser.add_argument("--use_interro", type=bool, default=True, help="input test batch size")
    parser.add_argument("--data_rate", type=float, default="1", help="input epoch number")

    #model_hyper_parameter
    parser.add_argument("--embed_size", type=int, default="300", help="input rnn hidden size")
    parser.add_argument("--hidden_size", type=int, default="600", help="input rnn hidden size")
    parser.add_argument("--epoch_num", type=int, default="200", help="input epoch number")
    parser.add_argument("--dropout", type=float, default="0.3", help="input epoch number")
    parser.add_argument("--layer_size", type=int, default="2", help="input epoch number")
    parser.add_argument("--vocab_size", type=int, default="10000", help="input epoch number")
    parser.add_argument("--lr", type=float, default="0.001", help="input epoch number")
    parser.add_argument("--teacher_rate", type=float, default="0.5", help="input epoch number")

    #そのたのパラメーター
    parser.add_argument("--print_iter", type=int, default="50", help="input epoch number")
    parser.add_argument("--not_train", type=bool, default=False, help="input epoch number")
    parser.add_argument("--use_train_data", type=bool, default=False, help="input epoch number")
    parser.add_argument("--model_name", type=str, default="", help="input epoch number")
    parser.add_argument("--include_pad", type=bool, default=False, help="input epoch number")
    args = parser.parse_args()
    args.start_time=str(datetime.datetime.today()).replace(" ","-")
    args.high_epoch=0
    args.high_score=0
    args.system=platform.system()

    return args

#ファイルから文、質問文、word2idなどを読み込み、辞書形式で返す
def data_loader(args,path,first=True):
    with open(path,"r")as f:
        t=json.load(f)
        sources=t["source"]
        targets=t["target"]
    with open("data/word2id.json","r")as f:
        t=json.load(f)#numpy(vocab_size*embed_size)
        s_word2id=t["s_word2id"]
        t_word2id=t["t_word2id"]

    data_size=int(len(sources)*args.data_rate)
    train_data_size=int(data_size*0.9)

    s_word2id={w:i for w,i in s_word2id.items() if i<args.vocab_size}
    s_id2word={i:w for w,i in s_word2id.items()}

    t_word2id={w:i for w,i in t_word2id.items() if i<args.vocab_size}
    t_id2word={i:w for w,i in t_word2id.items()}


    sources_id=[[s_word2id[w] if w in s_word2id else s_word2id["<UNK>"] for w in sent.split()] for sent in sources]
    targets_id=[[t_word2id[w] if w in t_word2id else t_word2id["<UNK>"] for w in sent.split()] for sent in targets]
    targets_id=[[t_word2id["<SOS>"]] + sent + [t_word2id["<EOS>"]] for sent in targets_id]

    train_sources=sources_id[0:train_data_size]
    train_targets=targets_id[0:train_data_size]
    test_sources=sources_id[train_data_size:data_size]
    test_targets=targets_id[train_data_size:data_size]

    train_data={"sources":train_sources,
        "targets":train_targets,
        "s_id2word":s_id2word,
        "t_id2word":t_id2word}

    test_data={"sources":test_sources,
        "targets":test_targets,
        "s_id2word":s_id2word,
        "t_id2word":t_id2word}

    logger(args,"data_size:{}".format(data_size))

    return train_data,test_data


#lossの計算
def loss_calc(predict,target):
    criterion = nn.CrossEntropyLoss(ignore_index=constants.PAD)#<pad>=0を無視
    batch=predict.size(0)
    seq_len=predict.size(1)
    #batchとseq_lenを掛けて一次元にしてentropyを計算
    predict=predict.contiguous().view(batch*seq_len,-1)#(batch*seq_len,vocab_size)
    target=target.contiguous().view(-1)#(batch*seq_len)
    loss=criterion(predict,target)
    return loss

#一つの文につき単語の正解率を計算
#これをbatchにつき計算してsumを返す
def predict_calc(predict,target):
    #predict:(batch,seq_len,embed_size)
    #target:(batch,seq_len)
    type="normal"
    if type=="normal":
        batch=predict.size(0)
        seq_len=predict.size(1)
        predict=predict.contiguous().view(batch*seq_len,-1)
        target=target.contiguous().view(-1)
        predict_rate=(torch.argmax(predict,dim=-1)==target).sum().item()/seq_len
        return predict_rate
    elif type=="bleu":
        predict=torch.argmax(predict,dim=-1).tolist()#(batch,seq_len,embed_size)
        target=target.tolist()#(batch,seq_len)
        predict_sum=0
        for p,t in zip(predict,target):#batchごと
            predict_sum+=nltk.bleu_score.sentence_bleu([p],t)
        return predict_sum

#idからid2wordを使ってwordに戻して返す
def predict_sentence(predict,target,id2word):
    #predict:(batch,seq_len)
    #target:(batch,seq_len)
    predict=torch.argmax(predict,dim=-1).tolist()#(batch,seq_len)
    #EOSの前まで
    predict_list=[]
    #batchの中の一つずつ
    predict_list=[" ".join([id2word[w] for w in sentence[0:index_remake(sentence,constants.EOS)]]) \
                    for sentence in predict] if args.include_pad==True else
                [" ".join([id2word[w] for w in sentence[0:index_remake(sentence,constants.EOS)]])\
                    for sentence in predict]
    #predict_list=[" ".join([id2word[w] for w in sentence[0:index_ramake(sentence,constants.EOS)]])\
    #                                    for sentence in predict]
    return predict_list

#indexの改造,要素がない場合はリストの長さを返す
def index_remake(sentence_list,word):
    if word in sentence_list:
        return sentence_list.index(word)
    else:
        return len(sentence_list)

"""
def predict_sentence(predict,target,id2word):
    #predict:(batch,beam_width,seq_len)
    #target:(batch,seq_len)
    predict=torch.argmax(predict,dim=-1).tolist()#(batch,seq_len)
    #EOSの前まで
    predict_list=[]
    #batchの中の一つずつ
    predict_list=[[" ".join([id2word[w] for w in sentence[0:index_remake(sentence,constants.EOS)]])\
                                        for sentence in sentences]\
                                        for sentences in predict]
    for sentence in predict:
        sentence=[id2word[w] for w in sentence[0:index_remake(sentence,constants.EOS)]]
        sentence=" ".join(sentence)
        predict_list.append(sentence)
    return predict_list
"""

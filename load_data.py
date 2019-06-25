# _*_ coding: utf-8 _*_
import os
import sys
import torch
from torch.nn import functional as F
import numpy as np
from torchtext import data
from torchtext import datasets
from torchtext.vocab import Vectors, GloVe

def load_dataset(test_sen=None):

    """
    tokenizer : Breaks sentences into a list of words. If sequential=False, no tokenization is applied　将句子分成单词列表。 如果sequential = False，沒有ｔｏｋｅｎ被添加
    Field : A class that stores information about the way of preprocessing　存儲processing的方式的信息　https://zhuanlan.zhihu.com/p/31139113
    fix_length : An important property of TorchText is that we can let the input to be variable length, and TorchText will
                 dynamically pad each sequence to the longest sequence in that "batch". But here we are using fi_length which
                 will pad each sequence to have a fix length of 200.   自動pooling到相同長度
                 
    build_vocab : It will first make a vocabulary or dictionary mapping all the unique words present in the train_data to an
                  idx and then after it will use GloVe word embedding to map the index to the corresponding word embedding.
                  先將次變成唯一的idx,然後用glove映射到相應的詞嵌入
                  
    vocab.vectors : This returns a torch tensor of shape (vocab_size x embedding_dim) containing the pre-trained word embeddings.(vocab_size x embedding_dim)的pretrain的東西就產生了
    BucketIterator : Defines an iterator that batches examples of similar lengths together to minimize the amount of padding needed.
    ＃將長度差不多的放在一起pading就不用那麼麻煩
    """
    tokenize = lambda x: x.split()
    TEXT = data.Field(sequential=True, tokenize=tokenize, lower=True, include_lengths=True, batch_first=True, fix_length=20)
    LABEL = data.LabelField(tensor_type=torch.FloatTensor)
    train_data,valid_data,test_data=data.TabularDataset.splits(
        path='./sarcasmdata/ttssvv/rioff/rioffnor/', train='trainData.tsv',test='testData.tsv',validation='devData.tsv', format='tsv',skip_header=True,
        fields=[('text', TEXT), ('label', LABEL)])    #csv有逗号分隔问题改用tsv

    dimen=100       #其实三百维更好
    TEXT.build_vocab(train_data ,vectors=GloVe(name='6B',dim=dimen))
    LABEL.build_vocab(train_data)
    word_embeddings = TEXT.vocab.vectors
    print ("Length of Text Vocabulary: " + str(len(TEXT.vocab)))
    print ("Vector size of Text Vocabulary: ", TEXT.vocab.vectors.size())
    print ("Label Length: " + str(len(LABEL.vocab)))

    train_iter, valid_iter, test_iter = data.BucketIterator.splits((train_data, valid_data, test_data), batch_size=32, sort_key=lambda x: len(x.text), repeat=False, shuffle=True)

    '''Alternatively we can also use the default configurations'''
    # train_iter, test_iter = datasets.IMDB.iters(batch_size=32)

    vocab_size = len(TEXT.vocab)

    return TEXT, vocab_size, word_embeddings, train_iter, valid_iter, test_iter,dimen,



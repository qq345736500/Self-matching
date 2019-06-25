import torch
import torch.nn as nn
import torch.nn.functional as F

class CoattentionNet(nn.Module):
    def __init__(self, batch_size, output_size, vocab_size, embedding_length, weights):
        super().__init__()
        self.batch_size = batch_size
        self.output_size = output_size
        self.vocab_size = vocab_size
        self.embedding_length = embedding_length
        self.word_embeddings = nn.Embedding(vocab_size, embedding_length)       #, padding_idx=0    ????
        self.word_embeddings.weight = nn.Parameter(weights, requires_grad=False)

        self.word_parallel = ParallelCoattention(D=embedding_length)
        self.dp = nn.Dropout(p=0.5)
        self.tanh = nn.Tanh()
        self.linear2 = nn.Linear(embedding_length, output_size)

    def forward(self, input_sentence, batch_size=None):
        word_embeddings = self.word_embeddings(input_sentence)
        word_embeddings = word_embeddings.transpose(2, 1) #[32, 300, 20
        f_w = self.word_parallel(word_embeddings)   #32x1x300         
        f_w=torch.squeeze(f_w, 1)
        final=self.linear2(f_w)  #300x2    32x1x300
        return final



class ParallelCoattention(nn.Module):
    def __init__(self, D):
        super().__init__()
        self.W_b_weight = nn.Parameter(nn.init.xavier_uniform_(torch.FloatTensor(D, D)))
        self.tanh = nn.Tanh()
        self.dp = nn.Dropout(p=0.5)
        self.maxp = nn.MaxPool1d(kernel_size=D)
    def forward(self,  embedding):
        C = self.dp(self.tanh(torch.matmul(torch.matmul(embedding.transpose(2, 1), self.W_b_weight), embedding)))   #32x20x20
        Max_cow=self.maxp(C)
        a_q = F.softmax(Max_cow, dim=1)  #32x20x1
        b = torch.bmm(a_q.transpose(2, 1), embedding.transpose(2, 1))               #   32x1x20  32x20x300

        return b                       #
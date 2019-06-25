import os
import time
import load_data
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
from co_attention import CoattentionNet

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import inspect

TEXT, vocab_size, word_embeddings, train_iter, valid_iter, test_iter,dimen = load_data.load_dataset()    #reture 的東西抓出來用

def clip_gradient(model, clip_value):           #裁剪梯度吧
    params = list(filter(lambda p: p.grad is not None, model.parameters()))
    for p in params:
        p.grad.data.clamp_(-clip_value, clip_value)
    
def train_model(model, train_iter, epoch):
    total_epoch_loss = 0
    total_epoch_acc = 0
    model.cuda()
    optim = torch.optim.RMSprop(filter(lambda p: p.requires_grad, model.parameters()), weight_decay=1e-5)      #改成adam更好
    steps = 0
    model.train()
    for idx, batch in enumerate(train_iter):
        text = batch.text[0]
        target = batch.label
        target = torch.autograd.Variable(target).long()
        if torch.cuda.is_available():
            text = text.cuda()
            target = target.cuda()
        if (text.size()[0] is not 32):# One of the batch returned by BucketIterator has length different than 32.
            continue
        optim.zero_grad()
        prediction = model(text)
        loss = loss_fn(prediction, target)
        num_corrects = (torch.max(prediction, 1)[1].view(target.size()).data == target.data).float().sum()
        acc = 100.0 * num_corrects/len(batch)
        loss.backward()
      #前面define
        optim.step()
        steps += 1
        
        if steps % 100 == 0:
            print ('Epoch: {0:02}'.format(epoch+1), 'Idx: {0:02}'.format(idx+1), 'Training Loss: {0:.4f}'.format(loss.item()), 'Training Accuracy: {0: .2f}%'.format(acc.item()))

        total_epoch_loss += loss.item()
        total_epoch_acc += acc.item()
        
    return total_epoch_loss/len(train_iter), total_epoch_acc/len(train_iter)


def calculate_metric(metric_fn, true_y, pred_y):
    # multi class problems need to have averaging method
    if "average" in inspect.getfullargspec(metric_fn).args:
        return metric_fn(true_y, pred_y, average="macro")
    else:
        return metric_fn(true_y, pred_y)
def print_scores(p, r, f1, a, batch_size):
    # just an utility printing function
    for name, scores in zip(("precision", "recall", "F1", "accuracy"), (p, r, f1, a)):
        print('\t{0}'.format(name) ,  '{0:.4f}'.format(sum(scores)/batch_size))

def eval_model(model, val_iter):
    total_epoch_loss = 0
    total_epoch_acc = 0
    model.eval()
    precision, recall, f1, accuracy = [], [], [], []
    ooo=[]
    with torch.no_grad():
        for idx, batch in enumerate(val_iter):
            text = batch.text[0]
            if (text.size()[0] is not 32):
                continue
            target = batch.label
            target = torch.autograd.Variable(target).long()
            if torch.cuda.is_available():
                text = text.cuda()
                target = target.cuda()          #Y
            prediction = model(text)   #outputs

            loss = loss_fn(prediction, target)
            total_epoch_loss += loss.item()  # val_losses
            predicted_classes = torch.max(prediction, 1)[1]

            for acc, metric in zip((precision, recall, f1, accuracy),(precision_score, recall_score, f1_score, accuracy_score)):
                acc.append(calculate_metric(metric, target.cpu(), predicted_classes.cpu()))
    print(' validation loss:{0:.5f}'.format(total_epoch_loss/len(val_iter)))
    print_scores(precision, recall, f1, accuracy, len(val_iter))
	

learning_rate = 1e-2
batch_size = 32
output_size = 2
embedding_length = dimen

model =CoattentionNet(batch_size, output_size, vocab_size, embedding_length, word_embeddings)
loss_fn = F.cross_entropy

for epoch in range(200):
    train_loss, train_acc = train_model(model, train_iter, epoch)
    print('Epoch: {0:02}'.format(epoch + 1), 'Train Loss: {0:.3f}'.format(train_loss),
          'Train Acc: {0:.2f}%'.format(train_acc))

    eval_model(model, valid_iter)

eval_model(model, test_iter)        #test



# # test_sen2 = "Ohh, such a ridiculous movie. Not gonna recommend it to anyone. Complete waste of time and money."
# # test_sen2 = TEXT.preprocess(test_sen2)
# # test_sen2 = [[TEXT.vocab.stoi[x] for x in test_sen2]]
#
# test_sen = np.asarray(test_sen1)
# test_sen = torch.LongTensor(test_sen)
# test_tensor = Variable(test_sen, volatile=True)
# test_tensor = test_tensor.cuda()
# print(test_tensor)
# model.heat = False
# model.eval()
#
# output = model(test_tensor, 1)
#
# out = F.softmax(output, 1)
# if (torch.argmax(out[0]) == 1):
#     print ("Sentiment: Sarc")
# else:
#     print ("Sentiment: Not Sarc")





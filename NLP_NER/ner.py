import torch 
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import numpy as np
import pandas as pd


# if a word is not in your vocabulary use len(vocabulary) as the encoding
class NERDataset(Dataset):
    def __init__(self, df_enc):
        self.df = df_enc

    def __len__(self):
        """ Length of the dataset """
        ### BEGIN SOLUTION
        L = len(self.df)-4
        ### END SOLUTION
        return L

    def __getitem__(self, idx):
        """ returns x[idx], y[idx] for this dataset
        
        x[idx] should be a numpy array of shape (5,)
        """
        ### BEGIN SOLUTION
        x = self.df.word[idx:idx+5].values.reshape(5,)
        y = self.df.label.values[idx+2]
        ### END SOLUTION
        return x, y 


def label_encoding(cat_arr):
   """ Given a numpy array of strings returns a dictionary with label encodings.

   First take the array of unique values and sort them (as strings). 
   """
   ### BEGIN SOLUTION
   cat_arr = cat_arr.astype(str)
   input_unique = sorted(np.unique(cat_arr))
   encoding = np.arange(len(input_unique))
   vocab2index = dict(zip(input_unique, encoding))
   ### END SOLUTION
   return vocab2index


def dataset_encoding(df, vocab2index, label2index):
    """Apply vocab2index to the word column and label2index to the label column

    Replace columns "word" and "label" with the corresponding encoding.
    If a word is not in the vocabulary give it the index V=(len(vocab2index))
    """
    V = len(vocab2index)
    df_enc = df.copy()
    ### BEGIN SOLUTION
    df_enc.word = df_enc.word.apply(lambda x: vocab2index.get(x, V))
    df_enc.label = df_enc.label.apply(lambda x: label2index[x])
    ### END SOLUTION
    return df_enc


class NERModel(nn.Module):
    def __init__(self, vocab_size, n_class, emb_size=50, seed=3):
        """Initialize an embedding layer and a linear layer
        """
        super(NERModel, self).__init__()
        torch.manual_seed(seed)
        ### BEGIN SOLUTION
        self.emb_size = emb_size
        self.word_emb = nn.Embedding(vocab_size, emb_size)
        self.linear = nn.Linear(emb_size*5, n_class)
        ### END SOLUTION
        
    def forward(self, x):
        """Apply the model to x
        
        1. x is a (N,5). Lookup embeddings for x
        2. reshape the embeddings (or concatenate) such that x is N, 5*emb_size 
           .flatten works
        3. Apply a linear layer
        """
        ### BEGIN SOLUTION
        # W1 = self.word_emb(x[0])
        # W2 = self.word_emb(x[1])
        # W3 = self.word_emb(x[2])
        # W4 = self.word_emb(x[3])
        # W5 = self.word_emb(x[4])
        # X = torch.cat((W1, W2, W3, W4, W5), 1)
        X = self.word_emb(x).reshape(-1, 5*self.emb_size)
        x = self.linear(X)
        ### END SOLUTION
        return x

def get_optimizer(model, lr = 0.01, wd = 0.0):
    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    return optim

def train_model(model, optimizer, train_dl, valid_dl, epochs=10):
    for i in range(epochs):
        ### BEGIN SOLUTION
        model.train()
        total_loss = 0
        total = 0
        for x, y in train_dl:
            words =torch.LongTensor(x)
            label =torch.LongTensor(y)

            y_hat = model(words)
            loss = F.cross_entropy(y_hat, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += x.size(0)*loss.item()
            total += x.size(0)

        train_loss = total_loss/total
        ### END SOLUTION
        valid_loss, valid_acc = valid_metrics(model, valid_dl)
        print("train loss  %.3f val loss %.3f and accuracy %.3f" % (
            train_loss, valid_loss, valid_acc))

def valid_metrics(model, valid_dl):
    ### BEGIN SOLUTION
    model.eval()
    y_hats = []
    ys = []
    total_loss = 0
    total = 0
    for x, y in valid_dl:
        words = torch.LongTensor(x)
        label = torch.LongTensor(y)
        y_hat = model(words)
        loss = F.cross_entropy(y_hat, label)

        y_hats.extend(y_hat.detach().numpy())
        ys.extend(label.detach().numpy())
        total_loss += x.size(0) * loss.item()
        total += x.size(0)

    y_pred = np.argmax(y_hats, axis=1)
    correct = np.sum(y_pred == ys)
    accuracy = correct/len(y_hats)
    val_loss = total_loss/total
    val_acc = accuracy

    # END SOLUTION
    ### END SOLUTION
    return val_loss, val_acc


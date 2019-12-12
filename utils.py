import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.model_selection import train_test_split


def split_train_test(test_size, seq_id, seq_input, seq_mask, seq_AO, seq_CP):
    '''
    To split the dataset to train set and valid set
    '''
    idx = np.random.permutation(range(seq_id.shape[0]))
    train_idx, valid_idx = train_test_split(idx, test_size=test_size)

    trainset = (seq_id[train_idx], seq_input[train_idx], seq_mask[train_idx], seq_AO[train_idx], seq_CP[train_idx])
    validset = (seq_id[valid_idx], seq_input[valid_idx], seq_mask[valid_idx], seq_AO[valid_idx], seq_CP[valid_idx])

    return trainset, validset

def dataloader(trainset, validset, batchSize=1):
    '''
    To return the dataloader of trainset and valid set
    Change into tensor can calculate in torch
    '''

    train_id, train_input, train_mask, train_AO, train_CP = trainset
    valid_id, valid_input, valid_mask, valid_AO, valid_CP = validset

    # make tensor for train set
    train_id = torch.LongTensor(train_id)
    train_input = torch.LongTensor(train_input)
    train_mask = torch.LongTensor(train_mask)
    train_AO = torch.LongTensor(train_AO)
    train_CP = torch.LongTensor(train_CP)

    # make tensor dataset for train set
    train_dataset = TensorDataset(train_id, train_input, train_mask, train_AO, train_CP)
    train_sampler = RandomSampler(train_dataset)
    train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batchSize)

    # make tensor for valid set
    valid_id = torch.LongTensor(valid_id)
    valid_input = torch.LongTensor(valid_input)
    valid_mask = torch.LongTensor(valid_mask)
    valid_AO = torch.LongTensor(valid_AO)
    valid_CP = torch.LongTensor(valid_CP)

    # make tensor dataset for valid set
    valid_dataset = TensorDataset(valid_id, valid_input, valid_mask, valid_AO, valid_CP)
    valid_sampler = RandomSampler(valid_dataset)
    valid_loader = DataLoader(valid_dataset, sampler=valid_sampler, batch_size=batchSize)

    return train_loader, valid_loader

def cal_metrics(pred, true_l):
    eps = 0.00001
    pred_len = len(pred)
    gold_len = len(true_l)
    right = len(set(pred) & set(true_l))
        
    precision = right / (pred_len + eps)
    recall = right / (gold_len + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    print(precision, recall, f1, "\n") 
    return f1
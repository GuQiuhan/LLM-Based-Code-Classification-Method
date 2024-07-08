import pandas as pd
from tqdm.auto import tqdm
tqdm.pandas()
import random
import torch
import time
import numpy as np
from gensim.models.word2vec import Word2Vec
from model import BatchProgramClassifier
from torch.autograd import Variable
from torch.utils.data import DataLoader
from config import *
import os
import sys


def get_batch(dataset, idx, bs):
    tmp = dataset.iloc[idx: idx+bs]
    data, labels = [], []
    for _, item in tmp.iterrows():
        data.append(item[1])
        labels.append(item[2]-1) # 这里为什么要减1？
    return data, torch.LongTensor(labels)


if __name__ == '__main__':
    root = 'data/'
    #train_data, val_data, test_data=Pipeline(root).run()
    train_data = pd.read_pickle(root+'train/blocks.pkl')
    train_data = train_data.iloc[:700]
    #val_data = pd.read_pickle(root + 'dev/blocks.pkl')
    #test_data = pd.read_pickle(root+'test/blocks.pkl')

    word2vec = Word2Vec.load(root+"train/embedding/node_w2v_128").wv
    embeddings = np.zeros((word2vec. vectors.shape[0] + 1, word2vec.vectors.shape[1]), dtype="float32")
    embeddings[:word2vec.vectors.shape[0]] = word2vec.vectors

    MAX_TOKENS = word2vec.vectors.shape[0]
    EMBEDDING_DIM = word2vec.vectors.shape[1]

    model = BatchProgramClassifier(EMBEDDING_DIM,HIDDEN_DIM,MAX_TOKENS+1,ENCODE_DIM,LABELS,BATCH_SIZE,
                                   USE_GPU, embeddings)
    if USE_GPU:
        model.cuda()

    parameters = model.parameters()
    optimizer = torch.optim.Adamax(parameters)
    loss_function = torch.nn.CrossEntropyLoss()

    train_loss_ = []
    val_loss_ = []
    train_acc_ = []
    val_acc_ = []
    best_acc = 0.0


    log = open("log.txt", mode = "a+", encoding = "utf-8")
    print('Start training...')
    print('Start training...',file=log)
    # training procedure
    best_model = model
    for epoch in range(EPOCHS):
        torch.cuda.empty_cache()

        start_time = time.time()

        total_acc = 0.0
        total_loss = 0.0
        total = 0.0
        i = 0
        
        # Create inner progress bar
        pbar = tqdm(total=len(train_data), desc=f'Epoch {epoch+1}/{EPOCHS}', position=1, leave=False)

        while i < len(train_data):
            torch.cuda.empty_cache()
            batch = get_batch(train_data, i, BATCH_SIZE)
            i += BATCH_SIZE
            train_inputs, train_labels = batch
            if USE_GPU:
                train_inputs, train_labels = train_inputs, train_labels.cuda()

            model.zero_grad()
            model.batch_size = len(train_labels)
            model.hidden = model.init_hidden()
            output = model(train_inputs)

            loss = loss_function(output, Variable(train_labels))
            loss.backward()
            optimizer.step()

            # calc training acc
            _, predicted = torch.max(output.data, 1)
            total_acc += (predicted == train_labels).sum()
            total += len(train_labels)
            total_loss += loss.item()*len(train_inputs)
            pbar.update(BATCH_SIZE)

        train_loss_.append(total_loss / total)
        train_acc_.append(total_acc.item() / total)
        
        '''
        # validation epoch
        total_acc = 0.0
        total_loss = 0.0
        total = 0.0
        i = 0
        while i < len(val_data):
            torch.cuda.empty_cache()
            batch = get_batch(val_data, i, BATCH_SIZE)
            i += BATCH_SIZE
            val_inputs, val_labels = batch
            if USE_GPU:
                val_inputs, val_labels = val_inputs, val_labels.cuda()

            model.batch_size = len(val_labels)
            model.hidden = model.init_hidden()
            output = model(val_inputs)

            loss = loss_function(output, Variable(val_labels))

            # calc valing acc
            _, predicted = torch.max(output.data, 1)
            total_acc += (predicted == val_labels).sum()
            total += len(val_labels)
            total_loss += loss.item()*len(val_inputs)
        val_loss_.append(total_loss / total)
        val_acc_.append(total_acc.item() / total)
        
        '''
        end_time = time.time()
        if total_acc/total > best_acc:
            best_model = model
        #print('[Epoch: %3d/%3d] Training Loss: %.4f, Validation Loss: %.4f, Training Acc: %.3f, Validation Acc: %.3f, Time Cost: %.3f s'% (epoch + 1, EPOCHS, train_loss_[epoch], val_loss_[epoch],train_acc_[epoch], val_acc_[epoch], end_time - start_time))
        
        print('[Epoch: %3d/%3d] Training Loss: %.4f, Training Acc: %.3f, Time Cost: %.3f s'% (epoch + 1, EPOCHS, train_loss_[epoch],train_acc_[epoch], end_time - start_time))
        print('[Epoch: %3d/%3d] Training Loss: %.4f, Training Acc: %.3f, Time Cost: %.3f s'% (epoch + 1, EPOCHS, train_loss_[epoch],train_acc_[epoch], end_time - start_time),file=log)

        best_encoder_model=model.state_dict()
        torch.save(best_encoder_model, 'saved_model/best_encoder_model_'+str(epoch)+'.bin')
    
    log.close()
    '''
    total_acc = 0.0
    total_loss = 0.0
    total = 0.0
    i = 0
    model = best_model
    while i < len(test_data):
        batch = get_batch(test_data, i, BATCH_SIZE)
        i += BATCH_SIZE
        test_inputs, test_labels = batch
        if USE_GPU:
            test_inputs, test_labels = test_inputs, test_labels.cuda()

        model.batch_size = len(test_labels)
        model.hidden = model.init_hidden()
        output = model(test_inputs)

        loss = loss_function(output, Variable(test_labels))

        _, predicted = torch.max(output.data, 1)
        total_acc += (predicted == test_labels).sum()
        total += len(test_labels)
        total_loss += loss.item() * len(test_inputs)
    print("Testing results(Acc):", total_acc.item() / total)
    '''

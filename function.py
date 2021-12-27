'''
this file includes parts of functions that should be used in homework1
'''

from tqdm import tqdm
import random
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
import utils.dataset as dataset

def evaluate(model, data_loader):
    '''
    loss:cross entropy
    this function aims to evalute the model performance wrt a specific data loader
    *input: 
    model, data loader
    *output:
    accuracy, loss
    '''
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    model.eval()
    acc = 0
    num = 0
    total_loss = 0
    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device).float(), y.to(device)
            out = model(x)
            loss = criterion(out, y.long())
            total_loss += loss.item()
            num += y.size(0)
            
            pred_ind = torch.max(out, 1).indices.view(-1)
            acc += (pred_ind == y).sum().item()
            
        acc /= num
        total_loss = total_loss/num
        return acc, total_loss


def train(model, optimizer, scheduler, train_loader, val_loader, EPOCHS, MODELPATH):
    '''
    training procedure based on cross entropy loss(or your preferable loss)
    *input:
    model: model
    optimizer: SGD or Adam or st else
    scheduler: make learning rate designable, for eg, ExponentialLR
    criterion:default is cross entropy loss
    train_loader: training set
    val_loader: validation set
    EPOCHS: total epochs
    MODELPATH: the place where you save your model
    '''
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    criterion = nn.CrossEntropyLoss()
    
    optimizer = optimizer
    scheduler = scheduler
    
    model.to(device)
    model.train()
    
    best_loss = 99999
    # print('start training..')
    for epoch in range(EPOCHS):
        epoch_loss = 0
        train_acc=0
        num = 0
        for x, y in tqdm(train_loader):
            x = x.to(device).float()
            y = y.to(device)
            
            out = model(x)
            
            #calculate loss part
            loss = criterion(out, y.long())
            epoch_loss += loss.item()
            num += y.size(0)
            
            #calculate train acc
            pred_ind = torch.max(out, 1).indices.view(-1)
            train_acc += (pred_ind == y).sum().item()
            
            #backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if scheduler is not None:
            scheduler.step() 

        epoch_loss = epoch_loss / num
        train_acc /= num
        test_acc, test_loss = evaluate(model, val_loader)
        if test_loss < best_loss:
            best_acc = test_acc
            torch.save(model.state_dict(), MODELPATH)
        
        if (epoch + 1) % 1 == 0:
            print(f'epoch {epoch+1}%%%%%%%%%%%%%%%%%%%%')
            print(f"train set===>loss: {epoch_loss:.4f}   acc: {100* train_acc:.2f}%")
            print(f"val set=====>loss: {test_loss:.4f}   acc: {100* test_acc:.2f}%")
            print()


def inference(model, weights_path, test_data_dir, save_sub_path, num_augmentation, test_trans, batch_size):
    test_size = 13153
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    FishNames = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']

    # get all filenames of testing data
    test_stg1_name = os.listdir(os.path.join(test_data_dir, 'test_stg1'))
    test_stg1_name = sorted(test_stg1_name)
    test_stg2_name = os.listdir(os.path.join(test_data_dir, 'test_stg2'))
    test_stg2_name = sorted(test_stg2_name)
    for i in range(len(test_stg2_name)):
        test_stg2_name[i] = 'test_stg2/' + test_stg2_name[i]

    img_name_list = test_stg1_name + test_stg2_name

    # test data generator for prediction
    testset = dataset.fish_test(test_data_dir, transform=test_trans)
    test_loader = DataLoader(dataset = testset,
                            batch_size = batch_size,
                            shuffle = False,
                            num_workers = 0,
                            pin_memory = True)

    print('Loading model and weights from training process ...')

    model = model.to(device).eval()
    model.load_state_dict(torch.load(weights_path))
    final_predictions = torch.zeros([test_size, 8], device=device).float()
    for idx in range(num_augmentation):
        print('{}th augmentation for testing ...'.format(idx))
        predictions = torch.empty(0).to(device=device).float()
        # lower=0
        with torch.no_grad():
            for images, _ in tqdm(test_loader):
                images = images.to(device)
                predictions = torch.concat((predictions, model(images)), dim=0)
        predictions = nn.Softmax(dim=1)(predictions)
        final_predictions += predictions
        
    final_predictions /= num_augmentation

    print('Begin to write submission file ..')
    f_submit = open(save_sub_path, 'w')
    f_submit.write('image,ALB,BET,DOL,LAG,NoF,OTHER,SHARK,YFT\n')
    for i, image_name in enumerate(img_name_list):
        pred = ['%.6f' % p for p in final_predictions[i, :]]
        if i % 100 == 0:
            print('{} / {}'.format(i, test_size))
        f_submit.write('%s,%s\n' % (image_name, ','.join(pred)))

    f_submit.close()

    print('Submission file successfully generated!')
 
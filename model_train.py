import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from resnet import Bottleneck, ResNet, BasicBlock
import shutil
import os
import random
from torchvision import models
from sklearn.metrics import confusion_matrix
import numpy as np
from PIL import Image

def loading_model(model_name , class_num  , pretrain_model = None , print_model = False ):
    if (model_name == 'ResNet18'):
        model = ResNet(BasicBlock, [3, 3, 3, 3],class_num)
    elif (model_name == 'ResNet34'):
        model = ResNet(BasicBlock, [3, 4, 6, 3],class_num)
    elif (model_name == 'ResNet50'):
        model = ResNet(Bottleneck, [3, 4, 6, 3],class_num)
    elif (model_name == 'ResNet101'):
        model = ResNet(Bottleneck, [3, 4, 23, 3],class_num)
    elif (model_name == 'ResNet152'):
        model = ResNet(Bottleneck, [3, 8, 36, 3],class_num)
    elif (model_name == 'Densenet121'):
        model = models.densenet121(pretrained=False)
        fc_features = model.classifier.in_features  
        model.classifier = torch.nn.Linear(fc_features, class_num)
    elif (model_name == 'shufflenet'):
        model = models.shufflenet_v2_x1_0()
        fc_features = model.fc.in_features  
        model.fc = torch.nn.Linear(fc_features, class_num)
    return model

ImageDatatransforms = transforms.Compose([
        #transforms.RandomAffine(degrees=30, scale=(.9, 1.1), shear=0),
        #transforms.RandomRotation(degrees=20,expand=False),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
        ])

def load_image_data(Data_Folder_name , BATCH_SIZE = 128):
    train_data = datasets.ImageFolder(Data_Folder_name + '/training_set', transform=ImageDatatransforms)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_data = datasets.ImageFolder(Data_Folder_name + '/test_set', transform=ImageDatatransforms)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)
    class_dict = train_data.class_to_idx
    return train_loader, test_loader, class_dict

def model_train(train_loader):
        model.train()
        train_loss = 0
        train_acc = 0
        total = 0
        for i, (data, target) in enumerate(train_loader):
            if train_on_gpu:
                data, target = data.to(device), target.to(device)
            output = model(data)
            _, preds = torch.max(output.data, 1)
            loss = criterion(output, target.long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()*data.size(0)
            train_acc  += (preds == target.data).sum().float()
            total += len(target)
        acc = 100 * train_acc/total
        return acc , train_loss
    
def model_val(test_loader):
    model.eval()
    total = 0
    valid_loss = 0
    valid_acc = 0
    for i, (data, target) in enumerate(test_loader):
        if train_on_gpu:
            data, target = data.to(device), target.to(device)
        output = model(data)
        _, pred = torch.max(output.data, 1)
        loss = criterion(output, target.long())
        valid_loss += loss.item()*data.size(0)
        valid_acc  += (pred == target.data).sum().float()
        total += len(target)
    val_acc = 100*valid_acc/total
    return val_acc , valid_loss

import matplotlib.pyplot as plt
def show_history(tain , val):
    epochs = len(tain)
    epoch = [x for x in range(epochs)]
    plt.plot(epoch ,tain , color = 'blue' , label = 'train_acc')
    plt.plot(epoch ,val , color = 'green' , label = 'val_acc')
    plt.legend()
    plt.title('Train history')
    plt.ylabel('acc')
    plt.xlabel('Epoch')
    plt.savefig("train_history.png")
    plt.show()

def save_model(val_acc_max , acc , val_acc , epoch , save_dir):
    #  if validation acc has increase
    if val_acc >= val_acc_max:
        print('Validation acc increased ({:.6f} --> {:.6f}).  Saving model ...'.format(val_acc_max , val_acc))
        val_acc_max = val_acc
        torch.save(model.state_dict(),  os.path.join(save_dir, 'Ultimately_the_best_model.pth'))
    return val_acc_max

def Training_AI_model(n_epochs, train_loader, test_loader,save_dir):
    acc_list  = []
    train_loss_list = [] 
    val_acc_list = []
    val_loss_list = []
    val_acc_max = 0
    for epoch in range(n_epochs):
        acc , train_loss = model_train(train_loader)
        val_acc , valid_loss = model_val(test_loader)
        train_loss = train_loss/len(train_loader.dataset)
        valid_loss = valid_loss/len(test_loader.dataset)
        if (epoch+1) % 10 == 0:
            print('Train epoch: {}/{} Training Loss: {:.3f} Trainning Accuracy: {:.2f}% Validation Loss: {:.3f} Validation Accuracy: {:.2f}%'.format(epoch+1, num_epochs, train_loss, acc, valid_loss ,val_acc))
        acc_list.append(acc)
        train_loss_list.append(train_loss)
        val_acc_list.append(val_acc)
        val_loss_list.append(valid_loss)
        val_acc_max = save_model(val_acc_max , acc , val_acc , epoch , save_dir)
        torch.cuda.empty_cache() 
    return acc_list , train_loss_list , val_acc_list , val_loss_list

def model_test(model_name, class_num, model_path = 'Ultimately_the_best_model.pth'):
    test_path = 'DATA\\test_set'
    model_folder = 'model'
    mis_path = 'MIS'
    if not os.path.isdir(mis_path):
        os.makedirs(mis_path)
    data = []
    for j in os.listdir(test_path):
        path2 = os.path.join(test_path,j)
        for t in os.listdir(path2):
            data.append([os.path.join(path2,t),j])
    data = np.array(data)
    model = loading_model(model_name , class_num)
    model.load_state_dict(torch.load(model_folder + '/' + model_path))
    model.to(device)
    model.eval()
    prdict_list = torch.zeros(0,dtype=torch.long , device = 'cpu')
    label_list = torch.zeros(0,dtype=torch.long , device = 'cpu')
    for i,t in data:
        image = Image.open(i)
        image1 = image.convert('RGB')
        image = ImageDatatransforms(image1)
        image = image.unsqueeze(0).to(device)
        output = model(image)
        _, pred = torch.max(output.data, 1)
        if pred.cpu()!=int(t):
            path = os.path.join(mis_path,t+'to'+str(pred.cpu().item()))
            if not os.path.isdir(path):
                os.makedirs(path)
            image1.save(os.path.join(path,os.path.split(i)[1]))
        prdict_list = torch.cat([prdict_list,pred.view(-1).cpu()])
        label_list = torch.cat([label_list,torch.tensor(int(t)).view(-1)])
    test_acc = len(prdict_list.numpy()[prdict_list.numpy()==label_list.numpy()])/len(prdict_list.numpy())
    c_matrix = confusion_matrix(label_list.numpy(), prdict_list.numpy())
    return test_acc,c_matrix
  
  

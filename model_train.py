import torch
from torch.autograd import Variable
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
from torchvision.datasets.folder import default_loader
import torch.nn as nn
import shutil
import os
os.chdir(r'C:\Users\USER\Desktop\0513_data')
import random
from sklearn.metrics import confusion_matrix
import numpy as np
from PIL import Image
import time
#from models.resnext import ResNeXt101_32x4d
#from models.inception_v4 import InceptionV4
#from models.xception import Xception
import matplotlib.pyplot as plt
#from models.senet import se_resnet101, se_resnext101_32x4d
#from efficientnet import EfficientNet
import pandas as pd
import cv2
from imgaug import augmenters as iaa
import imgaug

def loading_model(model_name , class_num):

    if (model_name == 'vgg19_bn'):
        model = models.vgg19_bn(pretrained=False)
        fc_features = model.classifier[-1].in_features  
        model.classifier[-1] = torch.nn.Linear(fc_features, class_num, bias=True)
        image_size = 224
    elif (model_name == 'ResNet18'):
        model = models.resnet18(pretrained=False)
        fc_features = model.fc.in_features  
        model.fc = torch.nn.Linear(fc_features, class_num)  
        image_size = 112
    elif (model_name == 'resnet101'):
        model = models.resnet101(pretrained=False)
        fc_features = model.fc.in_features  
        model.fc = torch.nn.Linear(fc_features, class_num, bias=True)
        image_size = 224
    elif (model_name == 'Densenet121'):
        model = models.densenet121(pretrained=False)
        fc_features = model.classifier.in_features  
        model.classifier = torch.nn.Linear(fc_features, class_num, bias=True)
        image_size = 224
    elif (model_name == 'ResNeXt101_32x4d'):
        model = ResNeXt101_32x4d()
        fc_features = model.last_linear.in_features
        model.last_linear = torch.nn.Linear(fc_features, class_num, bias=True)
        image_size = 224
    elif (model_name == 'se_resnet101'):
        model = se_resnet101(num_classes=class_num)
        image_size = 224
    elif (model_name == 'se_resnext101_32x4d'):
        model = se_resnext101_32x4d(num_classes=class_num)
        image_size = 224
    elif (model_name == 'InceptionV4'):
        model = InceptionV4()
        fc_features = model.last_linear.in_features
        model.last_linear = torch.nn.Linear(fc_features, class_num, bias=True)
        image_size = 299
    elif (model_name == 'Xception'):
        model = Xception()
        fc_features = model.fc.in_features
        model.fc = torch.nn.Linear(fc_features, class_num, bias=True)
        image_size = 299
    elif (model_name == 'EfficientNet'):
        model = EfficientNet(num_classes=2)
        image_size = 224
    return model, image_size

seq = iaa.Sequential([
    iaa.imgcorruptlike.Brightness(severity=3),
    iaa.LinearContrast(alpha=3.0)
])

import albumentations as A
from albumentations.pytorch import ToTensorV2

class MySpecialDataset(datasets.ImageFolder):
    def __init__(self, root, transform = None, loader = default_loader, is_test = False):
        super(MySpecialDataset, self).__init__(root=root, loader=loader)
        self.classes, self.class_to_idx = self._find_classes(root)
        self.transform = transform
        
    def __getitem__(self, index):
        image_path, target = self.samples[index]
        # do your magic here
        img = cv2.imread(image_path)
        if 'Task01' in image_path: 
            #imgumantation
            img = seq.augment_image(img) 
            img = img.copy()
        else:
            img = img
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            img = self.transform(image=img)["image"]
        return img, target

class Training_AI_model():
    def __init__(self,Data_Folder_name, model_name, class_num, BATCH_SIZE, criterion, LR ):
        self.model, self.image_size = loading_model(model_name,class_num)
        self.class_num = class_num
        self.model_name = model_name
        self.criterion = criterion
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LR,weight_decay = 0.001)
        self.ImageDatatransforms = A.Compose([
            A.Resize(self.image_size, self.image_size),
            A.OneOf([
                A.HorizontalFlip(p=1),
                A.RandomRotate90(p=1),
                A.VerticalFlip(p=1)            
            ], p=1),
            #A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
            ])
        self.ImageDatatransforms1 = A.Compose([
            A.Resize(self.image_size, self.image_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
            ])
        train_data = MySpecialDataset(Data_Folder_name + '/training_set',self.ImageDatatransforms)
        self.train_loader = torch.utils.data.DataLoader(train_data, batch_size = BATCH_SIZE, shuffle = True)
        test_data = MySpecialDataset(Data_Folder_name + '/test_set',self.ImageDatatransforms1)
        self.test_loader = torch.utils.data.DataLoader(test_data, batch_size = BATCH_SIZE, shuffle = True)
        self.train_on_gpu = torch.cuda.is_available()
        self.device = torch.device('cuda' if self.train_on_gpu else 'cpu')
        self.sigmoid = nn.Sigmoid()
        self.class_dict = train_data.class_to_idx
        self.idx_to_class = {j:i for (i,j) in self.class_dict.items()}
        self.classes = train_data.classes
        self.Data_Folder_name = Data_Folder_name
        
    def model_train(self):
        self.model.train()
        train_loss = 0
        train_acc = 0
        total = 0
        for i, (data, target) in enumerate(self.train_loader):
            if self.train_on_gpu:
                self.model.to(self.device)
                data, target = data.to(self.device), target.to(self.device)
            output = self.model(data)
            if self.class_num == 1:
                output = output.view(-1)
                preds = self.sigmoid(output.data)>0.5
            else:
            #print(output.size(),target.unsqueeze(1).size())
                _, preds = torch.max(output.data, 1)
            loss = self.criterion(output, target.long())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()*data.size(0)
            train_acc  += (preds == target.data).sum().float()
            total += len(target)
        acc = 100 * train_acc/total
        return acc , train_loss
    
    def model_val(self):
        self.model.eval()
        total = 0
        valid_loss = 0
        valid_acc = 0
        for i, (data, target) in enumerate(self.test_loader):
            if self.train_on_gpu:
                self.model.to(self.device)
                data, target = data.to(self.device), target.to(self.device)
            output = self.model(data)
            if self.class_num == 1:
                output = output.view(-1)
                pred = self.sigmoid(output.data)>0.5
            else:
            #print(output.size(),target.unsqueeze(1).size())
                _, pred = torch.max(output.data, 1)
            loss = self.criterion(output, target.long())
            valid_loss += loss.item()*data.size(0)
            valid_acc  += (pred == target.data).sum().float()
            total += len(target)
        val_acc = 100*valid_acc/total
        return val_acc , valid_loss
    
    def save_model(self, val_acc_max , acc , val_acc , epoch , save_dir):
        #  if validation acc has increase
        if val_acc >= val_acc_max:
            #print('Validation acc increased ({:.6f} --> {:.6f}).  Saving model ...'.format(val_acc_max , val_acc))
            val_acc_max = val_acc
            torch.save(self.model.state_dict(),  os.path.join(save_dir, 'Ultimately_the_best_model({}).pth'.format(self.model_name)))
        return val_acc_max
    
    def Training_model(self, n_epochs ,save_dir):
        t1 = time.time()
        acc_list  = []
        train_loss_list = [] 
        val_acc_list = []
        val_loss_list = []
        val_acc_max = 0
        for epoch in range(n_epochs):
            acc , train_loss = self.model_train()
            val_acc , valid_loss = self.model_val()
            train_loss = train_loss/len(self.train_loader.dataset)
            valid_loss = valid_loss/len(self.test_loader.dataset)
            if (epoch+1) % 1 == 0:
                print('Train epoch: {}/{} Training Loss: {:.3f} Trainning Accuracy: {:.2f}% Validation Loss: {:.3f} Validation Accuracy: {:.2f}%'.format(epoch+1, n_epochs, train_loss, acc, valid_loss ,val_acc))
            acc_list.append(acc)
            train_loss_list.append(train_loss)
            val_acc_list.append(val_acc)
            val_loss_list.append(valid_loss)
            val_acc_max = self.save_model(val_acc_max , acc , val_acc , epoch , save_dir)
            torch.cuda.empty_cache() 
        t2 = time.time()
        return acc_list , train_loss_list , val_acc_list , val_loss_list, t2-t1
    
    def model_test(self):
        #test_path = 'all'
        test_path = os.path.join(self.Data_Folder_name,'test_set')
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
        model,_ = loading_model(self.model_name , len(self.class_dict))
        model.load_state_dict(torch.load(model_folder + '/' + 'Ultimately_the_best_model({}).pth'.format(self.model_name)))
        model.to(self.device)
        model.eval()
        prdict_list = torch.zeros(0,dtype=torch.long , device = 'cpu')
        label_list = torch.zeros(0,dtype=torch.long , device = 'cpu')
        for i,t in data:
            #image = Image.open(i)
            img = cv2.imread(i)
            img1 = cv2.resize(img, (self.image_size,self.image_size))
            if 'Task01' in i: 
            #imgumantation
                img1 = seq.augment_image(img1) 
                img1 = img1.copy()
            else:
                img1 = img1
            img1 = Image.fromarray(cv2.cvtColor(img1,cv2.COLOR_BGR2RGB))
            img1 = self.ImageDatatransforms1(img1)
            img1 = img1.unsqueeze(0).to(self.device)
            output = model(img1)
            _, pred = torch.max(output.data, 1)
            if pred.cpu()!=int(self.class_dict[t]):
                path = os.path.join(mis_path,t+' to '+self.idx_to_class[pred.cpu().item()])
                if not os.path.isdir(path):
                    os.makedirs(path)
                cv2.imwrite(os.path.join(path,os.path.split(i)[1]),img)
            prdict_list = torch.cat([prdict_list,pred.view(-1).cpu()])
            label_list = torch.cat([label_list,torch.tensor(int(self.class_dict[t])).view(-1)])
        test_acc = len(prdict_list.numpy()[prdict_list.numpy()==label_list.numpy()])/len(prdict_list.numpy())
        c_matrix = confusion_matrix(label_list.numpy(), prdict_list.numpy())
        c_matrix = pd.DataFrame(c_matrix, index = self.classes, columns = self.classes)
        return test_acc,c_matrix
    
def show_history(tain , val, model_name):
    epochs = len(tain)
    epoch = [x for x in range(epochs)]
    plt.plot(epoch ,tain , color = 'blue' , label = 'train_acc')
    plt.plot(epoch ,val , color = 'green' , label = 'val_acc')
    plt.legend()
    plt.title('Train history of '+ model_name )
    plt.ylabel('acc')
    plt.xlabel('Epoch')
    plt.savefig("train_history({}).png".format(model_name))
    plt.show()

#%%
if __name__ == "__main__":
    
    Folder_path = 'DATA_binary'
    model_folder = 'model'
    if not os.path.isdir(model_folder): 
        os.makedirs(model_folder)
    LR = 0.0001
    batch_size = 32
    num_epochs = 100
    #class_num = len(os.listdir(Folder_path + '/training_set'))
    class_num = 2
    model_name = 'ResNet18'
    file = open("model_information.txt",mode="w",encoding="UTF-8")
    file.write( model_name   + '\n' )
    file.write( str(112) + '\n' )
    file.write( str(class_num)  + '\n' )
    criterion = nn.CrossEntropyLoss()
    Training = Training_AI_model(Folder_path,model_name,class_num,batch_size,criterion,LR) 
    for i in Training.class_dict.keys():
        file.write( i  + '\n' )
    file.close()
    acc_list , train_loss_list , val_acc_list , val_loss_list, training_time = Training.Training_model(num_epochs ,model_folder)
    show_history(acc_list , val_acc_list ,model_name)
    test_acc,c_matrix = Training.model_test()


    
    
    
    
    
    
    
    
    

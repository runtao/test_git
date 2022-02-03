import torch
import torch.nn as nn
import torch.optim as optim 
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import dataset, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from config import Config
from models import resnet18



data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transfroms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])
        }


image_datasets = {x: datasets.ImageFolder(os.path.join(Config.data_dir, x), # ImageFolder 作为数据集加载接口
                    data_transforms[x]) for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x],batch_size=Config.batch_size,shuffle=True,pin_memory=True,
                        num_workers=0) for x in ['train','val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train','val']}
class_names = image_datasets['train'].classes


def train_model(model, device, criterion, optimizer, scheduler, num_epochs=25):
    
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict()) # 初始化最好的模型
    best_acc = 0.0

    for epoch in range(num_epoch):
        print("Epoch {}/{}".format(epoch+1, num_epochs))
        print("-"*10)

        # each epoch has a training and validation phase
        for phase in ['train','val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            # iterate over data
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # zero the parameter gradients
                if phase == "train":
                    optimizer.zero_grad()

                # forward 
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, pred = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)


                    # backward + optimizer only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item()*inputs.size()[0]
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print("{} Loss: {:.5f} Acc: {:.5f}".format(phase, epoch_loss, epoch_acc))

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

            # 每隔2个epoch保存一次最好的模型
            # if (epoch+1) % 2 == 0:
            #     torch.save(best_model_wts, Config.save_model_dir + 'epochs{}_lr{}_bs{}_stepsize{}_gamma{}.pt'.format(Config.num_epochs,
            #                 Config.learning_rate,Config.batch_size,Config.step_size,Config.gamma))
            

        print()

    time_elapsed = time.time() - since
    print("Training complete in {:.0f}m {:.0f}s".format(time_elapsed//60,time_elapsed % 60))
    print("Best val Acc: {:.5f}".format(best_acc))
    torch.save(best_model_wts, Config.save_model_dir + 'epochs{}_lr{}_bs{}_stepsize{}_gamma{}.pt'.format(Config.num_epochs,
                                Config.learning_rate,Config.batch_size,Config.step_size,Config.gamma))


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    model = resnet18(Config.num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.SGD(model_ft.parameters(),lr=Config.learning_rate,momentum=0.9)

    # decay LR by a factor of 0.1 every 7 epochs
    ep_lr_scheduler = lr_scheduler.StepLR(optimizer_ft,step_size=Config.step_size,gamma=Config.gamma)

    train_model(model, device, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=Config.num_epochs)

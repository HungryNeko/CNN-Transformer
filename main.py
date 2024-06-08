# 训练模型
import copy
from train import model,criterion,optimizer
import torch
from train import train_model
from datapre import dataloaders,dataset_sizes,device

if __name__ == '__main__':

    model = train_model(model, dataloaders, dataset_sizes, criterion, optimizer, device, num_epochs=200,save_path='./pth')
    #model, dataloaders, dataset_sizes, criterion, optimizer, device, num_epochs=25, save_path=None

    # 保存模型
    torch.save(model.state_dict(), 'cat_dog_model.pth')


import copy

import torch
from torch import nn, optim
from model import SwinTransformer
from tqdm import tqdm
from datapre import size
from datapre import class_names, device, dataloaders, dataset_sizes
from model import SimpleCNN, CombinedModel

cnn_model = SimpleCNN()
transformer_model = SwinTransformer(num_classes=2)
# 初始化模型
model = CombinedModel(cnn_model, transformer_model)
# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


# 实例化组合模型


# 训练函数略有修改
def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, device, num_epochs=25, save_path=None):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    model = model.to(device)
    model_save_path=f"{save_path}/model.pth"
    try:
        model.load_state_dict(torch.load(model_save_path))
        print(f'loaded {model_save_path}')
    except Exception as e:
        print(e)
        pass
    print(f'training on {device}')
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            dataloader = tqdm(dataloaders[phase], desc=f'{phase} Epoch {epoch + 1}')

            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    # 使用 CNN 提取特征，不需要展平
                    cnn_features = model.cnn.features(inputs)

                    # 使用 Transformer 进行分类
                    outputs = model.transformer(cnn_features)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                dataloader.set_postfix(loss=loss.item())

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        if save_path:
            torch.save(model.state_dict(), model_save_path)

    print(f'Best test Acc: {best_acc:.4f}')
    model.load_state_dict(best_model_wts)
    return model
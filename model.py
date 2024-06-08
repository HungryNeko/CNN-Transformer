import torch
from torch import nn
from torchvision.models import swin_transformer


# 定义CNN模型
# 定义CNN模型
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 3, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        x = self.features(x)
        return x


# 定义一个Swin Transformer模型
class SwinTransformer(nn.Module):
    def __init__(self, num_classes):
        super(SwinTransformer, self).__init__()
        # 根据需要修改模型参数
        patch_size = [4, 4]
        embed_dim = 96
        depths = [2, 2, 6, 2]
        num_heads = [3, 6, 12, 24]
        window_size = [7,7]

        self.model = swin_transformer.SwinTransformer(
            patch_size=patch_size,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size
        )

        # 修改Swin Transformer的头部，适应分类任务
        self.model.head = nn.Linear(self.model.head.in_features, num_classes)

    def forward(self, x):
        return self.model(x)


class CombinedModel(nn.Module):
    def __init__(self, cnn_model, transformer_model):
        super(CombinedModel, self).__init__()
        self.cnn = cnn_model
        self.transformer = transformer_model

    def forward(self, x):
        # 使用 CNN 提取特征
        cnn_features = self.cnn(x)
        print(cnn_features.size)
        # 使用 Transformer 进行分类
        transformer_output = self.transformer(cnn_features)
        return transformer_output
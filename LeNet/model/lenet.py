import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np


class LeNet(nn.Module):
    """
    LeNet卷积神经网络
    
    架构:
    - 输入: 1x28x28 (MNIST图像)
    - C1: 6个5x5卷积核 -> 6x28x28
    - S2: 2x2最大池化 -> 6x14x14
    - C3: 16个5x5卷积核 -> 16x10x10
    - S4: 2x2最大池化 -> 16x5x5
    - 展平 -> 16*5*5
    - F5: 400 -> 120
    - F6: 120 -> 84
    - 输出: 10个神经元 (数字0-9)
    """
    
    def __init__(self, num_classes=10):
        """
        初始化LeNet模型
        
        Args:
            num_classes (int): 分类数量，MNIST为10
        """
        super(LeNet, self).__init__()
        
        # 卷积层
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, padding=0)  
        
        # 全连接层
        self.fc1 = nn.Linear(16*5*5, 84)
        self.fc2 = nn.Linear(84, num_classes)

        # 原lenet使用平均池化
        #池化层
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 原lenet没有dropout层
        # Dropout层（防止过拟合）
        self.dropout = nn.Dropout(0.5)

        #展平层
        self.flatten = nn.Flatten()
        # 原lenet使用sigmoid作为激活函数
        # 激活函数
        self.relu = nn.ReLU()

        self.net = nn.Sequential(
            self.conv1,
            self.relu,  # 卷积层后添加激活函数
            self.pool1,
            self.conv2,
            self.relu,  # 卷积层后添加激活函数
            self.pool2,
            self.flatten,
            self.fc1,
            self.relu,  # 全连接层后添加激活函数
            self.dropout,
            self.fc2
            # 注意：输出层前不添加ReLU，因为后续通常会使用softmax或交叉熵损失函数
        )
        
    def forward(self, x):
        """
        前向传播
        
        Args:
            x (torch.Tensor): 输入张量，形状为 (batch_size, 1, 28, 28)
            
        Returns:
            torch.Tensor: 输出张量，形状为 (batch_size, num_classes)
        """
        return self.net(x)
        
def test_model():
    net = LeNet()
    X = torch.randn(1, 1, 28, 28)
    for layer in net.net:
        X = layer(X)
        print(layer.__class__.__name__, 'output shape:\t', X.shape)



if __name__ == "__main__":
    test_model() 
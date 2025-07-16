import torch
import torch.nn as nn
import torch.nn.functional as F


class AlexNet(nn.Module):
    """
    AlexNet卷积神经网络
    
    原始AlexNet架构:
    - 输入: 1x224x224 (灰度图像)
    - Conv1: 96个11x11卷积核，步长4 -> 96x54x54
    - MaxPool1: 3x3，步长2 -> 96x26x26
    - Conv2: 256个5x5卷积核 -> 256x26x26
    - MaxPool2: 3x3，步长2 -> 256x12x12 
    - Conv3: 384个3x3卷积核 -> 384x12x12
    - Conv4: 384个3x3卷积核 -> 384x12x12
    - Conv5: 256个3x3卷积核 -> 256x12x12
    - MaxPool3: 3x3，步长2 -> 256x5x5
    - 展平 -> 256*5*5 = 6400
    - FC1: 6400 -> 4096
    - FC2: 4096 -> 4096
    - FC3: 4096 -> 10 (Fashion-MNIST类别数)
    """
    
    def __init__(self, num_classes=10):
        """
        初始化AlexNet模型
        
        Args:
            num_classes (int): 分类数量，Fashion-MNIST为10
        """
        super(AlexNet, self).__init__()
        
        # 特征提取部分
        self.features = nn.Sequential(
            # 第一层卷积
            nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            # 第二层卷积
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            # 第三层卷积
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            # 第四层卷积
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            # 第五层卷积
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        
        # 分类器部分
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(6400, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
        
        # 初始化权重
        self._initialize_weights()
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x (torch.Tensor): 输入张量，形状为 (batch_size, 3, 224, 224)
            
        Returns:
            torch.Tensor: 输出张量，形状为 (batch_size, num_classes)
        """
        x = self.features(x)
        x = torch.flatten(x, 1)  # 展平除批次维度外的所有维度
        x = self.classifier(x)
        return x
    
    def _initialize_weights(self):
        """
        初始化模型权重
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def test_model():
    """
    测试AlexNet模型
    """
    net = AlexNet()
    X = torch.randn(1, 3, 224, 224)  # 批次大小为1，3通道，224x224
    
    # 逐层测试
    x = X
    for i, layer in enumerate(net.features):
        x = layer(x)
        print(f"特征层 {i}, {layer.__class__.__name__}, 输出形状: {x.shape}")
    
    x = torch.flatten(x, 1)
    print(f"展平后形状: {x.shape}")
    
    for i, layer in enumerate(net.classifier):
        x = layer(x)
        print(f"分类器层 {i}, {layer.__class__.__name__}, 输出形状: {x.shape}")
    
    # 整体测试
    output = net(X)
    print(f"\n整体输出形状: {output.shape}")


if __name__ == "__main__":
    test_model() 
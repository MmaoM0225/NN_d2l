import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os

# 设置中文字体支持
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False


class FashionMNISTDataLoader:
    """Fashion-MNIST数据集加载器"""
    
    def __init__(self, batch_size=64, data_dir='../data', num_workers=0):
        """
        初始化数据加载器
        
        Args:
            batch_size (int): 批次大小
            data_dir (str): 数据存储目录
            num_workers (int): 工作进程数
        """
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.num_workers = num_workers
        
        # 确保数据目录存在
        os.makedirs(data_dir, exist_ok=True)
        
        # 定义数据预处理 - 调整图像大小
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # 调整大小为较小的尺寸
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))  # 标准化
        ])
        
        # 类别名称
        self.classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                       'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
        
        # 加载训练和测试数据
        self.train_loader = None
        self.test_loader = None
        self._load_data()
    
    def _load_data(self):
        """加载Fashion-MNIST数据集"""
        # 训练数据
        train_dataset = torchvision.datasets.FashionMNIST(
            root=self.data_dir,
            train=True,
            download=True,
            transform=self.transform
        )
        
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )
        
        # 测试数据
        test_dataset = torchvision.datasets.FashionMNIST(
            root=self.data_dir,
            train=False,
            download=True,
            transform=self.transform
        )
        
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )
        
        print(f"训练集大小: {len(train_dataset)}")
        print(f"测试集大小: {len(test_dataset)}")
        print(f"训练批次数量: {len(self.train_loader)}")
        print(f"测试批次数量: {len(self.test_loader)}")
    
    def get_train_loader(self):
        """获取训练数据加载器"""
        return self.train_loader
    
    def get_test_loader(self):
        """获取测试数据加载器"""
        return self.test_loader
    
    def visualize_samples(self, num_samples=8):
        """
        可视化一些样本数据
        
        Args:
            num_samples (int): 要显示的样本数量
        """
        # 获取原始数据用于可视化
        vis_transform = transforms.Compose([
            transforms.ToTensor()  # 不调整大小，便于可视化
        ])
        
        vis_dataset = torchvision.datasets.FashionMNIST(
            root=self.data_dir,
            train=True,
            download=True,
            transform=vis_transform
        )
        
        # 随机选择样本
        indices = torch.randperm(len(vis_dataset))[:num_samples]
        
        # 创建子图
        fig, axes = plt.subplots(2, 4, figsize=(12, 6))
        axes = axes.ravel()
        
        for i, idx in enumerate(indices):
            img, label = vis_dataset[idx]
            img = img.squeeze().numpy()
            
            axes[i].imshow(img, cmap='gray')
            axes[i].set_title(f'标签: {self.classes[label]}')
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def get_data_info(self):
        """获取数据集信息"""
        data_iter = iter(self.train_loader)
        images, labels = next(data_iter)
        
        print("数据集信息:")
        print(f"图像形状: {images.shape}")
        print(f"标签形状: {labels.shape}")
        print(f"图像数据类型: {images.dtype}")
        print(f"标签数据类型: {labels.dtype}")
        print(f"图像值范围: [{images.min():.3f}, {images.max():.3f}]")
        print(f"标签值范围: [{labels.min()}, {labels.max()}]")


if __name__ == "__main__":
    # 测试数据加载器
    loader = FashionMNISTDataLoader(batch_size=32)
    loader.get_data_info()
    loader.visualize_samples() 
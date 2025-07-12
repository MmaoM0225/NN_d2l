import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import os

# 导入我们的模块
from dataloader.mnist_loader import MNISTDataLoader
from model.lenet import LeNet
from utils.visualization import plot_training_history, visualize_predictions


class Trainer:
    """训练器类"""
    
    def __init__(self, model, train_loader, test_loader, device='cuda'):
        """
        初始化训练器
        
        Args:
            model: 要训练的模型
            train_loader: 训练数据加载器
            test_loader: 测试数据加载器
            device: 训练设备
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        
        # 损失函数和优化器
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.9)
        
        # 训练历史
        self.train_losses = []
        self.train_accs = []
        self.test_losses = []
        self.test_accs = []
        
        # 最佳模型
        self.best_acc = 0.0
        self.best_model_path = os.path.join(os.path.dirname(__file__), 'checkpoints', 'best_lenet.pth')
        
        # 创建保存目录
        os.makedirs(os.path.dirname(self.best_model_path), exist_ok=True)
    
    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        # 使用tqdm显示进度条
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}')
        
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            
            # 反向传播
            loss.backward()
            self.optimizer.step()
            
            # 统计
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            # 更新进度条
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
        
        # 计算平均损失和准确率
        avg_loss = total_loss / len(self.train_loader)
        avg_acc = 100. * correct / total
        
        return avg_loss, avg_acc
    
    def test_epoch(self):
        """测试一个epoch"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        avg_loss = total_loss / len(self.test_loader)
        avg_acc = 100. * correct / total
        
        return avg_loss, avg_acc
    
    def save_best_model(self, acc):
        """保存最佳模型"""
        if acc > self.best_acc:
            self.best_acc = acc
            torch.save({
                'epoch': len(self.train_losses),
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'best_acc': self.best_acc,
                'train_losses': self.train_losses,
                'train_accs': self.train_accs,
                'test_losses': self.test_losses,
                'test_accs': self.test_accs
            }, self.best_model_path)
            print(f'保存最佳模型，准确率: {acc:.2f}%')
    
    def train(self, epochs=10):
        """完整训练过程"""
        print(f"开始训练，设备: {self.device}")
        print(f"训练集大小: {len(self.train_loader.dataset)}")
        print(f"测试集大小: {len(self.test_loader.dataset)}")
        print(f"模型参数数量: {sum(p.numel() for p in self.model.parameters()):,}")
        print("=" * 50)
        
        start_time = time.time()
        
        for epoch in range(epochs):
            # 训练
            train_loss, train_acc = self.train_epoch(epoch)
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)
            
            # 测试
            test_loss, test_acc = self.test_epoch()
            self.test_losses.append(test_loss)
            self.test_accs.append(test_acc)
            
            # 学习率调度
            self.scheduler.step()
            
            # 打印结果
            print(f'Epoch {epoch+1}/{epochs}:')
            print(f'  训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.2f}%')
            print(f'  测试损失: {test_loss:.4f}, 测试准确率: {test_acc:.2f}%')
            print(f'  学习率: {self.optimizer.param_groups[0]["lr"]:.6f}')
            print('-' * 30)
            
            # 保存最佳模型
            self.save_best_model(test_acc)
        
        # 训练完成
        total_time = time.time() - start_time
        print(f"训练完成！总用时: {total_time/60:.2f}分钟")
        print(f"最佳测试准确率: {self.best_acc:.2f}%")
        
        # 绘制训练历史
        self.plot_training_history()
    
    def plot_training_history(self):
        """绘制训练历史"""
        plot_training_history(
            self.train_losses, 
            self.train_accs,
            self.test_losses,
            self.test_accs,
            save_path=os.path.join(os.path.dirname(__file__), 'training_history.png')
        )
    
    def visualize_predictions(self, num_samples=8):
        """可视化预测结果"""
        visualize_predictions(
            self.model, 
            self.test_loader, 
            num_samples, 
            self.device
        )


def main():
    """主函数"""
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载数据
    print("加载数据...")
    data_loader = MNISTDataLoader(batch_size=64, num_workers=4)
    train_loader = data_loader.get_train_loader()
    test_loader = data_loader.get_test_loader()
    
    # 创建模型
    print("创建模型...")
    model = LeNet(num_classes=10)
    
    # 创建训练器
    trainer = Trainer(model, train_loader, test_loader, device)
    
    # 开始训练
    trainer.train(epochs=10)
    
    # 可视化预测结果
    print("可视化预测结果...")
    trainer.visualize_predictions()


if __name__ == "__main__":
    main() 
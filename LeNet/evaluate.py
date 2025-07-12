import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import os

# 导入我们的模块
from dataloader.mnist_loader import MNISTDataLoader
from model.lenet import LeNet
from utils.visualization import plot_confusion_matrix


class Evaluator:
    """评估器类"""
    
    def __init__(self, model, test_loader, device='cuda'):
        """
        初始化评估器
        
        Args:
            model: 要评估的模型
            test_loader: 测试数据加载器
            device: 评估设备
        """
        self.model = model.to(device)
        self.test_loader = test_loader
        self.device = device
        
        # 损失函数
        self.criterion = nn.CrossEntropyLoss()
    
    def evaluate(self):
        """完整评估过程"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        # 存储所有预测和真实标签
        all_predictions = []
        all_targets = []
        
        print("开始评估...")
        
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
                
                # 收集预测结果
                all_predictions.extend(pred.cpu().numpy().flatten())
                all_targets.extend(target.cpu().numpy())
        
        # 计算指标
        avg_loss = total_loss / len(self.test_loader)
        accuracy = 100. * correct / total
        
        print(f"评估结果:")
        print(f"  平均损失: {avg_loss:.4f}")
        print(f"  准确率: {accuracy:.2f}%")
        print(f"  正确预测: {correct}/{total}")
        
        return avg_loss, accuracy, all_predictions, all_targets
    
    def detailed_analysis(self, predictions, targets):
        """详细分析"""
        print("\n详细分析:")
        print("=" * 50)
        
        # 分类报告
        print("分类报告:")
        print(classification_report(targets, predictions, digits=4))
        
        # 混淆矩阵
        cm = confusion_matrix(targets, predictions)
        print(f"混淆矩阵形状: {cm.shape}")
        
        # 绘制混淆矩阵
        plot_confusion_matrix(cm, list(range(10)), save_path=os.path.join(os.path.dirname(__file__), 'confusion_matrix.png'))
        
        # 每个类别的准确率
        print("\n每个类别的准确率:")
        for i in range(10):
            class_correct = cm[i, i]
            class_total = cm[i, :].sum()
            class_acc = 100. * class_correct / class_total
            print(f"  数字 {i}: {class_acc:.2f}% ({class_correct}/{class_total})")
    
    def load_model(self, model_path):
        """加载训练好的模型"""
        if not os.path.exists(model_path):
            print(f"模型文件不存在: {model_path}")
            return False
        
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"成功加载模型: {model_path}")
            print(f"最佳准确率: {checkpoint['best_acc']:.2f}%")
            return True
        except Exception as e:
            print(f"加载模型失败: {e}")
            return False


def main():
    """主函数"""
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载数据
    print("加载数据...")
    data_loader = MNISTDataLoader(batch_size=64, num_workers=4)
    test_loader = data_loader.get_test_loader()
    
    # 创建模型
    print("创建模型...")
    model = LeNet(num_classes=10)
    
    # 创建评估器
    evaluator = Evaluator(model, test_loader, device)
    
    # 尝试加载训练好的模型
    model_path = os.path.join(os.path.dirname(__file__), 'checkpoints', 'best_lenet.pth')
    if evaluator.load_model(model_path):
        # 评估模型
        loss, accuracy, predictions, targets = evaluator.evaluate()
        
        # 详细分析
        evaluator.detailed_analysis(predictions, targets)
        
        print(f"\n评估完成！")
        print(f"最终准确率: {accuracy:.2f}%")
    else:
        print("请先运行训练脚本: python train.py")


if __name__ == "__main__":
    main() 
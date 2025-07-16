import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

from model.alexnet import AlexNet
from dataloader.fashion_mnist_loader import FashionMNISTDataLoader
from utils.visualization import visualize_predictions, plot_confusion_matrix, visualize_feature_maps


def load_model(model_path, device):
    """
    加载训练好的模型
    
    Args:
        model_path (str): 模型文件路径
        device (torch.device): 设备
        
    Returns:
        model: 加载的模型
    """
    # 创建模型
    model = AlexNet(num_classes=10)
    
    # 加载权重
    checkpoint = torch.load(model_path, map_location=device)
    
    # 如果保存的是整个检查点
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"加载检查点，Epoch: {checkpoint.get('epoch', 'N/A')}, 最佳准确率: {checkpoint.get('best_acc', 'N/A'):.2f}%")
    else:
        model.load_state_dict(checkpoint)
        print("加载模型权重")
    
    model = model.to(device)
    return model


def evaluate_model(model, test_loader, device):
    """
    评估模型性能
    
    Args:
        model: 训练好的模型
        test_loader: 测试数据加载器
        device: 设备
        
    Returns:
        tuple: (准确率, 预测标签, 真实标签)
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    
    all_preds = []
    all_targets = []
    correct = 0
    total = 0
    test_loss = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            # 计算损失
            loss = criterion(output, target)
            test_loss += loss.item()
            
            # 获取预测结果
            _, pred = torch.max(output, 1)
            
            # 统计准确率
            total += target.size(0)
            correct += (pred == target).sum().item()
            
            # 收集预测和真实标签
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    # 计算准确率和平均损失
    accuracy = 100 * correct / total
    avg_loss = test_loss / len(test_loader)
    
    print(f"测试损失: {avg_loss:.4f}")
    print(f"测试准确率: {accuracy:.2f}%")
    
    return accuracy, np.array(all_preds), np.array(all_targets)


def main():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载数据
    print("加载测试数据...")
    data_loader = FashionMNISTDataLoader(batch_size=64, num_workers=0)
    test_loader = data_loader.get_test_loader()
    
    # 类别名称
    class_names = data_loader.classes
    
    # 加载模型
    model_path = os.path.join(os.path.dirname(__file__), 'checkpoints', 'best_alexnet.pth')
    if not os.path.exists(model_path):
        print(f"错误: 找不到模型文件 '{model_path}'")
        print("请先训练模型或检查文件路径")
        return
    
    print(f"加载模型: {model_path}")
    model = load_model(model_path, device)
    
    # 评估模型
    print("评估模型...")
    accuracy, all_preds, all_targets = evaluate_model(model, test_loader, device)
    
    # 打印分类报告
    print("\n分类报告:")
    print(classification_report(all_targets, all_preds, target_names=class_names))
    
    # 计算混淆矩阵
    cm = confusion_matrix(all_targets, all_preds)
    
    # 绘制混淆矩阵
    print("绘制混淆矩阵...")
    plot_confusion_matrix(cm, class_names)
    
    # 可视化预测结果
    print("可视化预测结果...")
    visualize_predictions(model, test_loader, num_samples=8, device=device, class_names=class_names)
    
    # 可视化特征图
    print("可视化特征图...")
    # 获取一个样本
    data_iter = iter(test_loader)
    images, _ = next(data_iter)
    sample_image = images[0].to(device)
    
    # 可视化第一个卷积层的特征图
    print("第一个卷积层的特征图:")
    visualize_feature_maps(model, sample_image, layer_idx=0, num_maps=16)
    
    # 可视化第三个卷积层的特征图
    print("第三个卷积层的特征图:")
    visualize_feature_maps(model, sample_image, layer_idx=6, num_maps=16)


if __name__ == "__main__":
    main() 
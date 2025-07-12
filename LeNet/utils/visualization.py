import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import torch


def setup_chinese_font():
    """设置中文字体支持"""
    matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    matplotlib.rcParams['axes.unicode_minus'] = False


def plot_training_history(train_losses, train_accs, val_losses=None, val_accs=None, save_path=None):
    """
    绘制训练历史
    
    Args:
        train_losses (list): 训练损失
        train_accs (list): 训练准确率
        val_losses (list, optional): 验证损失
        val_accs (list, optional): 验证准确率
        save_path (str, optional): 保存路径
    """
    setup_chinese_font()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # 绘制损失
    ax1.plot(train_losses, label='训练损失', color='blue')
    if val_losses:
        ax1.plot(val_losses, label='验证损失', color='red')
    ax1.set_xlabel('轮次')
    ax1.set_ylabel('损失')
    ax1.set_title('训练损失曲线')
    ax1.legend()
    ax1.grid(True)
    
    # 绘制准确率
    ax2.plot(train_accs, label='训练准确率', color='blue')
    if val_accs:
        ax2.plot(val_accs, label='验证准确率', color='red')
    ax2.set_xlabel('轮次')
    ax2.set_ylabel('准确率')
    ax2.set_title('训练准确率曲线')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def visualize_predictions(model, data_loader, num_samples=8, device='cpu'):
    """
    可视化模型预测结果
    
    Args:
        model: 训练好的模型
        data_loader: 数据加载器
        num_samples (int): 要显示的样本数量
        device (str): 设备类型
    """
    setup_chinese_font()
    
    model.eval()
    data_iter = iter(data_loader)
    images, labels = next(data_iter)
    
    # 获取预测结果
    with torch.no_grad():
        images = images.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
    
    # 创建子图
    fig, axes = plt.subplots(2, 4, figsize=(15, 8))
    axes = axes.ravel()
    
    for i in range(num_samples):
        # 反归一化图像
        img = images[i].cpu().squeeze().numpy()
        img = (img * 0.3081) + 0.1307  # 反归一化
        
        axes[i].imshow(img, cmap='gray')
        
        # 设置标题颜色
        true_label = labels[i].item()
        pred_label = predicted[i].item()
        
        if true_label == pred_label:
            color = 'green'
            title = f'预测: {pred_label} [正确]'
        else:
            color = 'red'
            title = f'预测: {pred_label} [错误] (真实: {true_label})'
        
        axes[i].set_title(title, color=color, fontsize=12)
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(cm, class_names, save_path=None):
    """
    绘制混淆矩阵
    
    Args:
        cm (numpy.ndarray): 混淆矩阵
        class_names (list): 类别名称
        save_path (str, optional): 保存路径
    """
    setup_chinese_font()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    # 设置刻度标签
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names,
           yticklabels=class_names,
           title='混淆矩阵',
           ylabel='真实标签',
           xlabel='预测标签')
    
    # 在格子中显示数字
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_feature_maps(feature_maps, num_maps=16, save_path=None):
    """
    绘制特征图
    
    Args:
        feature_maps (torch.Tensor): 特征图张量
        num_maps (int): 要显示的特征图数量
        save_path (str, optional): 保存路径
    """
    setup_chinese_font()
    
    # 获取第一个样本的特征图
    feature_maps = feature_maps[0].cpu().detach().numpy()
    
    # 限制显示数量
    num_maps = min(num_maps, feature_maps.shape[0])
    
    # 计算子图布局
    cols = 4
    rows = (num_maps + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 3*rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_maps):
        row = i // cols
        col = i % cols
        
        im = axes[row, col].imshow(feature_maps[i], cmap='viridis')
        axes[row, col].set_title(f'特征图 {i+1}')
        axes[row, col].axis('off')
    
    # 隐藏多余的子图
    for i in range(num_maps, rows * cols):
        row = i // cols
        col = i % cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show() 
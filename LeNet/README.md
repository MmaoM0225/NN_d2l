# LeNet 卷积神经网络实现

本项目实现了经典的LeNet-5卷积神经网络架构。LeNet-5是由Yann LeCun等人在1998年提出的用于手写数字识别的卷积神经网络。

## 项目结构

```
LeNet/
├── data/                   # 数据集
│   └── MNIST               #MNIST数据集（下载后生成）
├── dataloader/             # 数据加载器
│   ├── __init__.py
│   └── mnist_loader.py     # MNIST数据集加载器
├── model/                  # 模型相关代码
│   └── lenet.py           # LeNet模型定义
├── utils/                  # 工具函数
│   └── visualization.py    # 可视化工具
├── train.py               # 训练脚本
└── evaluate.py            # 评估脚本
```

## 环境要求
- Python 3.12+
- PyTorch
- torchvision
- numpy
- matplotlib
- tqdm

## 使用说明
1. 运行训练：`python train.py`
2. 评估模型：`python evaluate.py`

## 训练参数
- 训练轮数：10 epochs
- 批次大小：64
- 学习率：0.001
- 优化器：Adam
- 学习率调度：StepLR (每5轮衰减0.1)

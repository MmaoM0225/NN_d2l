# 深度学习网络实现项目

本项目实现了多种经典的深度学习网络架构，用于图像分类任务。

## 项目结构

```
NN_d2l/
├── LeNet/                    # LeNet 卷积神经网络
├── AlexNet/                    # AlexNet 卷积神经网络
├── requirements.txt           # 项目依赖
└── README.md                 # 项目总说明
```

## 已实现的网络

### 1. LeNet
- **任务**：MNIST手写数字识别
- **架构**：经典卷积神经网络（LeNet）
- **预期性能**：98%+ 准确率

### 2. AlexNet
- **任务**：fashion-MNIST物品识别
- **架构**：经典卷积神经网络（AlexNet）
- **预期性能**：90%+ 准确率

## 环境要求

- Python 3.12+
- PyTorch >= 1.9.0
- torchvision >= 0.10.0
- 其他依赖见 `requirements.txt`

## 快速开始

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 训练模型

详见对应模型READ.md文件

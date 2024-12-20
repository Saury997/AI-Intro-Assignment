# AI-Intro-Assignment  
“人工智能导论”课程实验的源代码。

## 项目概述  
该项目包含了“人工智能导论”课程的实验解决方案，涉及将各种人工智能技术应用于实际问题的解决，如图像识别、图像分类和使用遗传算法进行优化。

## 实验内容  

### **实验 1: 手写数字识别**  
- **目标**: 构建一个模型，用于识别手写数字。  
- **使用模型**:  
  - MLP（多层感知器）  
  - CNN（卷积神经网络）  

### **实验 2: 基于CIFAR-100数据集的图像分类**  
- **目标**: 使用GPU训练VGG13模型，将CIFAR-100数据集中的图像分类到100个类别中。  
- **使用模型**:  
  - VGG（视觉几何组网络）  
  - ResNet（残差网络）  
  - DenseNet（密集连接卷积网络）  

### **实验 3：使用U-net进行医学图像分割**

- **目标**: 使用U-net模型对肝脏图像数据集执行分割任务
- **使用模型**：
  - U-net

### **实验 4：图神经网络（GNN）应用于图数据分类 - Cora 数据集**
- **目标**：使用Cora数据集，基于节点的特征和图结构进行分类。
- **使用模型**：
  - GCN
  - GAT
  - SplineGCN
  - ChebNet

### **实验 5: 使用遗传算法（GA）解决旅行商问题（TSP）**  

- **目标**: 使用遗传算法解决TSP优化问题，最小化一组城市的总旅行成本。  
- **关键特性**:  
  - 使用遗传算法解决TSP问题  
  - 实现多种变异、交叉和选择方法  
  - 提供UI界面用于交互式的参数调整和结果可视化  

## 环境要求  
- Python 3.8 或更高版本  
- 依赖：  
  - PyQt5  
  - Matplotlib  
  - Geopandas  
  - Scikit-learn  
  - PyTorch
  - Torch_geometric
    
  详细请看requirements.txt

## 如何运行  
1. 克隆该仓库：  
   ```bash  
   git clone https://github.com/Saury997/AI-Intro-Assignment.git
   
2. 实验代码以exp*.py格式命名

**如果你觉得这个项目有用，请点个star！**

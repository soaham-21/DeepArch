# DeepArch: Neural Architecture Search for CIFAR-10

> Automatically designing and optimizing deep learning architectures using **PyTorch** and **Optuna**.

# Overview

**DeepArch** is an AI-driven system that performs **Neural Architecture Search (NAS)** — a process where the computer automatically discovers the best-performing **Convolutional Neural Network (CNN)** architecture for image classification on the **CIFAR-10 dataset**.

Instead of manually tuning layers, filters, and hyperparameters, DeepArch intelligently explores different architectures and finds the one that gives the best accuracy — saving time and human effort.

# Objectives

- Implement a **Neural Architecture Search (NAS)** pipeline from scratch  
- Define a **flexible CNN search space** (layers, activations, pooling, dropout, etc.)  
- Use **Bayesian optimization (Optuna)** to explore and prune models efficiently  
- Train and evaluate architectures on **CIFAR-10**  
- Select, retrain, and evaluate the **best discovered model**

# Tech Stack

|   Component   |    Tool / Library  |
|---------------|--------------------|
| Language      | Python 3.9+        |
| Deep Learning | PyTorch            | 
| Optimization  | Optuna             |
| Dataset       | CIFAR-10           |
| Visualization | Matplotlib         |
| Utilities     | scikit-learn, tqdm |

# Setup Instructions

# Clone the repository
```bash
git clone https://github.com/soaham-21/DeepArch.git
cd DeepArch

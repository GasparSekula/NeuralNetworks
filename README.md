# NeuralNetworks
Multilayer Perceptron implemented from scratch. 

This project focuses on implementing a **Multilayer Perceptron (MLP)** from scratch using **NumPy**, without relying on high-level libraries like TensorFlow or PyTorch.
---

## 🎯 Project Overview

This laboratory project dives deep into the foundations of neural networks, with an emphasis on the MLP model, one of the core building blocks of deep learning. The primary goal was not only to understand how MLPs work internally but also to:

- 🛠️ Build a functional neural network from scratch.
- 🔬 Explore and analyze the influence of hyperparameters on learning.
- 📊 Validate theoretical knowledge through hands-on experiments and visualizations.

As part of the lab course, we proceed through a step-by-step development*of an MLP — from forward propagation to training, evaluation, and visualization — all implemented using **only NumPy**.

---

## 🧪 Laboratory Progress & Notebook Summary

Each Jupyter Notebook (`NN1.ipynb` → `NN6.ipynb`) reflects a major milestone in the development process.

| Notebook | Topic | Highlights |
|----------|-------|------------|
| `NN1.ipynb` | 🔢 Manual Weights & Forward Pass | Initial experiments with **manually chosen weights** and simple forward function logic |
| `NN2.ipynb` | 🔁 Backpropagation & Training | Implemented **backpropagation**, training using **mini-batches** vs **full dataset** |
| `NN3.ipynb` | ⚡ Optimization Techniques | Compared **RMSProp** and **Momentum** optimizers |
| `NN4.ipynb` | 🧩 Classification Tasks | Switched to **classification** problems and adjusted loss functions accordingly |
| `NN5.ipynb` | 🏗️ Architecture Design | Experiments on **number of layers**, **neurons**, and **activation functions** |
| `NN6.ipynb` | 🛡️ Regularization | Compared techniques like **L2 regularization**, **Dropout**, and **Early Stopping** |

> ⚠️ **Note:** Some notebooks reflect evolving ideas and may contain legacy code from earlier stages.

---

## 📁 Repository Structure
NeuralNetworks/ │ ├── data/ # Datasets used for training/testing │ ├── metrics/ # Evaluation metrics │ ├── init.py │ └── metrics.py │ ├── network/ # Core neural network components │ ├── activations.py │ ├── layers.py │ ├── losses.py │ ├── mlp.py │ ├── preprocessing.py │ └── regularization.py │ ├── plots/ # (Optional) Plotting utilities │ ├── visualization/ # Analysis & visualization helpers │ ├── analysis.py │ └── visualization.py │ ├── NN1.ipynb → NN6.ipynb # Jupyter Notebooks for each lab stage ├── report.ipynb # Final report or analysis summary └── README.md # Project documentation (you are here!)

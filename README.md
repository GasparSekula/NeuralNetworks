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

## 📁 Repository Structure & Description

- **`data/`**  
  Contains datasets used for training and evaluation.

- **`metrics/`**  
  Includes evaluation metric logic. Used in former implementation.
  - `metrics.py`: Implements performance metrics such as MSE and Cross Entropy.

- **`network/`**  
  Core components for building and training the MLP model.
  - `activations.py`: Contains activation functions like ReLU, Sigmoid, Tanh.
  - `layers.py`: Defines neural network layers.
  - `losses.py`: Implements various loss functions.
  - `mlp.py`: Main class for the MLP model.
  - `preprocessing.py`: Preprocessing utilities for dataset handling like Standard Scaler and One Hot Encoding.
  - `regularization.py`: Techniques like L1 and L2 regularization.

- **`plots/`**  
  Placeholder for example visualization.

- **`visualization/`**  
  Tools for analysis and plotting.
  - `analysis.py`: Code for analyzing model results. Examples in NNx.ipynb.
  - `visualization.py`: Code for visualizing training progress, weights, etc.

- **`NN1.ipynb`**  
  Manual weight initialization and forward pass demonstrations.

- **`NN2.ipynb`**  
  Implementation of backpropagation; comparison of mini-batch vs full-data training.

- **`NN3.ipynb`**  
  Experiment comparing RMSProp and Momentum optimization strategies.

- **`NN4.ipynb`**  
  Shift to classification tasks and adjustments to loss/activation functions.

- **`NN5.ipynb`**  
  Comparison of architectures with varying layers, neurons, and activations.

- **`NN6.ipynb`**  
  Regularization experiments: L1, L2, and early stopping.

- **`report.ipynb`**  
  Draft notebook for report.

- **`report.pdf`**
  Final report including experiments, results analysis and comparison.

- **`README.md`**  
  This documentation file.


---

## 📌 Key Takeaways

- ✅ **Implemented from scratch**: All components such as layers, losses, activations, and training loops were developed manually.
- 🔍 **Empirical Analysis**: Rich set of experiments to explore how various parameters affect learning.
- 📈 **Visualization & Reporting**: Comprehensive visual tools to interpret network performance.

---

## 🧾 Final Notes

This project is a **learning-focused implementation** — designed to understand how MLPs work under the hood. Feel free to explore, experiment, and expand! 🚀

---

> Created as part of an academic laboratory course on Computational Intelligence Methods in Data Analysis (Metody Inteligencji Obliczeniowej w Analizie Danych), a part of Data Science Bachelor Engineering programme at Warsaw University of Technology.

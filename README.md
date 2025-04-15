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
```NeuralNetworks/ │ ├── data/ # Datasets used for training/testing │ ├── metrics/ # Evaluation metrics │ ├── __init__.py │ └── metrics.py │ ├── network/ # Core neural network components │ ├── __init__.py │ ├── activations.py │ ├── layers.py │ ├── losses.py │ ├── mlp.py │ ├── preprocessing.py │ └── regularization.py │ ├── plots/ # (Optional) Plotting utilities │ ├── visualization/ # Analysis & visualization helpers │ ├── __init__.py │ ├── analysis.py │ └── visualization.py │ ├── NN1.ipynb # Manual weights choice, forward function examples ├── NN2.ipynb # Backpropagation, mini-batch vs full data training ├── NN3.ipynb # RMSProp vs Momentum comparison ├── NN4.ipynb # Classification tasks ├── NN5.ipynb # Architecture experiments (layers, activations, etc.) ├── NN6.ipynb # Regularization techniques comparison │ ├── report.ipynb # Final report or analysis summary └── README.md # Project documentation (you are here!)```

---

## 📌 Key Takeaways

- ✅ **Implemented from scratch**: All components such as layers, losses, activations, and training loops were developed manually.
- 🔍 **Empirical Analysis**: Rich set of experiments to explore how various parameters affect learning.
- 📈 **Visualization & Reporting**: Comprehensive visual tools to interpret network performance.

---

## 🧾 Final Notes

This project is a **learning-focused implementation** — designed to solidify your understanding of how MLPs work under the hood. Feel free to explore, experiment, and expand! 🚀

---

> Created as part of an academic laboratory course on Neural Networks.

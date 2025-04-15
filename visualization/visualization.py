import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import imageio
import time

from network.mlp import MLP
from network.losses import LossFunction
from network.activations import Softmax
from network.preprocessing import classes_to_one_hot, StandardScaler


def plot_weights(network: MLP, filename: str, plot_every_k_epochs: int = 1, fps: int = 10) -> None:
    """
    Plot the weights of the network for each layer over the course of training.
    Args:
        network (MLP): The trained network.
        filename (str): The filename to save the plot.
        plot_every_k_epochs (int): The interval at which to plot the weights.
    """
    
    weights = network.get_training_history()["weights"]
    epochs = len(weights)
    layers = len(weights[0])
    
    y_min = float("inf")
    y_max = -float("inf")
    
    for epoch in range(epochs):
        for layer in range(layers):
            if weights[epoch][layer].min() < y_min:
                y_min = weights[epoch][layer].min()
            if weights[epoch][layer].max() > y_max:
                y_max = weights[epoch][layer].max()
                
    y_min *= 1.05
    y_max *= 1.05
    images = []
            
    fig, axs = plt.subplots(layers, figsize=(10, 4*layers))
    for i in range(0, epochs, plot_every_k_epochs):
        for j in range(layers):
            plt.cla()
            axs[j].clear()
            axs[j].set(xlabel='Weigths', ylabel='Value',title='Weights of layer '+str(layer)+' in epoch '+str(epoch))
            axs[j].bar(np.arange(len(weights[i][j].flatten())), weights[i][j].flatten())
            axs[j].set_ylim(y_min, y_max)
            
        fig.canvas.draw()       
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        images.append(image)
    images += [images[-1]] * fps 
    imageio.mimsave(filename,images, fps=fps)
    
    plt.close(fig)
    

def plot_dataset_classes(train, test):
    fig, (ax_1, ax_2) = plt.subplots(1, 2,figsize=(10, 5))
    sns.scatterplot(x = train[["x"]].to_numpy()[:,0], y = train[["y"]].to_numpy()[:,0], 
                    hue = train[["c"]].to_numpy()[:,0], palette="viridis", ax = ax_1)
    ax_1.set_title("Training dataset")
    sns.scatterplot(x = test[["x"]].to_numpy()[:,0], y = test[["y"]].to_numpy()[:,0], 
                    hue = test[["c"]].to_numpy()[:,0], palette="viridis", ax = ax_2)
    ax_2.set_title("Test dataset")
    plt.show()
    
    
def plot_predicted_classes(test, pred_classes, true_classes):
    x_test = test[["x", "y"]].to_numpy()

    good_pred = (pred_classes == true_classes.flatten()).reshape(-1, 1)
    
    x = x_test[:, 0]
    y = x_test[:, 1]

    true_classes = true_classes.flatten()
        
    fig, (ax_1, ax_2, ax_3) = plt.subplots(1, 3,figsize=(12, 4))
    sns.scatterplot(x=x, y=y, hue=true_classes, palette="viridis", ax=ax_1)
    ax_1.set_title("True classes")
    sns.scatterplot(x=x, y=y, hue=pred_classes, palette="viridis", ax=ax_2)
    ax_2.set_title("Predicted classes")
    sns.scatterplot(x=x, y=y, hue=good_pred.flatten(), palette="hls", ax=ax_3)
    ax_3.set_title("Good vs. bad predictions")
    plt.show()

def run_classification(train: pd.DataFrame,
                       test: pd.DataFrame,
                       layers_init: list,
                       loss_function: LossFunction,
                       epochs: int,
                       learning_rate: float,
                       method: str = "sgd",
                       scaling: bool = False,
                       stop_f1: float = None,
                       random_state = 42,
                       batch_size: int = None):
    
    print("Dataset overview.")
    plot_dataset_classes(train, test)
    
    if scaling:    
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(train[["x", "y"]].to_numpy())
        X_test_scaled = scaler.transform(test[["x", "y"]].to_numpy())
    else:
        X_train_scaled = train[["x", "y"]].to_numpy()
        X_test_scaled = test[["x", "y"]].to_numpy()
    
    if isinstance(layers_init[-1]["activation"], Softmax):
        y_train_one_hot = classes_to_one_hot(train[["c"]].to_numpy() * 1)
        y_test_one_hot = classes_to_one_hot(test[["c"]].to_numpy() * 1)
    else:
        y_train_one_hot = train[["c"]].to_numpy() * 1
        y_test_one_hot = test[["c"]].to_numpy() * 1

    nn = MLP(layers_init=layers_init,
        input=X_train_scaled,
        loss_function=loss_function,
        random_state=random_state)

    print("Training network.")
    t1 = time.time()
    nn.train(X_train=X_train_scaled,
            Y_train=y_train_one_hot,
            epochs=epochs,
            learning_rate=learning_rate,
            method=method,
            stop_f1=stop_f1)
    t2 = time.time()

    print("Training results.")
    nn.plot_training()

    pred = nn.forward(X_test_scaled)
    f1 = nn.calculate_f1_macro(y_test_one_hot, pred)
    print(f"F-Measure on TEST dataset: {f1:.4f}")
    
    print("\nResults.")
    true_classes = test[["c"]].to_numpy() * 1
    if isinstance(layers_init[-1]["activation"], Softmax):
        pred_classes = np.argmax(pred, axis=1)
    else:
        pred_classes = np.round(pred).flatten()
        
    plot_predicted_classes(test, pred_classes, true_classes)
    
    number_of_epochs = len(nn.get_training_history()["losses"])
    training_time = t2 - t1
    final_f1_train = nn.get_training_history()["f1_scores"][-1]
    final_loss_train = nn.get_training_history()["losses"][-1]
    
    return {
        "number_of_epochs": number_of_epochs,
        "training_time": training_time,
        "final_f1_train": final_f1_train,
        "f1_test": f1,
        "loss_func": loss_function,
        "final_loss_train": final_loss_train,
    }
    

def plot_loss_over_time(results_df, start_epoch=10):
    plt.figure(figsize=(20, 6))
    for activation in results_df["activation"].unique():
        subset = results_df[results_df["activation"] == activation]
        mean_losses = np.nanmean(subset["training_history_loss"].tolist(), axis=0)
        sns.lineplot(x=range(start_epoch, len(mean_losses)), y=mean_losses[start_epoch:], label=activation)
        plt.fill_between(range(start_epoch, len(mean_losses)), 
                         mean_losses[start_epoch:] - np.std(mean_losses[start_epoch:]), 
                         mean_losses[start_epoch:] + np.std(mean_losses[start_epoch:]), 
                         alpha=0.1)
    plt.title("Loss Over Time (Starting from Epoch 10)")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(title="Activation Function")
    plt.show()
    

def plot_heatmap_training_time(results_df):
    pivot_table = results_df.pivot(index="activation", columns=["numbers_of_hidden_layers", "numbers_of_neurons"], values="training_time")
    plt.figure(figsize=(20, 6))
    sns.heatmap(pivot_table, cmap="coolwarm", annot=True, fmt=".2f")
    plt.title("Training Time Heatmap")
    plt.xlabel("(Layers, Neurons)")
    plt.ylabel("Activation Function")
    plt.show()
    
def plot_heatmap_mse_final_loss(results_df):
    pivot_table = results_df.pivot(index="activation", columns=["numbers_of_hidden_layers", "numbers_of_neurons"], values="final_mse_train")
    plt.figure(figsize=(20, 6))
    sns.heatmap(pivot_table, cmap="coolwarm", annot=True, fmt=".2f")
    plt.title("MSE Train Loss Heatmap")
    plt.xlabel("(Layers, Neurons)")
    plt.ylabel("Activation Function")
    plt.show()
    
def plot_heatmap_mse_test_loss(results_df):
    pivot_table = results_df.pivot(index="activation", columns=["numbers_of_hidden_layers", "numbers_of_neurons"], values="mse_test")
    plt.figure(figsize=(20, 6))
    sns.heatmap(pivot_table, cmap="coolwarm", annot=True, fmt=".2f")
    plt.title("MSE Test Loss Heatmap")
    plt.xlabel("(Layers, Neurons)")
    plt.ylabel("Activation Function")
    plt.show()
    
def plot_predictions(results_df, x_values, true_values):
    total_plots = len(results_df)
    rows = total_plots // 3 + (total_plots % 3 > 0)
    fig, axes = plt.subplots(rows, 3, figsize=(15, rows * 5))
    axes = axes.flatten()
    
    for i, (idx, row) in enumerate(results_df.iterrows()):
        if row["pred"] is not None and not np.isnan(row["pred"]).any():
            ax = axes[i]
            ax.scatter(x_values, true_values, label="True", color="blue")
            ax.scatter(x_values, row["pred"], label="Predicted", color="red")
            ax.set_title(f"{row['activation']}, {row['numbers_of_hidden_layers']}L, {row['numbers_of_neurons']}N")
            ax.set_xlabel("X values")
            ax.set_ylabel("Values")
            ax.legend()
    
    plt.tight_layout()
    plt.show()
    
def plot_train_test_dataset_regression(train: pd.DataFrame, test: pd.DataFrame) -> None:
    fig, (ax_1, ax_2) = plt.subplots(1, 2,figsize=(10, 5))
    sns.scatterplot(x = train[["x"]].to_numpy().flatten(), y = train[["y"]].to_numpy().flatten(), 
                    color="navy", ax = ax_1)
    ax_1.set_title("Training dataset")
    sns.scatterplot(x = test[["x"]].to_numpy().flatten(), y = test[["y"]].to_numpy().flatten(), 
                    color="navy", ax = ax_2)
    ax_2.set_title("Test dataset")
    plt.show()
    
def plot_loss_over_epoch_regularization_comparison(losses_noreg, L1_results, L2_results):
    loss_data = []
    for epoch, loss in enumerate(losses_noreg):
        loss_data.append({"Epoch": epoch, "Loss": loss, "Regularization": "None", "Lambda": 0})
    for lambd, history in L1_results.items():
        for epoch, loss in enumerate(history["losses"]):
            loss_data.append({"Epoch": epoch, "Loss": loss, "Regularization": "L1", "Lambda": lambd})
    for lambd, history in L2_results.items():
        for epoch, loss in enumerate(history["losses"]):
            loss_data.append({"Epoch": epoch, "Loss": loss, "Regularization": "L2", "Lambda": lambd})
    loss_df = pd.DataFrame(loss_data)

    plt.figure(figsize=(12, 8))
    sns.lineplot(data=loss_df[loss_df["Epoch"] >= 100], x="Epoch", y="Loss", hue="Regularization", markers=True, palette="bright")
    plt.title("Loss over Epochs for Different Regularization Types (from epoch 100)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(title="Regularization & Lambda")
    plt.grid()
    plt.show()
    
def plot_loss_lambda_regularization_comparison(L1_results, L2_results):
    test_loss_data = []
    for lambd, history in L1_results.items():
        test_loss_data.append({"Lambda": lambd, "Loss": history["loss_test"], "Regularization": "L1"})

    for lambd, history in L2_results.items():
        test_loss_data.append({"Lambda": lambd, "Loss": history["loss_test"], "Regularization": "L2"})

    test_loss_df = pd.DataFrame(test_loss_data)

    plt.figure(figsize=(12, 8))
    sns.lineplot(data=test_loss_df, x="Lambda", y="Loss", hue="Regularization", marker="o", palette="bright")
    plt.title("Test Loss over Lambda for Different Regularization Types")
    plt.xscale("log")
    plt.xlabel("Lambda")
    plt.ylabel("Test Loss")
    plt.legend(title="Regularization")
    plt.grid()
    plt.show()
    
def plot_regression_prediction_regularization_comparison(test_df, y_pred_noreg, loss_noreg, L1_results, L2_results):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)

    axes[0].scatter(test_df["x"], test_df["y"], label="True", color="blue")
    axes[0].scatter(test_df["x"], y_pred_noreg, label="Predicted", color="red")
    axes[0].set_title(f"No Regularization with loss = {loss_noreg:.4f}")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].legend()

    best_l1_lambda = min(L1_results, key=lambda lambd: L1_results[lambd]["loss_test"])
    axes[1].scatter(test_df["x"], test_df["y"], label="True", color="blue")
    axes[1].scatter(test_df["x"], L1_results[best_l1_lambda]["predictions"], label=f"Predicted (λ={best_l1_lambda})", color="red")
    axes[1].set_title(f"Best L1 Regularization with loss = {L1_results[best_l1_lambda]['loss_test']:.4f}")
    axes[1].set_xlabel("x")
    axes[1].legend()

    best_l2_lambda = min(L2_results, key=lambda lambd: L2_results[lambd]["loss_test"])
    axes[2].scatter(test_df["x"], test_df["y"], label="True", color="blue")
    axes[2].scatter(test_df["x"], L2_results[best_l2_lambda]["predictions"], label=f"Predicted (λ={best_l2_lambda})", color="red")
    axes[2].set_title(f"Best L2 Regularization with loss = {L2_results[best_l2_lambda]['loss_test']:.4f}")
    axes[2].set_xlabel("x")
    axes[2].legend()

    plt.tight_layout()
    plt.show()
    



def plot_f1_score_regularization_comparison(f1_scores_noreg: list, L1_results: dict, L2_results: dict) -> None:
    f1_data = []
    for epoch, f1 in enumerate(f1_scores_noreg):
        f1_data.append({"Epoch": epoch, "F1 Score": f1, "Regularization": "None", "Lambda": 0})
    for lambd, history in L1_results.items():
        for epoch, f1 in enumerate(history["f1_scores"]):
            f1_data.append({"Epoch": epoch, "F1 Score": f1, "Regularization": "L1", "Lambda": lambd})
    for lambd, history in L2_results.items():
        for epoch, f1 in enumerate(history["f1_scores"]):
            f1_data.append({"Epoch": epoch, "F1 Score": f1, "Regularization": "L2", "Lambda": lambd})
    f1_df = pd.DataFrame(f1_data)

    plt.figure(figsize=(12, 8))
    sns.lineplot(data=f1_df[f1_df["Epoch"] >= 100], x="Epoch", y="F1 Score", hue="Regularization", markers=True, palette="bright")
    plt.title("F1 Score over Epochs for Different Regularization Types (from epoch 100)")
    plt.xlabel("Epoch")
    plt.ylabel("F1 Score")
    plt.legend(title="Regularization & Lambda")
    plt.grid()
    plt.show()
    
def plot_f1_score_regularization_lambda_comparison(L1_results: dict, L2_results: dict) -> None:
    test_f1_data = []
    for lambd, history in L1_results.items():
        test_f1_data.append({"Lambda": lambd, "F1 Score": history["f1_scores"][-1], "Regularization": "L1"})

    for lambd, history in L2_results.items():
        test_f1_data.append({"Lambda": lambd, "F1 Score": history["f1_scores"][-1], "Regularization": "L2"})

    test_f1_df = pd.DataFrame(test_f1_data)

    plt.figure(figsize=(12, 8))
    sns.lineplot(data=test_f1_df, x="Lambda", y="F1 Score", hue="Regularization", marker="o", palette="bright")
    plt.title("Test F1 Score over Lambda for Different Regularization Types")
    plt.xscale("log")
    plt.xlabel("Lambda")
    plt.ylabel("F1 Score")
    plt.legend(title="Regularization")
    plt.grid()
    plt.show()
    
    

def plot_predicted_classes_regularization_comparison(test, preds_l1, preds_l2, preds_noreg, true_classes):
    x_test = test[["x", "y"]].to_numpy()

    good_pred_l1 = (preds_l1 == true_classes.flatten()).reshape(-1, 1)
    good_pred_l2 = (preds_l2 == true_classes.flatten()).reshape(-1, 1)
    good_pred_noreg = (preds_noreg == true_classes.flatten()).reshape(-1, 1)
    
    x = x_test[:, 0]
    y = x_test[:, 1]

    true_classes = true_classes.flatten()
        
    fig, ((ax_1, ax_2, ax_3), (ax_4, ax_5, ax_6)) = plt.subplots(2, 3, figsize=(12, 8))
    sns.scatterplot(x=x, y=y, hue=preds_noreg, palette="viridis", ax=ax_1)
    ax_1.set_title("Predicted classes no regularization")
    sns.scatterplot(x=x, y=y, hue=preds_l1, palette="viridis", ax=ax_2)
    ax_2.set_title("Predicted classes L1")
    sns.scatterplot(x=x, y=y, hue=preds_l2, palette="viridis", ax=ax_3)
    ax_3.set_title("Predicted classes L2")
    sns.scatterplot(x=x, y=y, hue=good_pred_noreg.flatten(), palette="hls", ax=ax_4)
    ax_4.set_title("Good vs. bad predictions no regularization")
    sns.scatterplot(x=x, y=y, hue=good_pred_l1.flatten(), palette="hls", ax=ax_5)
    ax_5.set_title("Good vs. bad predictions L1")
    sns.scatterplot(x=x, y=y, hue=good_pred_l2.flatten(), palette="hls", ax=ax_6)
    ax_6.set_title("Good vs. bad predictions L2")
    plt.show()
    
    
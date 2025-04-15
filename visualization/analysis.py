import time

import numpy as np
import pandas as pd

from network.preprocessing import classes_to_one_hot
from network.regularization import RegularizationL1, RegularizationL2, RegularizationNone
from network.mlp import MLP
from network.activations import LeakyReLU, ReLU, Sigmoid, Softmax, Identity, Tanh
from network.losses import LossCrossEntropy, LossMSE, LossFunction
from network.preprocessing import StandardScaler
from visualization.visualization import plot_dataset_classes, plot_f1_score_regularization_comparison, plot_f1_score_regularization_lambda_comparison, plot_loss_lambda_regularization_comparison, plot_loss_over_epoch_regularization_comparison, plot_predicted_classes_regularization_comparison, plot_regression_prediction_regularization_comparison, plot_train_test_dataset_regression




def compare_all_methods(train: pd.DataFrame,
                        test: pd.DataFrame,
                        task: str,
                        numbers_of_hidden_layers: list,
                        numbers_of_neurons: list,
                        activations: list,
                        epochs: int,
                        method: str = "sgd",
                        learninig_rate: float = 0.001):
    if task == "regr":
        loss_func = LossMSE()
    elif task == "class":
        loss_func = LossCrossEntropy()
        num_classes = len(train[["c"]].unique())
    else:
        raise ValueError("Task must be either 'regr' or 'class'.")
    
    results = []
    for number_of_hidden_layers in numbers_of_hidden_layers:
        for number_of_neurons in numbers_of_neurons:
            for activation in activations:
                layers_init = []
                for i in range(number_of_hidden_layers):
                    if isinstance(activation, ReLU) or isinstance(activation, LeakyReLU):
                        init = "he_normal"
                    elif isinstance(activation, Sigmoid) or isinstance(activation, Tanh):
                        init = "xavier_normal"
                    else:
                        init = "uniform"
                    layers_init.append({"output_size": number_of_neurons, "activation": activation, "init": init})
                if task == "class":
                    layers_init.append({"output_size": num_classes, "activation": Softmax(), "init": "xavier_normal"})
                elif task == "regr":    
                    layers_init.append({"output_size": 1, "activation": Identity(), "init": "uniform"})
                
                # scaling
                scaler = StandardScaler()
                X_train = scaler.fit_transform(train[["x"]])
                X_test = scaler.transform(test[["x"]])
                y_train = train[["y"]].values if task == "regr" else pd.get_dummies(train[["c"]]).values
                y_test = test[["y"]].values if task == "regr" else pd.get_dummies(test[["c"]]).values
                
                # Train and evaluate the method
                result = analyze_method(X_train=X_train,
                                        y_train=y_train,
                                        X_test=X_test,
                                        y_test=y_test,
                                        task=task,
                                        layers_init=layers_init,
                                        epochs=epochs,
                                        loss_function=loss_func,
                                        method=method,
                                        learninig_rate=learninig_rate)
                results.append(result)
                print(f"Finished training with {number_of_hidden_layers} hidden layers, {number_of_neurons} neurons and {activation.__class__.__name__} activation function.")
                
    return results
        
            
    

def analyze_method(X_train: np.ndarray,
                   y_train: np.ndarray,
                   X_test: np.ndarray,
                   y_test: np.ndarray,
                   task: str,
                   layers_init: list,
                   epochs: int,
                   loss_function: LossFunction,
                   method: str = "sgd",
                   learninig_rate: float = 0.001, 
                   clip_value: float = None,
                   verbose: bool = False) -> dict:
    
    res = {}
    
    nn = MLP(layers_init=layers_init,
             input=X_train,
             loss_function=loss_function,
             random_state=42)
    
    print("Training...") if verbose else None
    
    t1 = time.time()
    nn.train(X_train=X_train,
            Y_train=y_train,
            epochs=epochs,
            learning_rate=learninig_rate,
            method=method,
            clip_value=clip_value,
            verbose=False)
    t2 = time.time()
    
    res["training_time"] = t2 - t1
    res["training_history_loss"] = nn.get_training_history()["losses"]
    
    pred = nn.predict(X_test)
    if task == "regr":
        pred = np.array(pred).flatten()
        y_test = np.array(y_test).flatten()
        
        final_mse_train = nn.get_training_history()["losses"][-1]
        mse = LossMSE()
        mse_test = mse(y_test, pred)
        
        res["final_mse_train"] = final_mse_train
        res["mse_test"] = mse_test
        
    elif task == "class":
        pred = np.argmax(pred, axis=1)
        y_test = np.argmax(y_test, axis=1)
        
        final_f1_train = nn.get_training_history()["f1_scores"][-1]
        f1_test = nn.get_f1_score(y_test, pred)
        final_loss_train = nn.get_training_history()["losses"][-1]
        
        res["training_history_f1"] = nn.get_training_history()["f1_scores"]
        res["final_f1_train"] = final_f1_train
        res["f1_test"] = f1_test
        res["final_loss_train"] = final_loss_train
    
    res["pred"] = pred
    res["numbers_of_hidden_layers"] = len(layers_init) - 1
    res["numbers_of_neurons"] = layers_init[0]["output_size"]
    res["activation"] = layers_init[0]["activation"].__class__.__name__
    
    return res


def run_regularization_regression(train_df: pd.DataFrame,
                       test_df: pd.DataFrame,
                       layers_init: list,
                       loss_function: LossFunction,
                       lambdas: list,
                       epochs: int = 1000,
                       method: str = "sgd",
                       learning_rate: float = 0.001,
                       batch_size: int = None,
                       clip_value: float = None,
                       patience: int = None,
                       delta: float = None,
                       plotting: bool = True,
                       random_state: int = 42
                       ) -> None:
    
    x_train = train_df[["x"]].to_numpy()
    y_train = train_df[["y"]].to_numpy()
    x_test = test_df[["x"]].to_numpy()
    y_test = test_df[["y"]].to_numpy()
    x_val = x_train[:int(len(x_train)/4)]
    y_val = y_train[:int(len(y_train)/4)]
    
    if plotting:
        print("Visualizing train and test datasets...")
        plot_train_test_dataset_regression(train_df, test_df)
    
    print("Training without regularization")
    
    nn = MLP(layers_init=layers_init, input=x_train, loss_function=loss_function, random_state=random_state)
    nn.train(X_train=x_train, Y_train=y_train, X_val=x_val, Y_val=y_val,
             regularization=RegularizationNone(), epochs=epochs, method=method, learning_rate=learning_rate,
             batch_size=batch_size, clip_value=clip_value,
             patience=patience, delta=delta, verbose = False)
    
    no_reg_hist = nn.get_training_history()
    losses_noreg = no_reg_hist["losses"]
    f1_scores_noreg = no_reg_hist["f1_scores"]
    
    y_pred_noreg = nn.predict(x_test)
    loss_noreg = nn._get_loss_with_regularization(regularization=RegularizationNone(), x=x_test, y=y_test)
    
    print("Training with L1 regularization")
    L1_results = {}
    
    for lambd in lambdas:
        nn = MLP(layers_init=layers_init, input=x_train, loss_function=loss_function, random_state=random_state)
        nn.train(X_train=x_train, Y_train=y_train, X_val=x_val, Y_val=y_val,
                regularization=RegularizationL1(lambd=lambd), epochs=epochs, method=method, learning_rate=learning_rate,
                batch_size=batch_size, clip_value=clip_value,
                patience=patience, delta=delta, verbose = False)
        
        L1_results[lambd] = nn.get_training_history()
        y_pred = nn.predict(x_test)
        L1_results[lambd]["predictions"] = y_pred
        
        loss_L1_test = nn._get_loss_with_regularization(regularization=RegularizationL1(lambd=lambd), x=x_test, y=y_test)
        L1_results[lambd]["loss_test"] = loss_L1_test
    
    print("Training with L2 regularization")
    L2_results = {}
    
    for lambd in lambdas:
        nn = MLP(layers_init=layers_init, input=x_train, loss_function=loss_function, random_state=random_state)
        nn.train(X_train=x_train, Y_train=y_train, X_val=x_val, Y_val=y_val,
                regularization=RegularizationL2(lambd=lambd), epochs=epochs, method=method, learning_rate=learning_rate,
                batch_size=batch_size, clip_value=clip_value, 
                patience=patience, delta=delta, verbose = False)
        
        L2_results[lambd] = nn.get_training_history()
        y_pred = nn.predict(x_test)
        L2_results[lambd]["predictions"] = y_pred
        
        loss_L2_test = nn._get_loss_with_regularization(regularization=RegularizationL2(lambd=lambd), x=x_test, y=y_test)
        L2_results[lambd]["loss_test"] = loss_L2_test
        
    if plotting:
        print("Visualizing loss over epoch for different regularization types...")
        plot_loss_over_epoch_regularization_comparison(losses_noreg, L1_results, L2_results)
        
        print("Visualizing loss on test dataset in respect to lambda for different regularization types...")
        plot_loss_lambda_regularization_comparison(L1_results, L2_results)
    
        print("Visualizing average predictions for different regularization types...")
        plot_regression_prediction_regularization_comparison(test_df, y_pred_noreg, loss_noreg, L1_results, L2_results)  
    
    return L1_results, L2_results, loss_noreg, y_pred_noreg, losses_noreg
    




def run_regularization_classification(train_df: pd.DataFrame,
                                       test_df: pd.DataFrame,
                                       layers_init: list,
                                       loss_function: LossFunction,
                                       lambdas: list,
                                       epochs: int = 1000,
                                       method: str = "sgd",
                                       learning_rate: float = 0.001,
                                       batch_size: int = None,
                                       clip_value: float = None,
                                       patience: int = None,
                                       delta: float = None,
                                       plotting: bool = True,
                                       random_state: int = 42
                                       ) -> None:
    
    x_train = train_df[["x", "y"]].to_numpy()
    y_train = train_df[["c"]].to_numpy()
    x_test = test_df[["x", "y"]].to_numpy()
    y_test = test_df[["c"]].to_numpy()
    x_val = x_train[:int(len(x_train)/4)]
    y_val = y_train[:int(len(y_train)/4)]
    
    number_of_classes = len(np.unique(y_train))
    scaler = StandardScaler()
    
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)
    x_test = scaler.transform(x_test)
    y_train_cl = y_train.copy()
    y_train = classes_to_one_hot(y_train, number_of_classes)
    y_val_cl = y_val.copy()
    y_val = classes_to_one_hot(y_val, num_classes=number_of_classes)
    y_test_cl = y_test.copy()
    y_test = classes_to_one_hot(y_test, number_of_classes)  
    
    L1_results, L2_results, f1_noreg_test, best_l1_lambda, best_l2_lambda, best_l1_f1, best_l2_f1, f1_scores_noreg = None, None, None, None, None, None, None, None
    
    if plotting:
        print("Visualizing train and test datasets...")
        plot_dataset_classes(train_df, test_df)  
    
    print("Training without regularization...")
    
    nn = MLP(layers_init=layers_init, input=x_train, loss_function=loss_function, random_state=random_state)
    nn.train(X_train=x_train, Y_train=y_train, X_val=x_val, Y_val=y_val,
             regularization=RegularizationNone(), epochs=epochs, method=method, learning_rate=learning_rate,
             batch_size=batch_size, clip_value=clip_value,
             patience=patience, delta=delta, verbose=False)
    
    no_reg_hist = nn.get_training_history()
    f1_scores_noreg = no_reg_hist["f1_scores"]
    y_pred_noreg = np.argmax(nn.forward(x_test), axis=1)
    f1_noreg_test = nn.calculate_f1_macro(y_test, nn.forward(x_test))
    
    print("Training with L1 regularization...")
    L1_results = {}
    
    for lambd in lambdas:
        nn = MLP(layers_init=layers_init, input=x_train, loss_function=loss_function, random_state=random_state)
        nn.train(X_train=x_train, Y_train=y_train, X_val=x_val, Y_val=y_val,
                 regularization=RegularizationL1(lambd=lambd), epochs=epochs, method=method, learning_rate=learning_rate,
                 batch_size=batch_size, clip_value=clip_value,
                 patience=patience, delta=delta, verbose=False)
        
        L1_results[lambd] = nn.get_training_history()
        y_pred = nn.predict(x_test)
        L1_results[lambd]["predictions"] = np.argmax(y_pred, axis=1)
        
        L1_f1_test = nn.calculate_f1_macro(y_test, y_pred)
        L1_results[lambd]["f1_test"] = L1_f1_test
        
    print("Training with L2 regularization...")
    L2_results = {}
    
    for lambd in lambdas:
        nn = MLP(layers_init=layers_init, input=x_train, loss_function=loss_function, random_state=random_state)
        nn.train(X_train=x_train, Y_train=y_train, X_val=x_val, Y_val=y_val,
                 regularization=RegularizationL2(lambd=lambd), epochs=epochs, method=method, learning_rate=learning_rate,
                 batch_size=batch_size, clip_value=clip_value,
                 patience=patience, delta=delta, verbose=False)
        
        L2_results[lambd] = nn.get_training_history()
        y_pred = nn.predict(x_test)
        L2_results[lambd]["predictions"] = np.argmax(y_pred, axis=1)
        
        L2_f1_test = nn.calculate_f1_macro(y_test, y_pred)
        L2_results[lambd]["f1_test"] = L2_f1_test
        
    if plotting:
        print("Visualizing F1 score over epochs...")
        plot_f1_score_regularization_comparison(f1_scores_noreg, L1_results, L2_results)
        
        print("Visualizing F1 score over lambda for different regularization types...")
        plot_f1_score_regularization_lambda_comparison(L1_results, L2_results)
        
        print("Visualizing best class predictions...")
        
        best_preds_l1 = None
        best_l1_lambda = None
        best_l1_f1 = 0
        for lambd, history in L1_results.items():
            if history["f1_test"] > best_l1_f1:
                best_l1_f1 = history["f1_test"]
                best_preds_l1 = history["predictions"]
                best_l1_lambda = lambd
        best_preds_l2 = None
        best_l2_lambda = None
        best_l2_f1 = 0
        for lambd, history in L2_results.items():
            if history["f1_test"] > best_l2_f1:
                best_l2_f1 = history["f1_test"]
                best_preds_l2 = history["predictions"]
                best_l2_lambda = lambd
                
        plot_predicted_classes_regularization_comparison(test_df, best_preds_l1, best_preds_l2, y_pred_noreg, y_test_cl)
        
        print("Best F1 scores:")
        print(f"No regularization: {f1_noreg_test:.4f}")
        print(f"L1 regularization (lambda={best_l1_lambda}): {best_l1_f1:.4f}")
        print(f"L2 regularization (lambda={best_l2_lambda}): {best_l2_f1:.4f}")
        
        
    return L1_results, L2_results, f1_noreg_test, best_l1_lambda, best_l2_lambda, best_l1_f1, best_l2_f1, f1_scores_noreg

    
    
    
import numpy as np
import matplotlib.pyplot as plt

from network.layers import FullyConnectedLayer
from network.losses import LossFunction, LossCrossEntropy
from network.activations import Softmax
from network.regularization import Regularization

class MLP:
    def __init__(self, layers_init: list, input: np.ndarray, loss_function: LossFunction , random_state: int = None) -> None:
        """
        Initializes the Multi-Layer Perceptron (MLP) model.
        Args:
            layers_init (list): A list of dictionaries where each dictionary defines the parameters 
                for a layer. Each dictionary should include:
                - "input_size" (int, optional): The size of the input to the layer. For the first layer, 
                  this is inferred from the input data.
                - "output_size" (int): The size of the output from the layer.
                - "activation" (callable): The activation function for the layer.
                - "init" (callable, optional): The initialization function for the weights and biases.
                - "weights" (np.ndarray, optional): Predefined weights for the layer.
                - "biases" (np.ndarray, optional): Predefined biases for the layer.
            input (np.ndarray): The input data to the MLP. Can be a 1D or 2D array.
            loss_function (LossFunction): The loss function to be used for training the MLP.
            random_state (int, optional): A seed for the random number generator to ensure reproducibility.
        Attributes:
            layers (list): A list of FullyConnectedLayer objects representing the layers of the MLP.
            weights (list): A list of weight matrices for each layer.
            biases (list): A list of bias vectors for each layer.
            layers_init (list): The initialization parameters for the layers.
            depth (int): The number of layers in the MLP.
            loss_function (LossFunction): The loss function used for training.
            training_history (dict): A dictionary to store the training history, including:
                - "weights" (list): The weights at each training step.
                - "biases" (list): The biases at each training step.
                - "losses" (list): The training losses at each step.
                - "f1_scores" (list): The F1 scores at each step.
                - "losses_val" (list): The validation losses at each step.
        Raises:
            ValueError: If the activation function is not specified for a layer.
            ValueError: If the dimensions of weights and biases do not match the layer dimensions.
        """
        
        
        np.random.seed(random_state)

        if input.ndim == 1:
            x_size = 1 
        else:
            x_size = input.shape[1]
            
        self.layers = []
        self.weights = []
        self.biases = []

        self.layers_init = layers_init

        for i, layer_params in enumerate(layers_init):
            
            input_size = layer_params.get("input_size")
            if i == 0:
                input_size = x_size
            else:
                input_size = layers_init[i-1]["output_size"]
                
            output_size = layer_params.get("output_size")

            activation = layer_params.get("activation")
            if activation is None:
                raise ValueError(f"Activation function specified incorrectly for layer{i}.")
            
            init_func = layer_params.get("init")

            weights = layer_params.get("weights")
            biases = layer_params.get("biases")

            if weights is not None and biases is not None:
                try:
                    layer = FullyConnectedLayer(input_size=input_size, output_size=output_size, 
                                                    activation=activation, init=init_func,
                                                    weights=weights, biases=biases)
                except:
                    raise ValueError(f"Dimensions of weights and biases do not match the dimensions of layer {i}.")

            else:
                layer = FullyConnectedLayer(input_size=input_size, output_size=output_size, activation=activation, init=init_func)
                

            self.layers.append(layer)
            self.weights.append(layer.get_weights())
            self.biases.append(layer.get_biases())
            
        self.depth = len(self.layers)
        
        self.loss_function = loss_function
        
        self.training_history = {
            "weights" : [],
            "biases" : [],
            "losses" : [],
            "f1_scores" : [],
            "losses_val" : [],
        }


    def forward(self, input: np.ndarray) -> np.ndarray:
        """
        Perform a forward pass through the network.
        Args:
            input (np.ndarray): The input data to the network.
        Returns:
            np.ndarray: The output of the network after passing through all layers.
        """

        output = input 
        for layer in self.layers:
            output = layer.forward(output)
        return output
    
    def backpropagation(self, x: np.ndarray, y: np.ndarray) -> tuple:
        """
        Perform backpropagation to calculate the gradients of the loss function with respect to the weights and biases.
        Args:
            x (np.ndarray): The input data to the network.
            y (np.ndarray): The true output of the network.
        Returns:
            tuple: A tuple containing the gradients of the loss function with respect to the weights and biases.
        """
        
        deltas = [None] * self.depth
        activation = x
        activations = [x]
        zs = []

        for layer in self.layers:
            z = np.dot(activation, layer.get_weights().T) + layer.get_biases().T
            zs.append(z)
            activation = layer.activation(z)
            activations.append(activation)
        
        if not (isinstance(self.layers[-1].activation, Softmax) and isinstance(self.loss_function, LossCrossEntropy)):    
            deltas[-1] = (activations[-1] - y) * self.layers[-1].activation.derivative(zs[-1])
        else:
            deltas[-1] = (activations[-1] - y)
        
        for l in range(len(self.layers) - 2, -1, -1):
            deltas[l] = np.multiply(np.dot(deltas[l+1], self.layers[l+1].get_weights()), self.layers[l].activation.derivative(zs[l]))
        
        nabla_w = [np.dot(deltas[i].T, activations[i]) for i in range(self.depth)]
        nabla_b = [np.sum(deltas[i], axis=0).reshape(self.layers[i].biases.shape) for i in range(self.depth)]
        
        return nabla_w, nabla_b
            

    
    def update_mini_batch(self, 
                          batch_x: np.ndarray, 
                          batch_y: np.ndarray, 
                          regularization: Regularization, 
                          method: str = "sgd", 
                          learning_rate: float = 0.001, 
                          weights_decay: float = 0.9, 
                          beta: float = 0.9,
                          beta1: float = 0.9, 
                          beta2: float = 0.999, 
                          eps: float = 10e-6, 
                          clip_value: float = None) -> None:
        """
        Update the model's weights and biases using a mini-batch of training data and a specified optimization method.
        Args:
            batch_x (np.ndarray): Input data for the mini-batch.
            batch_y (np.ndarray): Target labels for the mini-batch.
            regularization (Regularization): Regularization object to compute regularization gradients.
            method (str, optional): Optimization method to use. Options are "sgd", "momentum", "rmsprop", and "adam". 
                                    Defaults to "sgd".
            learning_rate (float, optional): Learning rate for the optimization. Defaults to 0.001.
            weights_decay (float, optional): Decay rate for momentum-based methods. Defaults to 0.9.
            beta (float, optional): Decay rate for RMSProp. Defaults to 0.9.
            beta1 (float, optional): Exponential decay rate for the first moment estimates in Adam. Defaults to 0.9.
            beta2 (float, optional): Exponential decay rate for the second moment estimates in Adam. Defaults to 0.999.
            eps (float, optional): Small constant to prevent division by zero in optimization methods. Defaults to 10e-6.
            clip_value (float, optional): Value to clip gradients to avoid exploding gradients. If None, no clipping is applied. 
                                          Defaults to None.
        Returns:
            None
        Raises:
            AssertionError: If the specified optimization method is not one of "sgd", "momentum", "rmsprop", or "adam".
        Notes:
            - The method supports four optimization algorithms:
                - "sgd": Stochastic Gradient Descent.
                - "momentum": SGD with momentum.
                - "rmsprop": Root Mean Square Propagation.
                - "adam": Adaptive Moment Estimation.
            - Regularization gradients are added to the weight gradients before applying updates.
            - Gradient clipping is applied if `clip_value` is specified.
            - For "momentum", "rmsprop", and "adam", internal state variables are initialized on the first call.
        """
        
        assert method in ["sgd", "momentum", "rmsprop", "adam"], "Invalid optimization method. Choose from sgd, momentum, rmsprop, adam."
        
        nabla_w, nabla_b = self.backpropagation(batch_x, batch_y)
        nabla_regularization = self._calculate_regularization_gradients(regularization)
        nabla_w = [nabla_w[i] + nabla_regularization[i] for i in range(len(nabla_w))]
        
        if clip_value is not None:
            nabla_w = [np.clip(w, -clip_value, clip_value) for w in nabla_w]
            nabla_b = [np.clip(b, -clip_value, clip_value) for b in nabla_b]
        
        if method == "sgd":
            for i, layer in enumerate(self.layers):
                new_weights = layer.get_weights() - learning_rate * nabla_w[i] / len(batch_x)
                nabla_b[i] = np.array(nabla_b[i]).reshape(layer.biases.shape)
                new_biases = layer.get_biases() - learning_rate * nabla_b[i] / len(batch_x)
                layer.set_weights(new_weights)
                layer.set_biases(new_biases)
        
        elif method == "momentum":
            if not hasattr(self, 'v_w'):
                self.v_w = [np.zeros_like(w) for w in nabla_w]
                self.v_b = [np.zeros_like(b) for b in nabla_b]
            
            for i, layer in enumerate(self.layers):
                self.v_w[i] = weights_decay * self.v_w[i] - learning_rate * nabla_w[i] / len(batch_x)
                self.v_b[i] = weights_decay * self.v_b[i] - learning_rate * nabla_b[i] / len(batch_x)
                
                new_weights = layer.get_weights() + self.v_w[i]
                new_biases = layer.get_biases() + self.v_b[i]
                layer.set_weights(new_weights)
                layer.set_biases(new_biases)
                
        elif method == "rmsprop":
            if not hasattr(self, 'mean_sq_grad_w'):
                self.mean_sq_grad_w = [np.zeros_like(w) for w in nabla_w]
                self.mean_sq_grad_b = [np.zeros_like(b) for b in nabla_b]
                
            for i, layer in enumerate(self.layers):
                self.mean_sq_grad_w[i] = beta * self.mean_sq_grad_w[i] + (1 - beta) * (nabla_w[i] ** 2)
                self.mean_sq_grad_b[i] = beta * self.mean_sq_grad_b[i] + (1 - beta) * (nabla_b[i] ** 2)
                
                new_weights = layer.get_weights() - ((learning_rate / np.sqrt(self.mean_sq_grad_w[i] + eps)) * nabla_w[i])
                new_biases = layer.get_biases() - ((learning_rate / np.sqrt(self.mean_sq_grad_b[i] + eps)) * nabla_b[i])
                layer.set_weights(new_weights)
                layer.set_biases(new_biases)
        
        elif method == "adam":
            if not hasattr(self, 'm_w'):
                self.m_w = [np.zeros_like(w) for w in nabla_w]
                self.m_b = [np.zeros_like(b) for b in nabla_b]
                self.v_w = [np.zeros_like(w) for w in nabla_w]
                self.v_b = [np.zeros_like(b) for b in nabla_b]
                self.t = 0
            
            self.t += 1
                
            for i, layer in enumerate(self.layers):
                self.m_w[i] = beta1 * self.m_w[i] + (1 - beta1) * nabla_w[i]
                self.m_b[i] = beta1 * self.m_b[i] + (1 - beta1) * nabla_b[i]
                self.v_w[i] = beta2 * self.v_w[i] + (1 - beta2) * (nabla_w[i] ** 2)
                self.v_b[i] = beta2 * self.v_b[i] + (1 - beta2) * (nabla_b[i] ** 2)
                
                m_w_hat = self.m_w[i] / (1 - beta1 ** self.t)
                m_b_hat = self.m_b[i] / (1 - beta1 ** self.t)
                v_w_hat = self.v_w[i] / (1 - beta2 ** self.t)
                v_b_hat = self.v_b[i] / (1 - beta2 ** self.t)
                
                new_weights = layer.get_weights() - (learning_rate / (np.sqrt(v_w_hat) + eps)) * m_w_hat / len(batch_x)
                new_biases = layer.get_biases() - (learning_rate / (np.sqrt(v_b_hat) + eps)) * m_b_hat / len(batch_x)
                    
                layer.set_weights(new_weights)
                layer.set_biases(new_biases)
                
                
            
    def train(self, 
              X_train: np.ndarray, 
              Y_train: np.ndarray, 
              regularization: Regularization,
              X_val: np.ndarray = None,
              Y_val: np.ndarray = None,
              epochs: int = 1000, 
              stop_loss: float = None,
              stop_f1: float = None, 
              method: str = "sgd", 
              learning_rate: float = 0.001, 
              weights_decay: float = 0.9, 
              beta: float = 0.9, 
              beta1: float = 0.9, 
              beta2: float = 0.999, 
              eps: float = 10e-6,
              batch_size: int = None,
              clip_value: float = None,
              verbose: bool = True,
              patience: int = None,
              delta: float = None) -> None:
        """
            Train the model using the specified parameters.
            Args:
                X_train (np.ndarray): Training input data.
                Y_train (np.ndarray): Training target data.
                regularization (Regularization): Regularization method to apply during training.
                X_val (np.ndarray, optional): Validation input data. Defaults to None.
                Y_val (np.ndarray, optional): Validation target data. Defaults to None.
                epochs (int, optional): Number of training epochs. Defaults to 1000.
                stop_loss (float, optional): Early stopping threshold for loss. Defaults to None.
                stop_f1 (float, optional): Early stopping threshold for F1 score. Defaults to None.
                method (str, optional): Optimization method (e.g., "sgd", "adam"). Defaults to "sgd".
                learning_rate (float, optional): Learning rate for the optimizer. Defaults to 0.001.
                weights_decay (float, optional): Weight decay factor for regularization. Defaults to 0.9.
                beta (float, optional): Momentum parameter for SGD. Defaults to 0.9.
                beta1 (float, optional): Exponential decay rate for the first moment estimates in Adam. Defaults to 0.9.
                beta2 (float, optional): Exponential decay rate for the second moment estimates in Adam. Defaults to 0.999.
                eps (float, optional): Small constant for numerical stability in optimizers. Defaults to 10e-6.
                batch_size (int, optional): Size of mini-batches for training. Defaults to None (full batch).
                clip_value (float, optional): Gradient clipping threshold. Defaults to None.
                verbose (bool, optional): Whether to print training progress. Defaults to True.
                patience (int, optional): Number of epochs to wait for early stopping. Defaults to None.
                delta (float, optional): Minimum change in validation loss for early stopping. Defaults to None.
            Returns:
                None
            """
        
        
        losses = []
        f1_scores = []
        losses_val = []
        
        for epoch in range(epochs):
            indices = np.arange(X_train.shape[0])
            np.random.shuffle(indices)
            X_train = X_train[indices]
            Y_train = Y_train[indices]
            if batch_size is None:
                self.update_mini_batch(X_train, Y_train, method=method, learning_rate=learning_rate, 
                                       weights_decay=weights_decay, beta=beta, beta1=beta1, beta2=beta2,
                                       eps=eps, clip_value = clip_value, regularization=regularization)
            else:
                for i in range(0, X_train.shape[0], batch_size):
                    batch_x = X_train[i:i+batch_size]
                    batch_y = Y_train[i:i+batch_size]
                    self.update_mini_batch(batch_x, batch_y, method=method, learning_rate=learning_rate, 
                                       weights_decay=weights_decay, beta=beta, beta1=beta1, beta2=beta2,
                                       eps=eps, clip_value = clip_value, regularization=regularization)
            
            loss = self._get_loss_with_regularization(regularization=regularization, x=X_train, y=Y_train)
            losses.append(loss)
            
            if X_val is not None and Y_val is not None:
                loss_val = self._get_loss_with_regularization(regularization=regularization, x=X_val, y=Y_val)
                losses_val.append(loss_val) 
            
            if self.loss_function.f1_score:
                f1 = self.calculate_f1_macro(Y_train, self.forward(X_train))
                f1_scores.append(f1)
                self.training_history["f1_scores"].append(f1)
                            
                self._update_training_history(weights = [layer.get_weights() for layer in self.layers],
                                          biases = [layer.get_biases() for layer in self.layers],
                                          loss = loss,
                                          f1_score = f1,
                                          loss_val = loss_val)
            else:
                self._update_training_history(weights = [layer.get_weights() for layer in self.layers],
                                          biases = [layer.get_biases() for layer in self.layers],
                                          loss = loss,
                                          f1_score = None,
                                          loss_val = loss_val)
            
            if verbose and self.loss_function.f1_score:
                self._printout_training(epoch=epoch, epochs=epochs, loss=loss, f1=f1, loss_val=loss_val)
            elif verbose:
                self._printout_training(epoch=epoch, epochs=epochs, loss=loss, f1=None, loss_val=loss_val)   
                     
            if stop_loss is not None and loss <= stop_loss:
                break 
            if stop_f1 is not None and self.loss_function.f1_score and f1 >= stop_f1:
                break
            if patience is not None and delta is not None:
                if epoch > patience:
                    if np.abs(losses_val[-patience] - loss_val) <= delta:
                        print(f"Early stopping at epoch {epoch} with loss {loss:.4f} and validation loss {loss_val:.4f}.")
                        break
        if verbose:
            if self.loss_function.f1_score:
                print(f"Final epoch:\t{epoch}\tLoss ({self.loss_function}):\t\t{loss:.4f}\tF1 Score:\t{f1:.4f}")
            else:
                print(f"Final epoch:\t{epoch}\tLoss ({self.loss_function}):\t{loss:.4f}")
    
    def _update_training_history(self, weights, biases, loss, f1_score, loss_val) -> None:
        self.training_history["weights"].append(weights)
        self.training_history["biases"].append(biases)
        self.training_history["losses"].append(loss)
        self.training_history["f1_scores"].append(f1_score)
        self.training_history["losses_val"].append(loss_val)
    
    def _calculate_f1_score_per_class(self, y_true: np.ndarray, y_pred_prob: np.ndarray, class_index) -> float:
        """
        Calculates the F1 score for a specific class based on the true labels and predicted probabilities.
        Args:
            y_true (np.ndarray): A 2D array of shape (n_samples, n_classes) containing the true binary labels 
                     for each class (0 or 1).
            y_pred_prob (np.ndarray): A 2D array of shape (n_samples, n_classes) containing the predicted 
                          probabilities for each class.
            class_index (int): The index of the class for which the F1 score is to be calculated.
        Returns:
            float: The F1 score for the specified class. Returns 0 if precision + recall is 0.
        """
        
        y_pred_class = np.where(y_pred_prob[:, class_index] >= 0.5, 1, 0)
        y_true_class = y_true[:, class_index]
        
        tp = ((y_pred_class == 1) & (y_true_class == 1)).sum()
        fp = ((y_pred_class == 1) & (y_true_class == 0)).sum()
        fn = ((y_pred_class == 0) & (y_true_class == 1)).sum()
        
        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
        
        return f1_score
    
    def calculate_f1_macro(self, y_true: np.ndarray, y_pred_prob: np.ndarray) -> float:
        """
        Calculates the macro-averaged F1 score for a multi-class classification problem.
        The macro F1 score is computed by calculating the F1 score for each class individually
        and then taking the average of these scores. This method is useful when all classes
        are equally important, regardless of their frequency in the dataset.
        Args:
            y_true (np.ndarray): A 1D array of true class labels with shape (n_samples,).
            y_pred_prob (np.ndarray): A 2D array of predicted probabilities for each class 
                with shape (n_samples, n_classes).
        Returns:
            float: The macro-averaged F1 score across all classes.
        """
        
        f1_scores = []
        for i in range(y_pred_prob.shape[1]):
            f1_score = self._calculate_f1_score_per_class(y_true, y_pred_prob, i)
            f1_scores.append(f1_score)
        return np.mean(f1_scores)
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predict the output of the network for a given input.
        Args:
            x (np.ndarray): The input data to the network.
        Returns:
            np.ndarray: The predicted output of the network.
        """
        
        return self.forward(x)
    
    def _calculate_regularization_gradients(self, regularization: Regularization) -> list:
        regularisation_gradients = [np.zeros(w_i.shape) for w_i in self.weights]
        for i in range(len(self.weights)):
            regularisation_gradients[i] = regularization.grad(self.weights[i])
        return regularisation_gradients
    
    def _get_loss_with_regularization(self, regularization: Regularization, x: np.ndarray, y: np.ndarray) -> float:
        y_pred = self.predict(x)
        loss = self.loss_function(y_pred, y)
        reg_loss = regularization.loss(self.weights)
        return loss + reg_loss
        
    def _printout_training(self, epoch: int, epochs: int, loss: float, f1: float, loss_val: float = None) -> None:
        """
        Print the training progress.
        Args:
            epoch (int): The current epoch.
            loss (float): The current loss.
            f1 (float): The current F1 score.
            loss_val (float): The current validation loss.
        """
        if epochs <= 100:        
            if epoch % 10 == 0:
                if self.loss_function.f1_score:
                    print(f"Epoch:\t{epoch}\tLoss ({self.loss_function}):\t\t{loss:.4f}\tF1 Score:\t{f1:.4f}\tValidation Loss:\t{loss_val:.4f}")
                else:
                    print(f"Epoch:\t{epoch}\tLoss ({self.loss_function}):\t{loss:.4f}\tValidation Loss:\t{loss_val:.4f}")
        elif epochs <= 2000:        
            if epoch % 100 == 0:
                if self.loss_function.f1_score:
                    print(f"Epoch:\t{epoch}\tLoss ({self.loss_function}):\t\t{loss:.4f}\tF1 Score:\t{f1:.4f}\tValidation Loss:\t{loss_val:.4f}")
                else:
                    print(f"Epoch:\t{epoch}\tLoss ({self.loss_function}):\t{loss:.4f}\tValidation Loss:\t{loss_val:.4f}")
        elif epochs <= 20_000:
            if epoch % 1000 == 0:
                if self.loss_function.f1_score:
                    print(f"Epoch:\t{epoch}\tLoss ({self.loss_function}):\t\t{loss:.4f}\tF1 Score:\t{f1:.4f}\tValidation Loss:\t{loss_val:.4f}")
                else:
                    print(f"Epoch:\t{epoch}\tLoss ({self.loss_function}):\t{loss:.4f}\tValidation Loss:\t{loss_val:.4f}")
        else:
            if epoch % 2500 == 0:
                if self.loss_function.f1_score:
                    print(f"Epoch:\t{epoch}\tLoss ({self.loss_function}):\t\t{loss:.4f}\tF1 Score:\t{f1:.4f}\tValidation Loss:\t{loss_val:.4f}")
                else:
                    print(f"Epoch:\t{epoch}\tLoss ({self.loss_function}):\t{loss:.4f}\tValidation Loss:\t{loss_val:.4f}")       
    
    def visualize_prediction(self, x: np.ndarray, y: np.ndarray) -> None:
        """
        Visualize the true and predicted output of the network.
        Args:
            x (np.ndarray): The input data to the network.
            y (np.ndarray): The true output of the network.
        """
        
        assert x.shape == y.shape, "Input and output dimensions do not match"
        
        y_pred = self.forward(x)
        loss = self.loss_function(y, y_pred) 
        
        plt.figure(figsize=(10, 6))
        plt.scatter(x, y, label="True", color="blue")
        plt.scatter(x, y_pred, label="Predicted", color="red")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title(f"Prediction vs. true values with loss {self.loss_function.__class__.__name__} = {loss:.4f}")
        plt.legend()
        plt.show()
        print(f"Loss {self.loss_function.__class__.__name__} = {loss}.")
        
    def plot_training(self):
        """
        Plots the training history of the model, including loss and optionally F1 score.
        This method visualizes the changes in the loss function and, if applicable, 
        the F1 score during the training process. It dynamically adjusts the number 
        of subplots based on whether the F1 score is being tracked.
        Args:
            None
        Returns:
            None: Displays the plot of training metrics.
        Attributes:
            self.training_history (dict): A dictionary containing the training metrics. 
                Expected keys are:
                - "losses": A list of loss values recorded at each epoch.
                - "f1_scores" (optional): A list of F1 score values recorded at each epoch.
            self.loss_function (object): The loss function used during training. 
                It must have:
                - `f1_score` (bool): Indicates whether F1 score is being tracked.
                - `__str__()` method: Returns a string representation of the loss function.
        Raises:
            None
        """
        
        
        if self.loss_function.f1_score:
            fig, ax = plt.subplots(1, 2,figsize=(12, 4))
        else:
            fig, ax = plt.subplots(1, 1,figsize=(6, 4))
            ax = [ax]

        ax[0].plot(self.training_history["losses"], color="navy")
        ax[0].set_xlabel("Epoch")
        ax[0].set_ylabel("Loss")
        ax[0].set_title(f"Loss {self.loss_function.__str__()} changes during training")
        ax[0].grid()

        if self.loss_function.f1_score:
            ax[1].plot(self.training_history["f1_scores"], color="navy")
            ax[1].set_xlabel("Epoch")
            ax[1].set_ylabel("F1 score")
            ax[1].set_title("F1 score changes during training")
            ax[1].grid()

        plt.show()
        
               
    def get_training_history(self) -> dict:
        """
        Get the training history of the network.
        Returns:
            dict: A dictionary containing the training history of the network.
        """
        
        return self.training_history

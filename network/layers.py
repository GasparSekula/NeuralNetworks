import numpy as np

from .activations import ActivationFunction, Sigmoid

class Layer:
    def __init__(self):
        """
        Initializes the layer with default values for input size, output size, and activation function.
        Attributes:
            input_size (int or None): The size of the input to the layer. Default is None.
            output_size (int or None): The size of the output from the layer. Default is None.
            activation (callable or None): The activation function to be used in the layer. Default is None.
        """
        self.input_size = None
        self.output_size = None
        self.activation = None
    
    def forward(self, input):
        pass


class FullyConnectedLayer(Layer):
    def __init__(self, input_size: int, 
                 output_size: int, 
                 weights: np.ndarray = None, 
                 biases: np.ndarray = None, 
                 activation: ActivationFunction = Sigmoid, 
                 init: str = None):
        """
        Initializes a layer in the neural network.
        Args:
            input_size (int): The size of the input to the layer.
            output_size (int): The size of the output from the layer.
            weights (np.ndarray, optional): The weights matrix of shape (output_size, input_size). Defaults to None.
            biases (np.ndarray, optional): The biases vector of shape (output_size, 1). Defaults to None.
            activation (ActivationFunction, optional): The activation function to use. Must be one of Sigmoid, Identity, Tanh, ReLU, LeakyReLU. Defaults to Sigmoid.
            init (str, optional): The initialization method for weights and biases. Must be one of [None, "uniform", "he_normal", "xavier_normal"]. Defaults to None.
        Raises:
            ValueError: If weights and biases are not provided when init is None.
            ValueError: If the shape of weights does not match (output_size, input_size).
            ValueError: If the shape of biases does not match (output_size, 1).
        Attributes:
            input_size (int): The size of the input to the layer.
            output_size (int): The size of the output from the layer.
            activation_functions (dict): A dictionary mapping activation function names to their corresponding functions.
            activation (function): The activation function to use.
            weights (np.ndarray): The weights matrix.
            biases (np.ndarray): The biases vector.
        """
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size

        assert init in [None, "uniform", "xavier_normal", "he_normal"]

        self.activation = activation
        
        if init is None:
            if weights is not None and biases is not None:
                if weights.shape != (output_size, input_size):
                    raise ValueError("Shape of weights must match layer's dimensions.")
                if biases.shape != (output_size, 1):
                    raise ValueError("Shape of biases must match layer's dimensions.")

                self.weights = weights
                self.biases = biases

            else:
                raise ValueError("Weights and biases must be provided if init method is None.")

        elif init == "uniform":
            self.weights = np.random.uniform(0, 1, (self.output_size, self.input_size))
            self.biases = np.random.uniform(0, 1, (self.output_size,1))
        elif init == "xavier_normal":
            self.weights = np.random.normal(0, np.sqrt(2/(self.output_size + self.input_size)), (self.output_size, self.input_size))
            self.biases = np.random.normal(0, np.sqrt(2/(self.output_size + self.input_size)), (self.output_size,1))
        elif init == "he_normal":
            self.weights = np.random.normal(0, np.sqrt(2/(self.input_size)), (self.output_size, self.input_size))
            self.biases = np.random.normal(0, np.sqrt(2/(self.input_size)), (self.output_size,1))
            
            
    def forward_no_activation(self, input):
        """
        Computes the forward pass of the layer without applying an activation function.
        Args:
            input (numpy.ndarray): The input data to the layer, typically a 2D array where each row is a data sample.
        Returns:
            numpy.ndarray: The output of the layer before applying any activation function.
        """
        
        self.input = input 
        self.output_no_activation = np.matmul(input, self.weights.T) + self.biases.T
        return self.output_no_activation
    
    def forward(self, input):
        """
        Performs the forward pass of the layer.
        Args:
            input (numpy.ndarray): The input data to the layer.
        Returns:
            numpy.ndarray: The output of the layer after applying the activation function.
        """
        
        self.output = self.activation(self.forward_no_activation(input))
        return self.output
    
    def get_biases(self):
        """
        Returns the biases of the layer.
        Returns:
            numpy.ndarray: The biases vector of the layer.
        """
        return self.biases

    def get_weights(self):
        """
        Returns the weights of the layer.
        Returns:
            numpy.ndarray: The weights matrix of the layer.
        """
        return self.weights
    
    def set_weights(self, weights):
        """
        Sets the weights of the layer.
        Args:
            weights (numpy.ndarray): The new weights matrix.
        """
        self.weights = weights
        
    def set_biases(self, biases):
        """
        Sets the biases of the layer.
        Args:
            biases (numpy.ndarray): The new biases vector.
        """
        self.biases = biases
        
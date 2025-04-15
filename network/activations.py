import numpy as np

class ActivationFunction:
    def __call__(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Subclasses should implement this method.")
    
    def derivative(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Subclasses should implement this method.")
    
    
class Sigmoid(ActivationFunction):
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))
        
    def derivative(self, x):
        s = self(x)
        return s * (1 - s)
    
    def __str__(self):
        return "Sigmoid"
    
class Identity(ActivationFunction):
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return x
    
    def derivative(self, x: np.ndarray) -> np.ndarray:
        return np.ones_like(x)
    
    def __str__(self):
        return "Identity"
    
class Tanh(ActivationFunction):
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(x)
    
    def derivative(self, x: np.ndarray) -> np.ndarray:
        return 1 - np.tanh(x) ** 2
    
    def __str__(self):
        return "Tanh"

class ReLU(ActivationFunction):
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)
    
    def derivative(self, x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, 1, 0)
    
    def __str__(self):
        return "ReLU"
    
class LeakyReLU(ActivationFunction):
    def __init__(self, alpha: float = 0.01) -> None:
        self.alpha = alpha
        
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, x, self.alpha * x)
    
    def derivative(self, x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, 1, self.alpha)
    
    def __str__(self):
        return "Leaky ReLU"
    
class Softmax(ActivationFunction):
    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = x - np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def derivative(self, x: np.ndarray) -> np.ndarray:
        s = self(x).reshape(-1, 1)  
        return np.diagflat(s) - np.dot(s, s.T)
    
    def __str__(self):
        return "Softmax"

    
# for older code compatibility
def identity(x):
    return x

def linear(x, a):
    return a * x

def sigmoid(x):
    if x >= 0:
        z = np.exp(-x)
        return 1/(1+z)
    else:
        z = np.exp(x)
        return z/(1+z)

def identity_derivative(x):
    return np.ones_like(x)

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x)**2

def softmax(x):
    exps = np.exp(x - np.max(x))
    return exps / np.sum(exps, axis=0, keepdims=True)
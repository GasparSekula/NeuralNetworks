import numpy as np

class LossFunction:
    def __init__(self, f1_score: bool) -> None:
        self.f1_score = f1_score
    
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:

        raise NotImplementedError("Subclasses should implement this method.")
    
    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Subclasses should implement this method.")
    
    
class LossMSE(LossFunction):
    def __init__(self, f1_score: bool = False) -> None:
        super().__init__(f1_score)
        self.f1_score = f1_score
        
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.mean((y_true - y_pred) ** 2)
    
    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return 2 * (y_pred - y_true) / len(y_true)
    
    def __str__(self):
        return "MSE"
    
    
class LossCrossEntropy(LossFunction):
    def __init__(self, f1_score: bool = True) -> None:
        super().__init__(f1_score)
        self.f1_score = f1_score
    
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]
    
    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return - (y_true / y_pred) + (1 - y_true) / (1 - y_pred)
    
    def __str__(self):
        return "Cross Entropy"

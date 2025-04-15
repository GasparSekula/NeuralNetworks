import numpy as np

    
class Regularization:
    def __init__(self):
        pass
    
    def loss(self):
        raise NotImplementedError
    
    def grad(self):
        raise NotImplementedError
    
class RegularizationNone(Regularization):
    def __init__(self):
        super().__init__()
        
    def loss(self, weights):
        return 0
    
    def grad(self, weights: list) -> list:
        return [np.zeros(w_i.shape) for w_i in weights]
    
class RegularizationL1(Regularization):
    def __init__(self, lambd: float = 0.5):
        super().__init__()
        self.lambd = lambd
        
    def loss(self, weights):
        return self.lambd * np.sum([np.sum(np.abs(w_i)) for w_i in weights])
    
    def grad(self, weights):
        return [self.lambd * np.sign(w_i) for w_i in weights] 
    
class RegularizationL2(Regularization):
    def __init__(self, lambd: float = 0.5):
        super().__init__()
        self.lambd = lambd
        
    def loss(self, weights):
        return self.lambd * np.sum([np.sum(w_i ** 2) for w_i in weights])
    
    def grad(self, weights):
        return [self.lambd * 2 * w_i for w_i in weights]
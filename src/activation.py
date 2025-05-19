import numpy as np

class ActivateFunction:
    def __init__(self):
        pass
        
        
class ReLU(ActivateFunction):
    def __init__(self):
        super().__init__()

    def activate(self, mat):
        return np.maximum(mat, 0)
    
    def derivative(self, mat):
        return (mat > 0).astype(float)




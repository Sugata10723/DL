import numpy as np
class ActivateFunction:
    def __init__(self):
        pass

class Identity(ActivateFunction):
    def __init__(self):
        super().__init__()

    def activate(self, mat):
        return mat
    
    def derivative(self, mat):
        return np.ones((mat.shape[0], mat.shape[1]))
        
        
class ReLU(ActivateFunction):
    def __init__(self):
        super().__init__()

    def activate(self, mat):
        return np.maximum(mat, 0)
    
    def derivative(self, mat):
        return (mat > 0).astype(float)
    
class Softmax(ActivateFunction):
    def __init__(self):
        super().__init__()

    def activate(self, mat):
        '''
        Input
            - mat: ndarray
        Output
            - probs: ndarray
        '''
        shift = mat - np.max(mat, axis=1, keepdims=True) # 1) 数値安定化のために各行からその行の最大値を引く
        exp_scores = np.exp(shift)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        
        return probs







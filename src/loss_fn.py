import numpy as np

class LossFunction:
    def __init__(self):
        pass

class MSE(LossFunction):
    def __init__(self):
        super().__init__()
    
    def calc_loss(self, ground_truth, pred):
        return np.sum((ground_truth - pred)**2)/2 
    
    def calc_delta(self, pred, ground_truth):
        return pred - ground_truth

class CrossEntropy(LossFunction):
    def __init__(self):
        super().__init__()

    def calc_loss(self, ground_truth, pred):
        '''
        Input 
            - ground_truh: ndarray
            - pred: ndarray
        '''
        loss = 0
        for i in range(len(pred)):
            loss += np.dot(ground_truth[i], pred[i])
        return loss
    
    def calc_delta(self, pred, ground_truth):
        return pred - ground_truth 

import numpy as np
import copy
from propagation import Propagation
import minibatch

class BackPropagation:
    ''' Computate '''
    def __init__(self, layers, activation, last_activation, loss_fn):
        '''
        Input:
            - layers: tuple
            - activation: class
            - last_activation: class
            - loss_fn: class
        '''
        self.layers = layers # 最終レイヤー=出力
        self.num_layer = len(layers)
        self.activation = activation 
        self.last_activation = last_activation
        self.loss_fn = loss_fn

        self.propagation = Propagation(layers=self.layers, activation=self.activation, last_activation=self.last_activation, loss_fn=self.loss_fn)
        self.best_weights = None
        self.best_biases = None


    def _gen_first_weights(self, data):
        '''重さの初期値を生成する関数'''
        weights = []
        weights.append((np.random.random((data.shape[1], self.layers[0]))))
        for i in range(1, self.num_layer):
            weights.append((np.random.random((self.layers[i-1], self.layers[i]))))
        return weights
    
    def _gen_first_biases(self):
        '''バイアスの初期値を生成する関数'''
        biases = []
        for i in range(self.num_layer):
            biases.append(((np.random.random(self.layers[i]))))
        return biases


    def fit(self, data, ground_truth, step_size=1e-4, threshold=1e-1, max_itr=1e+5):
        '''学習を行う関数
        Input:
            - data: ndarray
            - ground_truth: array
            - step_size: float
            - threshold: float
            - max_itr: int
        Output:
            - weights: list[ndarray]
            - biases: list[ndarray]
        '''
        loss = float("inf")
        itr = 0
        weights = self._gen_first_weights(data)
        biases = self._gen_first_biases()
        Xb, yb = minibatch.minibatch(data, ground_truth)

        while loss > threshold:
            print(f"\rEpoch: {itr}, loss: {loss:.4f}", end="", flush=True)

            if itr == max_itr:
                print("\n最大Epoch数に到達しました。")
                break

            for i in range(len(Xb)):
                loss, weights, biases = self._recursive_change_param(Xb[i], yb[i], weights, biases, step_size)
            
            itr += 1

        self.best_weights = weights
        self.best_biases = biases
        
    
    def _recursive_change_param(self, data, ground_truth, weights, biases, step_size):
        '''重みとバイアスを更新する関数
        Input:
            - data: list[ndarray]
            - ground_truth: array
            - weights: list[ndarray]
            - biases: list[array]
            - step_size: float
        Output:
            - loss: float
            - weights: list[ndarray]
            - biases: list[array]
        '''
        data = data.copy()
        n = data.shape[0]
        weights = copy.deepcopy(weights)
        biases = copy.deepcopy(biases)

        deltas = self._calc_deltas(data, ground_truth, weights, biases) 
        pred, mid_u, mid_z = self.propagation.propagate(data, weights, biases)
        mid_z.insert(0, data)
        
        for i in range(self.num_layer):
            diff_weight = -step_size * (mid_z[i].T @ deltas[i]) / n
            diff_bias = -step_size * (deltas[i].sum(axis=0)) / n
            weights[i] += diff_weight
            biases[i] += diff_bias
        
        pred, _, _ = self.propagation.propagate(data, weights, biases)
        loss = self.loss_fn.calc_loss(pred=pred, ground_truth=ground_truth)
        
        return loss, weights, biases
    
    def _calc_deltas(self, data, ground_truth, weights, biases):
        '''各レイヤー、ユニットのdeltaを計算する関数
        Input:
            - data: ndarray
            - ground_truth: array
            - weights: list[matrix]
            - biases: list[vector]
        Output:
            - deltas: list[vector]
        '''
        pred, mid_u, mid_z = self.propagation.propagate(data, weights, biases)
        deltas = [self.loss_fn.calc_delta(pred=pred, ground_truth=ground_truth)] # 出力層でのdelta
        
        for i in range(1, self.num_layer): 
            pre_weight = weights[-i]
            pre_delta = deltas[-i]
            delta_l = self.activation.derivative(mid_u[-(i+1)]) * (pre_delta @ pre_weight.T)
            deltas.insert(0, delta_l) 

        return deltas
    
    def predict(self, data):
        if self.best_weights is None or self.best_biases is None:
            raise RuntimeError (
                '学習が済んでいません'
            )
        else:
            pred, _, _ = self.propagation.propagate(data, self.best_weights, self.best_biases)
        return pred

    
    

        


import numpy as np
import copy
import activation
import loss_fn
from activation import ReLU
from loss_fn import MSE
from propagation import Propagation

class BackPropagation:
    ''' Computate '''
    def __init__(self, data, ground_truth, layers, loss_fn, activation):
        '''
        Input:
            - data: matrix(d, n)
            - ground_truth: vector 
            - layers: list[]
        '''
        self.data = data
        self.ground_truth = ground_truth
        self.layers = layers # 最終レイヤー=出力
        self.num_layer = len(layers)
        self.loss_fn = loss_fn
        self.activation = activation 
        self.propagation = Propagation(layers=self.layers, activation=self.activation, loss_fn=self.loss_fn)
        self.best_weights = None
        self.best_biases = None


    def _gen_first_weights(self):
        '''重さの初期値を生成する関数'''
        weights = []
        weights.append((np.random.random((self.data.shape[1], self.layers[0]))))
        for i in range(1, self.num_layer):
            weights.append((np.random.random((self.layers[i-1], self.layers[i]))))
        return weights
    
    def _gen_first_biases(self):
        '''バイアスの初期値を生成する関数'''
        biases = []
        for i in range(self.num_layer):
            biases.append(((np.random.random(self.layers[i]))))
        return biases


    def fit(self, step_size=2, threshold=1, max_itr=100):
        '''学習を行う関数
        Input:
            - step_size: float
            - threshold: float
            - max_itr: int
        Output:
            - weights: list[ndarray]
            - biases: list[ndarray]
        '''
        loss = float("inf")
        itr = 0
        weights = self._gen_first_weights()
        biases = self._gen_first_biases()

        while loss > threshold:
            print(f"\rEpoch: {itr}, loss: {loss:.4f}", end="", flush=True)
            if itr == max_itr:
                print("\n最大エポック数に到達しました。")
                break
            _loss, _weights, _biases = self._recursive_change_param(weights, biases, step_size)
            loss = _loss
            weights = _weights
            biases = _biases
            itr += 1
        self.best_weights = weights
        self.best_biases = biases
        
    
    def _recursive_change_param(self, weights, biases, step_size):
        '''重みとバイアスを更新する関数
        Input:
            - weights: list[ndarray]
            - biases: list[array]
            - step_size: float
        Output:
            - loss: float
            - weights: list[ndarray]
            - biases: list[array]
        '''
        n = self.data.shape[0]
        weights = copy.deepcopy(weights)
        biases = copy.deepcopy(biases)
        deltas = self._calc_deltas(weights, biases)
        pred, mid_u, mid_z = self.propagation.propagate(self.data, weights, biases)
        mid_z.insert(0, self.data)
        
        for i in range(self.num_layer):
            diff_weight = -step_size * (mid_z[i].T @ deltas[i]) / n
            diff_bias = -step_size * (deltas[i].sum(axis=0)) / n
            weights[i] += diff_weight
            biases[i] += diff_bias
        
        pred, _, _ = self.propagation.propagate(self.data, weights, biases)
        loss = self.loss_fn.calc_loss(pred=pred, ground_truth=self.ground_truth)
        
        return loss, weights, biases
    
    def _calc_deltas(self, weights, biases):
        '''各レイヤー、ユニットのdeltaを計算する関数
        Input:
            - weights: list[matrix]
            - biases: list[vector]
        Output:
            - deltas: list[vector]
        '''
        pred, mid_u, mid_z = self.propagation.propagate(self.data, weights, biases)
        deltas = [self.loss_fn.calc_delta(pred=pred, ground_truth=self.ground_truth)] # 出力層でのdelta
        
        for i in range(1, self.num_layer): 
            pre_weight = weights[-i]
            pre_delta = deltas[-i]
            delta_l = self.activation.derivative(mid_u[-(i+1)]) * (pre_delta @ pre_weight.T)
            deltas.insert(0, delta_l) 

        return deltas
    
    def predict(self, data):
        if self.best_weights is None or self.best_biases is None:
            print("\n学習が済んでいません")
        else:
            pred, _, _ = self.propagation.propagate(data, self.best_weights, self.best_biases)
        return pred

    
    

        


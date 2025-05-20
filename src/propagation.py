class Propagation:
    ''' 
    Computation for Propagation.
    Input
        data: ndarray(n, d)
        layers: tuple
        weights: list[ndarray]
        biases: list[array]
    Output
        matrix: matrix
        loss: float
    '''
    def __init__(self, layers, activation, last_activation, loss_fn):
        '''
        input:
            weights: list[ndarray]
            biases: list[array]
            layers: tuple
            active_func: class
            loss_fn: class
        '''
        self.layers = layers # 出力層も含む
        self.num_layer = len(layers)
        self.activation = activation
        self.last_activation = last_activation
        self.loss_fn = loss_fn

    def _multiple(self, mat_z, weight, bias):
        mat_u = mat_z @ weight +  bias # zt = xt @ wt 
        return mat_u       

    def propagate(self, data, weights, biases):
        mat_z = data.copy()
        mid_u = []
        mid_z = []
        for i in range(self.num_layer): 
            mat_u = self._multiple(mat_z, weights[i], biases[i]) 
            if i == self.num_layer - 1:
                mat_z = self.last_activation.activate(mat_u)
            else:
                mat_z = self.activation.activate(mat_u)
            mid_u.append(mat_u)
            mid_z.append(mat_z)
        pred = mat_z
        return pred, mid_u, mid_z
    
    def calc_loss(self, pred, ground_truth):
        return self.loss_fn.calc_loss(pred=pred, ground_truth=ground_truth)
    
import numpy as np
import sys
sys.path.append("./src/")

from propagation import Propagation
from backpropagation import BackPropagation
import activation
import loss_fn
from activation import ReLU
from loss_fn import LSM
from propagation import Propagation

data = np.array([
    [1, 1, 1, 1],
    [0, 0, 1, 0],
    [0, 0, 0, 9]
])
ground_truth = np.array([[1], [0], [1]])
layers = [3, 2, 1]

fitter = BackPropagation(data=data,  ground_truth=ground_truth, layers=layers, activation=ReLU(), loss_fn=LSM())
model = Propagation(layers=layers, activation=ReLU(), loss_fn=LSM())

loss, weights, biases = fitter.fit(step_size = 1, threshold=0.5)
pred, _, _ = model.propagate(data=data, weights=weights, biases=biases)

loss = model.calc_loss(ground_truth=ground_truth, pred=pred)
print(f"Loss: {loss}")
print(f"\npred: {pred}")



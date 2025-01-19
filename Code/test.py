import numpy as np
def val(inputs):
    exp_values = np.exp(inputs - np.max(inputs, keepdims=True))
    probabilities = exp_values / np.sum(exp_values, keepdims=True)
    print(probabilities)
val([1, 2])
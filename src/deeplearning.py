# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.
"""Softmax."""

import numpy as np

    

def softmax(x):
    """Compute softmax values for x."""
    if (x.ndim == 1):
        return softmax1d(x)
    else:
        result = np.zeroes(x.shape)
        for index in 0, x.shape[0]:
            result[index] = softmax(x[index])
        return result;

def softmax1d(x):
    result = np.exp(x)
    total = np.sum(result)
    for index in range(len(result)):
        result[index] /= total
    return result;


scores = np.array([3.0, 1.0, 0.2])
scores2 = np.array([[1, 2, 3, 6],
                   [2, 4, 5, 6],
                   [3, 8, 7, 6]])
               
print (softmax(scores))
print (softmax(scores2))

 
# Plot softmax curves
#import matplotlib.pyplot as plt
#x = np.arange(-2.0, 6.0, 0.1)
#scores = np.vstack([x, np.ones_like(x), 0.2 * np.ones_like(x)])

#plt.plot(x, softmax(scores).T, linewidth=2)
#plot.show()

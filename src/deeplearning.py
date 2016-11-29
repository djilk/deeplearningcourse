# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.
"""Softmax."""

scores = [3.0, 1.0, 0.2]

import numpy as np

def softmax(x):
    """Compute softmax values for x."""

    result = []
    sum = 0
    for index in range(len(x)):
        result[index] = x[index]*x[index]
        sum += result[index]
        
    for index in range(len(x)):
        result[index] /= sum
    
    return result;
    #pass # TODO: Compute and return softmax(x)


print (softmax(scores))

# Plot softmax curves
#import matplotlib.pyplot as plt
#x = np.arange(-2.0, 6.0, 0.1)
#scores = np.vstack([x, np.ones_like(x), 0.2 * np.ones_like(x)])

#plt.plot(x, softmax(scores).T, linewidth=2)
#plot.show()

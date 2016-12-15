# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.

# The goal of this is to create an "averaged histogram" function
# but first do it for the Copper data

import tensorflow as tf

# Bins will be:  (-inf, 1), [1, 2), [2, 3), [3, 4), [4, inf)
nbins = 5
#value_range = [0.0, 5.0]

new_values = tf.constant([-1.0, 0.0, 1.5, 2.0, 5.0, 15], tf.float32)

min = tf.reduce_min(new_values)
max = tf.reduce_max(new_values)
value_range = [max, min]

hist = tf.histogram_fixed_width(new_values, value_range, nbins)
sess = tf.Session()
print(sess.run(hist))
  
  #=> [2, 1, 1, 0, 2]


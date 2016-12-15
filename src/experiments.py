# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.

import tensorflow as tf

# Bins will be:  (-inf, 1), [1, 2), [2, 3), [3, 4), [4, inf)
nbins = 5
value_range = [0.0, 5.0]
new_values = [-1.0, 0.0, 1.5, 2.0, 5.0, 15]
#tf.initialize_all_variables().run()
sess = tf.Session() 
hist = tf.histogram_fixed_width(new_values, value_range, nbins)
print(sess.run(hist))
  
  #=> [2, 1, 1, 0, 2]


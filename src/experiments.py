# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.

# The goal of this is to create an "averaged histogram" function
# but first do it for the Copper data

import tensorflow as tf

# Bins will be:  (-inf, 1), [1, 2), [2, 3), [3, 4), [4, inf)
bins = 100
shifts = 10
#value_range = [0.0, 5.0]

#new_values = tf.constant([-1.0, 0.0, 1.5, 2.0, 5.0, 15], tf.float32)
normal = tf.random_normal([10000], 0.0, 100.0)

data = normal

min = tf.reduce_min(data)
max = tf.reduce_max(data)
range = [max, min]
half_bin = (max - min) / (bins * 2)
shifts = tf.linspace(-half_bin, half_bin, shifts)

def one_hist(accumulator, element):
    hist = tf.histogram_fixed_width(data, tf.add(range, [element, element]), bins)
    tf.add(accumulator, hist)

avg_hist = tf.foldl(one_hist, shifts, tf.zeros([bins]))


#print(sess.run(hist, feed_dict={shift: sess.run(shifts)}))
print(sess.run([avg_hist]))
  
  #=> [2, 1, 1, 0, 2]


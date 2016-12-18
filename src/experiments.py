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
range = [min, max]
half_bin = (max - min) / (bins * 2)
shift_tensor_1d = tf.linspace(-half_bin, half_bin, shifts)
shift_tensor_2d = tf.expand_dims(shift_tensor_1d, 1)
range_tensor_2d = tf.expand_dims(range, 0)
range_elements = tf.add(range_tensor_2d, shift_tensor_2d)
sess = tf.Session()
#print(sess.run([range, shift_tensor_1d, range_elements]))

initial_accumulator = tf.zeros([bins], tf.int32)

def one_hist(accumulator, element):
    #hist = tf.histogram_fixed_width(data, tf.add(range, [element, element]), bins)
    hist = tf.histogram_fixed_width(data, element, bins)
    return tf.add(accumulator, hist)

#sum_hist = tf.foldl(one_hist, shift_tensor_1d, initial_accumulator)
sum_hist = tf.foldl(one_hist, range_elements, initial_accumulator)
total = tf.to_float(tf.reduce_sum(sum_hist))
avg_hist = tf.div(tf.to_float(sum_hist), total)

#tf.initialize_all_variables().run()

print(sess.run([sum_hist, avg_hist]))
#print (sess.run(shift_tensor))
  
  #=> [2, 1, 1, 0, 2]


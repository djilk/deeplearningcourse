# This is an extremely simple TensorFlow graph and execution
# It adds two matrices and either provides a slice or a transpose of the result

# Might be helpful to show the exact same process with just numpy

import tensorflow as tf

first = tf.constant([[1, 2], [3, 4]], tf.float32)
add = tf.ones([2, 2], tf.float32)
sum = tf.add(first, add)
sumslice = tf.slice(sum, [0, 0], [2, 1])
transpose = tf.transpose(sum)
sess = tf.Session()
print (sess.run(transpose))
print (sess.run(sumslice))

# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.

import tensorflow as tf

first = tf.constant([[1, 2], [3, 4]], tf.float32)
add = tf.ones([2, 2], tf.float32)
sum = tf.add(first, add)
sumslice = tf.slice(sum, [0, 0], [1, 0])
printer = tf.Print(tf.ones([1]), [sumslice], "Output:", summarize=30)
sess = tf.Session()
sess.run(printer)



# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.

import tensorflow as tf

list = []
first = tf.constant([[1, 2], [3, 4]])
add = tf.ones([2, 2])
sum = tf.add(first, add)
list.append(sum)
printer = tf.Print(tf.ones([1]), list, "Output:")
sess = tf.Session()
sess.run(printer)



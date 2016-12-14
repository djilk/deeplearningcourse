# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.

import tensorflow as tf

list = []
list.append(tf.linspace(10.0, 12.0, 3))
print(list)
printer = tf.Print(tf.ones([1], tf.int32), list)
sess = tf.Session()
sess.run(printer)



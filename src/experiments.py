# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.

import tensorflow as tf

list = [[1]]
list.append(tf.linspace(10.0, 12.0, 3))
result = tf.Print([1], list)




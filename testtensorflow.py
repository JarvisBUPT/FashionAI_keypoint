import tensorflow as tf
import os
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
c = tf.constant(4.0)
print(c.graph is tf.get_default_graph())
w = tf.Variable(tf.random_normal([5, 2], stddev=0.01))
state = tf.Variable(0, name='counter')

one = tf.constant(1)
new_value = tf.add(state, one)
update = tf.assign(state, new_value)
init = tf.global_variables_initializer()
tf.Tensor
tf.GraphKeys.TRAINABLE_VARIABLES
tf.contrib.framework.local_variable
inputs = np.arange(48).reshape(2, 2, 3, 4)
print(inputs)
pad1 = tf.pad(inputs, [[1, 0], [0, 0], [0, 0], [0, 0]], name='pad_1')
print(pad1)

with tf.Session() as sess:
    sess.run(init)

    print(sess.run(pad1))

    print(tf.get_default_graph().get_all_collection_keys())

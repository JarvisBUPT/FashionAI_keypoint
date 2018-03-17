import tensorflow as tf

c = tf.constant(4.0)
print(c.graph is tf.get_default_graph())
w=tf.Variable(tf.random_normal([784, 10], stddev=0.01))
tf.Tensor
tf.GraphKeys.TRAINABLE_VARIABLES
tf.contrib.framework.local_variable
pad1 = tf.pad(inputs, [[0, 0], [2, 2], [2, 2], [0, 0]], name='pad_1')
print(pad1)
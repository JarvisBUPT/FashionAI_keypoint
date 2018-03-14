import tensorflow as tf

c = tf.constant(4.0)
print(c.graph is tf.get_default_graph())
w=tf.Variable(tf.random_normal([784, 10], stddev=0.01))
tf.Tensor

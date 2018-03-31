import tensorflow as tf
import os
import numpy as np
from tensorflow.contrib.layers.python.layers.initializers import xavier_initializer
from processconfig import process_config_clothes
# import pwd

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
c = tf.constant(4.0)
# print(c.graph is tf.get_default_graph())
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
inputs = tf.cast(inputs, dtype=tf.float32)
print(inputs)
# pad1 = tf.pad(inputs, [[1, 0], [0, 0], [0, 0], [0, 0]], name='pad_1')
# print(pad1)
kernel = tf.Variable(xavier_initializer(uniform=False)([1, 1, inputs.get_shape().as_list()[3], 8]), name='weights')
print('kernel', kernel)
conv = tf.nn.conv2d(inputs, kernel, [1, 2, 2, 1], padding='VALID', data_format='NHWC')
print('conv', conv)
# tf.GraphKeys.SUMMARIES = "summaries"
x = tf.constant(np.arange(24).reshape((2, 3, 4)), dtype=tf.float32)
print(x)
y = tf.constant(np.arange(24, 48).reshape((2, 3, 4)), dtype=tf.float32)
z = tf.constant(np.arange(48, 72).reshape((2, 3, 4)), dtype=tf.float32)
cc = [x, y, z]
stack = tf.stack([x], axis=1)
params = process_config_clothes()
logdir_train = os.path.join(os.getcwd(), params['log_dir_train'])
print(os.getcwd())
print(logdir_train)
logdir_test = params['log_dir_test']

loss = ...
tf.summary.scalar("loss", loss)
merged_summary = tf.summary.merge_all()

init = tf.global_variable_initializer()

with tf.Session() as sess:

    print(sess.run(x))
    s = sess.run(stack)
    train_summary = tf.summary.FileWriter(logdir_train, tf.get_default_graph())
    test_summary = tf.summary.FileWriter(logdir_test)
    sess.run(init)
    for i in range(100):
        _, summary = sess.run([x, merged_summary])
        test_summary.add_summary(summary, i)
    # train_summary.add_summary(s)
    # train_summary.flush()
    # # print(sess.run(pad1))
    # print(sess.run(conv))

    # print(tf.get_default_graph().get_all_collection_keys())

import tensorflow as tf
import os
from tensorflow.examples.tutorials.mnist import input_data
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# 超参数参数定义
learning_rate = 0.01
training_epoch = 25
batch_size = 100
display_step = 1
# 1,定义模型输入，建图和损失
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])
#  变量定义
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
#  计算预测值即输出
pred = tf.nn.softmax(tf.matmul(x, W) + b)

#  定义损失值 使用交叉熵函数计算损失值
cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(pred) + (1 - y) * tf.log(1 - pred), axis=1))

# 2,定义优化器，最小化cost值
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# 3,创建测试模型
correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))  # 预测值与真实值进行对比是否一样
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 4,初始化所有变量值
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)  # 运行初始化
    for epoch in range(training_epoch):
        avg_cost = 0.  # 存储平均cost值
        iteration = int(mnist.train.num_examples / batch_size)  # 计算一个周期需要的迭代次数
        for i in range(iteration):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)   # 取下一个迭代的真实输入值和输出值
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs, y: batch_ys})  # 在会话中运行模型
            avg_cost += c / iteration  # 计算每个迭代平均损失值
        if (epoch + 1) % display_step == 0:
            print("Epoch:", '%02d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))  # 打印一个周期训练后的损失值
    print("Optimization Finished!")  # 模型训练完成


    #  计算测试结果
    print("Accuracy:", accuracy.eval({x: mnist.test.images[:3000], y: mnist.test.labels[:3000]}))

"""2018.11.20.Tue"""

import tensorflow as tf

# Multi-variable linear regression x1, x2, x3, y
# using MATRIX
# Matrix 형태로 데이터를 표현
x_data = [[73., 80., 73.], [93., 88., 93.], [89., 91., 90.], [96., 98., 100.], [73., 66., 70.]]

y_data = [[152.], [185.], [180.], [196.], [142.]]

# placeholder --> shape 내용을 Matrix 크기에 맞게 수정
X = tf.placeholder(tf.float32, shape=[None, 3])
Y = tf.placeholder(tf.float32, shape=[None, 1])

# Weight
W = tf.Variable(tf.random_normal([3, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# Hypothesis H(x) = x1 * w1 + x2 * w2 + x3 * w3 + b
hypothesis = tf.matmul(X, W) + b

# cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Minimize cost
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

# Session
sess = tf.Session()

# Initialize
sess.run(tf.global_variables_initializer())

for step in range(3001):
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train], feed_dict={X: x_data, Y: y_data})
    if step % 10 == 0:
        print(step, "Cost: ", cost_val, "\nPrediction:\n", hy_val)
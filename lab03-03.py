# linear regression & gradient descent algorithm
import tensorflow as tf

X = [1, 2, 3]
Y = [1, 2, 3]

W = tf.Variable(-3.0)

hypothesis = W * X

cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Cost Minimize: Gradient Descent using W -= Learning_rate * derivative
# Gradient Descent Algorithm 공식에 맞춰 그대로 작성한 코드
# learning_rate = 0.1
# gradient = tf.reduce_mean((W * X - Y) * X)
# descent = W - learning_rate * gradient
# update = W.assign(descent)

# Minimize: Gradient Descent Magic --> 위의 공식 코드를 일일이 작성할 필요없이
# tensorflow에서 제공하는 GradientDescentOptimizer() 이용
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train = optimizer.minimize(cost)

# 세션 생성
sess = tf.Session()
# variables 초기화
sess.run(tf.global_variables_initializer())

for step in range(100):
    print(step, sess.run(W))
    sess.run(train)

# 실행결과 --> cost 값을 최소로 갖는 W의 값을 찾는 것으로 본 예제에서는 W 값이 1.0으로 수렴할 수록 학습결과가 좋은 것을 의미한다.
# 0 5.0
# 1 1.2666664
# 2 1.0177778
# 3 1.0011852
# 4 1.000079
# 5 1.0000052
# 6 1.0000004
# 7 1.0
# 8 1.0
# 9 1.0
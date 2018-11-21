# linear regression
import tensorflow as tf

# 1. Build graph using TF operations ==> 트레이닝을 위한 그래프 생성(구현)
# X and Y data
x_train = [1, 2, 3]
y_train = [1, 2, 3]

# W, b tensor(node) 정의
W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# Hypothesis H(x) = Wx + b
hypothesis = W * x_train + b

# cost(loss) function ==> tensorflow에서 표현하는 cost function
cost = tf.reduce_mean(tf.square(hypothesis - y_train))

# Minimize using GradientDescentOptimizer function
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

# 2. Run/update graph and get results ==> 그래프 구현 후, 세션을 만든다.
# Create Session
sess = tf.Session()
# Initializes global variables in the graph ==> variable을 이용하기 전에 반드시 초기화 해야 함!
sess.run(tf.global_variables_initializer())

# Fit the line
for step in range(2001):
    sess.run(train) # node train을 2000번 실행
    if step % 20 == 0:
        print(step, sess.run(cost), sess.run(W), sess.run(b))


# 실행결과
# [step]  [cost]            [W]              [b]
# 0     9.821513        [0.41136214]    [-1.9195802]
# 20    0.3387649       [1.4689769]     [-1.37629]
# 40    0.22986741      [1.5450078]     [-1.2684674]
# 60    0.20806392      [1.5287328]     [-1.2047464]
# 80    0.18896051      [1.5047735]     [-1.1477365]
# 100   0.17161687      [1.4811357]     [-1.0937599]
# 120   0.15586512      [1.4585321]     [-1.0423535]
# 140   0.14155923      [1.4369835]     [-0.9933663]
# 160   0.12856643      [1.4164469]     [-0.94668174]
# 180   0.11676603      [1.3968754]     [-0.90219116]
# 200   0.10604879      [1.3782237]     [-0.85979134]
# 220   0.09631518      [1.3604485]     [-0.81938434]
# 240   0.08747495      [1.3435087]     [-0.78087616]
# 260   0.07944616      [1.3273653]     [-0.7441779]
# 280   0.07215426      [1.3119801]     [-0.70920426]
# ...
# 1800  4.793376e-05    [1.008041]      [-0.01827919]
# 1820  4.353312e-05    [1.0076631]     [-0.01742013]
# 1840  3.953809e-05    [1.007303]      [-0.01660144]
# 1860  3.5909394e-05   [1.0069599]     [-0.01582129]
# 1880  3.261354e-05    [1.0066328]     [-0.01507784]
# 1900  2.962056e-05    [1.0063211]     [-0.01436926]
# 1920  2.6901775e-05   [1.006024]      [-0.01369398]
# 1940  2.4432053e-05   [1.005741]      [-0.01305046]
# 1960  2.2190006e-05   [1.0054711]     [-0.01243714]
# 1980  2.0153193e-05   [1.0052141]     [-0.01185267]
# 2000  1.8303781e-05   [1.0049691]     [-0.0112957]
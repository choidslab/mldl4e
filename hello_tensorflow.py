# tensorflow 시작
import tensorflow as tf

print(tf.__version__)

# hello라는 node를 만든다.
hello = tf.constant("Hello Tensorflow!")

# 텐서플로우에서는 Session을 만들어줘야 한다.
sess = tf.Session()

# hello라는 node를 실행한다.(run)
print(sess.run(hello))

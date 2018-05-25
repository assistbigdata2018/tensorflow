import tensorflow as tf
import random

from tensorflow.examples.tutorials.mnist import input_data

tf.set_random_seed(777)  # reproducibility

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
#X = tf.placeholder(tf.float32, [None, 784])

sess=tf.Session()
#First let's load meta graph and restore weights
saver = tf.train.import_meta_graph('./saved/lab-11-1-mnist_cnn.meta')
saver.restore(sess,tf.train.latest_checkpoint('./saved/'))

# Now, let's access and create placeholders variables and
# create feed-dict to feed new data
graph = tf.get_default_graph()
X = graph.get_tensor_by_name("X:0")

#Now, access the op that you want to run.
op_to_restore = graph.get_tensor_by_name("op_to_restore:0")

cnt = 15
print("[Label] [Prediction]")

for epoch in range(cnt):
    r = random.randint(0, mnist.test.num_examples - 1)
    print(sess.run(tf.argmax(mnist.test.labels[r:r + 1], 1)), "   ", sess.run(tf.argmax(op_to_restore, 1), feed_dict={X: mnist.test.images[r:r + 1]}))

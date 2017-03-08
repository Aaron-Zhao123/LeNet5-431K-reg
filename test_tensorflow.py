import tensorflow as tf

a = tf.random_uniform([2,3], minval = 0, maxval = 2,dtype = tf.int32)
b = tf.random_uniform(a.get_shape())

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    print(a.eval())
    print(a.eval())
    print(b.eval())

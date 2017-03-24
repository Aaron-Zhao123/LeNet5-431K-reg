from __future__ import print_function

# Import MNIST data
import sys
import getopt
import input_data
import os.path
import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pickle
import scipy.io as sio

class Usage(Exception):
    def __init__ (self,msg):
        self.msg = msg

# Parameters
training_epochs = 500
batch_size = 128
display_step = 1

# Network Parameters
IMAGE_SIZE = 28
NUM_CHANNELS = 1
NUM_LABELS = 10

n_hidden_1 = 300# 1st layer number of features
n_hidden_2 = 100# 2nd layer number of features
n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)
dropout = 0.5


# Store layers weight & bias
# def initialize_tf_variables(first_time_training):
#     if (first_time_training):
def initialize_variables(dir_name, file_name):
    with open(dir_name + file_name,'rb') as f:
        wc1, wc2, wd1, out, bc1, bc2, bd1, bout = pickle.load(f)
    weights = {
        # 5x5 conv, 1 input, 32 outputs
        'cov1': tf.Variable(wc1),
        # 5x5 conv, 32 inputs, 64 outputs
        'cov2': tf.Variable(wc2),
        # fully connected, 7*7*64 inputs, 1024 outputs
        'fc1': tf.Variable(wd1),
        # 1024 inputs, 10 outputs (class prediction)
        'fc2': tf.Variable(out)
    }

    biases = {
        'cov1': tf.Variable(bc1),
        'cov2': tf.Variable(bc2),
        'fc1': tf.Variable(bd1),
        'fc2': tf.Variable(bout)
    }
    return (weights, biases)

# Create model
def conv_network(x, weights, biases, keep_prob):
    conv = tf.nn.conv2d(x,
                        weights['cov1'],
                        strides = [1,1,1,1],
                        padding = 'VALID')
    relu = tf.nn.relu(tf.nn.bias_add(conv, biases['cov1']))
    l1_cov1 = tf.reduce_sum(tf.abs(weights['cov1']))
    l2_cov1 = tf.nn.l2_loss(weights['cov1'])
    pool = tf.nn.max_pool(
            relu,
            ksize = [1 ,2 ,2 ,1],
            strides = [1, 2, 2, 1],
            padding = 'VALID')

    conv = tf.nn.conv2d(pool,
                        weights['cov2'],
                        strides = [1,1,1,1],
                        padding = 'VALID')
    relu = tf.nn.relu(tf.nn.bias_add(conv, biases['cov2']))
    l1_cov2 = tf.reduce_sum(tf.abs(weights['cov2']))
    l2_cov2 = tf.nn.l2_loss(weights['cov2'])
    pool = tf.nn.max_pool(
            relu,
            ksize = [1 ,2 ,2 ,1],
            strides = [1, 2, 2, 1],
            padding = 'VALID')
    '''get pool shape'''
    pool_shape = pool.get_shape().as_list()
    reshape = tf.reshape(
        pool,
        [-1, pool_shape[1]*pool_shape[2]*pool_shape[3]])
    hidden = tf.nn.relu(tf.matmul(reshape, weights['fc1']) + biases['fc1'])
    l1_fc1 = tf.reduce_sum(tf.abs(weights['fc1']))
    l2_fc1 = tf.nn.l2_loss(weights['fc1'])
    hidden = tf.nn.dropout(hidden, keep_prob)
    output = tf.matmul(hidden, weights['fc2']) + biases['fc2']
    l1_fc2 = tf.reduce_sum(tf.abs(weights['fc2']))
    l2_fc2 = tf.nn.l2_loss(weights['fc2'])
    l1 = l1_cov1 + l1_cov2 + l1_fc1 + l1_fc2
    l2 = l2_cov1 + l2_cov2 + l2_fc1 + l2_fc2
    return output , reshape, l1, l2

def calculate_non_zero_weights(weight):
    count = (weight != 0).sum()
    size = len(weight.flatten())
    return (count,size)

'''
Prune weights, weights that has absolute value lower than the
threshold is set to 0
'''

'''
mask gradients, for weights that are pruned, stop its backprop
'''
def mask_gradients(weights, grads_and_names, weight_masks, biases, biases_mask):
    new_grads = []
    keys = ['cov1','cov2','fc1','fc2']
    for grad, var_name in grads_and_names:
        # flag set if found a match
        flag = 0
        index = 0
        for key in keys:
            if (weights[key]== var_name):
                # print(key, weights[key].name, var_name)
                mask = weight_masks[key]
                new_grads.append((tf.multiply(tf.constant(mask, dtype = tf.float32),grad),var_name))
                flag = 1
            if (biases[key] == var_name):
                mask = biases_mask[key]
                new_grads.append((tf.multiply(tf.constant(mask, dtype = tf.float32),grad),var_name))
                flag = 1
        # if flag is not set
        if (flag == 0):
            new_grads.append((grad,var_name))
    return new_grads

def ClipIfNotNone(grad):
    if grad is None:
        return grad
    return tf.clip_by_value(grad, -1, 1)

'''
Define a training strategy
'''
def main(argv = None):
    learning_rate = 1e-4
    mnist = input_data.read_data_sets("MNIST.data/", one_hot = True)
    # tf Graph input
    x = tf.placeholder("float", [None, n_input])
    y = tf.placeholder("float", [None, n_classes])

    keep_prob = tf.placeholder(tf.float32)
    keys = ['cov1','cov2','fc1','fc2']

    x_image = tf.reshape(x,[-1,28,28,1])
    # dir_name = '/Users/aaron/Projects/Mphil_project/tmp_LeNet5_reg/dropout_pretrain_LeNet5431K/'
    dir_name = '/Users/aaron/Projects/Mphil_project/tmp_LeNet5_reg/norm1/'
    file_name = 'tmp0.pkl'
    (weights, biases) = initialize_variables(dir_name, file_name)
    # Construct model
    pred, pool, l1, l2= conv_network(x_image, weights, biases, keep_prob)
    lambda_1 = 0.00001
    lambda_2 = 0.0005



    correct_prediction = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    merged = tf.merge_all_summaries()
    saver = tf.train.Saver()
    init = tf.initialize_all_variables()
    # print('dumping weights...')
    # save_file_name = '/Users/aaron/Projects/Mphil_project/tmp_LeNet5_reg/mat_file/shakeout_prob5'
    # dump_weights(dir_name + file_name, save_file_name)
    # Launch the graph
    with tf.Session() as sess:
        sess.run(init)
        keys = ['cov1','cov2','fc1','fc2']

        # Calculate accuracy
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        test_accuracy = accuracy.eval({x: mnist.test.images, y: mnist.test.labels, keep_prob : 1.0})
        print("Accuracy:", test_accuracy)
        # with open('acc_log_10.txt','a') as f:
        #     f.write(str(test_accuracy)+'\n')

def mask_gen(open_file_name):
    #generate mask based on weights
    with open(open_file_name,'rb') as f:
        wc1, wc2, wd1, out, bc1, bc2, bd1, bout = pickle.load(f)
    # print(wc1)
    weights = {
        # 5x5 conv, 1 input, 32 outputs
        'cov1': wc1,
        # 5x5 conv, 32 inputs, 64 outputs
        'cov2': wc2,
        # fully connected, 7*7*64 inputs, 1024 outputs
        'fc1': wd1,
        # 1024 inputs, 10 outputs (class prediction)
        'fc2': out
    }

    biases = {
        'cov1': bc1,
        'cov2': bc2,
        'fc1': bd1,
        'fc2': bout
    }
    keys = ['cov1', 'cov2', 'fc1', 'fc2']
    masks = {}
    b_masks = {}
    for key in keys:
        masks[key] = weights[key] != 0
        b_masks[key] = biases[key] != 0
    with open('mask.pkl', 'wb') as f:
        pickle.dump((masks, b_masks),f)

def dump_weights(open_file_name, save_file_name):
    #generate mask based on weights
    with open(open_file_name,'rb') as f:
        wc1, wc2, wd1, out, bc1, bc2, bd1, bout = pickle.load(f)
    # print(wc1)
    weights = {
        # 5x5 conv, 1 input, 32 outputs
        'cov1': wc1,
        # 5x5 conv, 32 inputs, 64 outputs
        'cov2': wc2,
        # fully connected, 7*7*64 inputs, 1024 outputs
        'fc1': wd1,
        # 1024 inputs, 10 outputs (class prediction)
        'fc2': out
    }

    biases = {
        'cov1': bc1,
        'cov2': bc2,
        'fc1': bd1,
        'fc2': bout
    }
    keys = ['cov1', 'cov2', 'fc1', 'fc2']
    print("try dumping weights")
    sio.savemat(save_file_name+'.mat',
                {'weights':weights})


if __name__ == '__main__':
    sys.exit(main())

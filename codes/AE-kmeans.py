# 该程序主要实现能够提取有效特征表示的自编码器（特征映射维数为3）的训练
# 随后利用 k-means 算法实现对 MNIST 数据集中 0, 1 图像已提取特征表示进行聚类
# 参考程序：https://github.com/jswanglp/MyML/blob/master/codes/Neural_network_models/Unsupervised_learning_models/AE.py
# coding: utf-8

import os, sys 
import numpy as np 
import tensorflow as tf 
import matplotlib.pyplot as plt 
from scipy.cluster.vq import kmeans2 
from mpl_toolkits.mplot3d import Axes3D 
from tensorflow.examples.tutorials.mnist import input_data

tf.logging.set_verbosity(tf.logging.ERROR)

# 定义打印进度函数
def print_progress(progress, epoch_num, loss):

    barLength = 30

    assert type(progress) is float, "id is not a float: %r" % id
    assert 0 <= progress <= 1, "variable should be between zero and one!"

    status = ""

    if progress >= 1:
        progress = 1
        status = "\r\n"

    indicator = int(round(barLength*progress))

    list = [str(epoch_num), "#"*indicator , ">"*(barLength-indicator), progress*100, loss]
    text = "\rEpoch {0[0]} {0[1]} {0[2]} {0[3]:.2f}% completed, total reconstruction loss: {0[4]:.4f}.{1}".format(list, status)
    sys.stdout.write(text)
    sys.stdout.flush()

# 提取 MNIST 数据集中的 0, 1 图像的函数
def extraction_fn(data):
    index_list = []
    for idx in range(data.shape[0]):
        if data[idx] == 0 or data[idx] == 1:
            index_list.append(idx)
    return index_list

# 将数字标签转换为颜色符号的函数
def index_to_color(idx):
    color_list = []
    for i in idx:
        if i == 0:
            color_list += 'b'
        else: color_list += 'g'
    return color_list

# Xavier Glorot 参数初始化 
def glorot_init(shape, name):
    initial = tf.truncated_normal(shape=shape, stddev=1. / tf.sqrt(shape[0] / 2.))
    return tf.Variable(initial, name=name)

# bias 参数初始化
def bias_init(shape, name):
    initial =  tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)

if __name__ == '__main__':

    # 定义一些传参
    tf.app.flags.DEFINE_float('learning_rate', 3e-3, 'initial learning rate, default is 3e-3.') 
    tf.app.flags.DEFINE_integer('num_epochs', 150, 'number of epochs, default is 150.') 
    tf.app.flags.DEFINE_integer('batch_size', 48, 'batch size, default is 48.') 
    FLAGS = tf.app.flags.FLAGS

    display_step = 10

    dir_path = os.path.dirname(os.path.abspath(__file__))
    event_path = os.path.join(dir_path, 'Tensorboard')
    checkpoint_path = os.path.join(dir_path, 'Checkpoints')

    # 图像预处理程序，包含 0, 1 图像及其标签的提取
    mnist = input_data.read_data_sets("MNIST_data", one_hot=False)
    data = {}

    index_list_train = extraction_fn(mnist.train.labels)
    index_list_test = extraction_fn(mnist.test.labels)

    data['train_imgs'], data['train_lbs'] = mnist.train.images[index_list_train], mnist.train.labels[index_list_train]
    data['test_imgs'], data['test_lbs'] = mnist.test.images[index_list_test], mnist.test.labels[index_list_test]

    data['train_imgs_lbs'] = np.c_[data['train_imgs'], data['train_lbs']]
    num_samples, num_features = data['train_imgs'].shape

    # 隐层参数
    num_hidden1 = 128
    num_hidden2 = 3
    num_input = 784

    # 构建网络图
    graph = tf.Graph()
    with graph.as_default():
        
        # 权重参数及偏置
        with tf.name_scope('Weights_and_bisaes'):
            weights = {
                'encoder_w1': glorot_init([num_features, num_hidden1], name='encoder_w1'), 
                'encoder_w2': glorot_init([num_hidden1, num_hidden2], name='encoder_w2'), 
                'decoder_w1': glorot_init([num_hidden2, num_hidden1], name='decoder_w1'), 
                'decoder_w2': glorot_init([num_hidden1, num_features], name='decoder_w2')
            }

            biases = {
                'encoder_b1': bias_init([num_hidden1], name='encoder_b1'), 
                'encoder_b2': bias_init([num_hidden2], name='encoder_b2'), 
                'decoder_b1': bias_init([num_hidden1], name='decoder_b1'), 
                'decoder_b2': bias_init([num_features], name='decoder_b2')
            }
        
        # 编码器和解码器函数
        with tf.name_scope('Encoder_and_decoder'):
            # 编码器函数
            def encoder(x):
                layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_w1']), 
                                                biases['encoder_b1']))
                layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_w2']), 
                                                biases['encoder_b2']))
                return layer_2

            # 解码器函数
            def decoder(x):
                layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_w1']), 
                                                biases['decoder_b1']))
                layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_w2']), 
                                                biases['decoder_b2']))
                return layer_2

        # 主网络结构
        with tf.name_scope('Main_structure'):
            with tf.name_scope('Input'):
                X = tf.placeholder("float", [None, num_features], name='input_images')
                encoder_op = encoder(X)

            with tf.name_scope('Output'):
                y_pred = decoder(encoder_op)

            with tf.name_scope('Loss'):
                # 重构误差（平方差）
                loss = tf.reduce_mean(tf.pow(X - y_pred, 2))
                # 重构误差（交叉熵）
                # loss = -tf.reduce_mean(X * tf.log(1e-10 + y_pred) + (1 - X) * tf.log(1e-10 + 1 - y_pred)) 
                # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=X, logits=y_pred))

            with tf.name_scope('Train'):
                train_op = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(loss)

        # summaries 的定义
        tf.summary.image('input_imgs', tf.reshape(X, [-1, 28, 28, 1]), max_outputs=3, collections=['train'])
        tf.summary.image('reconstructed_imgs', tf.reshape(y_pred, [-1, 28, 28, 1]), max_outputs=3, collections=['train'])
        tf.summary.scalar('loss', loss, collections=['train'])
        summ_train = tf.summary.merge_all('train') 

    # 模型的训练
    with tf.Session(graph=graph) as sess:
        sess.run(tf.global_variables_initializer())

        summ_writer = tf.summary.FileWriter(event_path)
        summ_writer.add_graph(sess.graph)

        max_batch = num_samples // FLAGS.batch_size
        for epoch_num in range(FLAGS.num_epochs):
            np.random.shuffle(data['train_imgs'])
            for batch_num in range(max_batch):
                index_start = batch_num * FLAGS.batch_size
                index_end = (batch_num + 1) * FLAGS.batch_size
                imgs_batch = data['train_imgs'][index_start:index_end, :]
                _, batch_loss = sess.run([train_op, loss], feed_dict={X: imgs_batch})
                
            total_loss, rs = sess.run([loss, summ_train], feed_dict={X: data['train_imgs']})
            summ_writer.add_summary(rs, global_step=epoch_num)

            progress = float(epoch_num % display_step + 1) / display_step
            print_progress(progress, epoch_num + 1, total_loss)

        print('Training completed.')

        # 编码需要显示的 400 幅图像
        encoder_imgs = sess.run(encoder_op, feed_dict={X: data['train_imgs'][:400]})

    # 通过 k-means 函数对特征表示进行聚类
    mu, label = kmeans2(encoder_imgs, k=2, iter=10)

    # 结果显示
    titles = ['Distribution of encoded images', 'Clustered data by kmeans']
    index_list = [np.zeros((400,), dtype=int), label] 
    fig = plt.figure(1, figsize=(16, 8))
    fig.subplots_adjust(wspace=0.01, hspace=0.02)
    for i, title, idx in zip([1, 2], titles, index_list):
        ax = fig.add_subplot(1, 2, i, projection='3d')
        color = index_to_color(idx)
        ax.scatter(encoder_imgs[:, 0], encoder_imgs[:, 1], encoder_imgs[:, 2], c=color, s=35, alpha=0.4, marker='o')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.view_init(elev=30., azim=-45)
        ax.set_title(title, fontsize=14)
    ax.scatter(mu[:, 0], mu[:, 1], mu[:, 2], c='r', s=250, alpha=0.8, marker='*')
    plt.show()

    

import os
import tensorflow as tf
import numpy as np
from sklearn.cluster import KMeans


class AutoEncoder():
    def __init__(self, encoder_dims, ae_weights_init):
        # 編碼器大小
        self.encoder_dims = encoder_dims
        # 權重初始化
        self.ae_weights_init = ae_weights_init

    def __call__(self, X, Y):
        with tf.name_scope('encoder'):
            self.H = self.encoder(X)
        with tf.name_scope('decoder'):
            self.X_ = self.decoder(self.H)

        self.loss = tf.reduce_mean((X - self.X_) ** 2)
        self.train_op = tf.train.MomentumOptimizer(learning_rate=1, momentum=0.9).minimize(self.loss)

        return self.train_op, self.loss, self.H

    def encoder(self, x):
        h = x
        n_stacks = len(self.encoder_dims) - 1
        # 中间隐隐层Encoder
        for i in range(n_stacks - 1):
            h = tf.layers.dense(h, self.encoder_dims[i+1], kernel_initializer=self.ae_weights_init, activation=tf.nn.relu, name='encoder_%d' % i)
        # 输出层Encoder（，10）
        h =tf.layers.dense(h,self.encoder_dims[-1], kernel_initializer=self.ae_weights_init, name='encoder_%d' % (n_stacks - 1)) # hidden layer, features are extracted from here
        return h
    def decoder(self, z):
        n_stacks = len(self.encoder_dims) - 1
        y = z
        # 中间隐藏层Decoder
        for i in range(n_stacks - 1, 0, -1):
            y = tf.layers.dense(y, self.encoder_dims[i], kernel_initializer=self.ae_weights_init, activation=tf.nn.relu, name='decoder_%d' % i)
        # Decoder输出层（，784）
        y = tf.layers.dense(y, self.encoder_dims[0],  kernel_initializer=self.ae_weights_init, name='decoder_0')
        return y

class Model():
    def __init__(self, encoder_dims, ae_weights_init, n_clusters):
        # 編碼器大小
        self.encoder_dims = encoder_dims
        self.ae_weights_init = ae_weights_init
        # 聚类个数
        self.n_cluster = n_clusters
        self.alpha = 1.0

        self.kmeans = KMeans(n_clusters=n_clusters, n_init=20)
        self.mu = tf.get_variable(shape=(self.n_cluster, self.encoder_dims[-1]), initializer=tf.glorot_uniform_initializer(), name="mu")

    def __call__(self, X, Y, input_batch_size):
        with tf.name_scope('encoder'):
            self.H = self.encoder(X)
        with tf.name_scope("distribution"):
            self.q = self._soft_assignment(self.H, self.mu, input_batch_size)
            self.p = tf.placeholder(tf.float32, shape=(None, self.n_cluster))

        with tf.name_scope("train"):
            self.loss = self._kl_divergence(self.p, self.q)

            self.train_op = tf.train.MomentumOptimizer(learning_rate=0.01, momentum=0.9).minimize(self.loss)
            # self.train_op =  self.q

        return self.train_op, self.loss, self.H


    def encoder(self, x):
        h = x
        n_stacks = len(self.encoder_dims) - 1
        # 中间隐隐层Encoder
        for i in range(n_stacks - 1):
            h = tf.layers.dense(h, self.encoder_dims[i+1], kernel_initializer=self.ae_weights_init, activation=tf.nn.relu, name='encoder_%d' % i)
        # 输出层Encoder（，10）
        h =tf.layers.dense(h,self.encoder_dims[-1], kernel_initializer=self.ae_weights_init, name='encoder_%d' % (n_stacks - 1)) # hidden layer, features are extracted from here
        return h

    def get_assign_cluster_centers_op(self, features):
        # init mu
        print("Kmeans train start.")
        kmeans = self.kmeans.fit(features)
        print("Kmeans train end.")
        return tf.assign(self.mu, kmeans.cluster_centers_)

    def _soft_assignment(self, embeddings, cluster_centers, input_batch_size):
        """Implemented a soft assignment as the  probability of assigning sample i to cluster j.

        Args:
            embeddings: (num_points, dim)
            cluster_centers: (num_cluster, dim)

        Return:
            q_i_j: (num_points, num_cluster)
        """
        # q = 1.0 / (1.0 + (K.sum(K.square(K.expand_dims(inputs, axis=1) - self.clusters), axis=2) / self.alpha))
        # q **= (self.alpha + 1.0) / 2.0
        # q = K.transpose(K.transpose(q) / K.sum(q, axis=1))
        def _pairwise_euclidean_distance(a, b):
            p1 = tf.matmul(
                tf.expand_dims(tf.reduce_sum(tf.square(a), 1), 1),
                tf.ones(shape=(1, self.n_cluster))
            )
            p2 = tf.transpose(tf.matmul(
                tf.reshape(tf.reduce_sum(tf.square(b), 1), shape=[-1, 1]),
                tf.ones(shape=(input_batch_size, 1)),
                transpose_b=True
            ))
            res = tf.sqrt(tf.add(p1, p2) - 2 * tf.matmul(a, b, transpose_b=True))
            return res

        dist = _pairwise_euclidean_distance(embeddings, cluster_centers)
        q = 1.0 / (1.0 + dist ** 2 / self.alpha) ** ((self.alpha + 1.0) / 2.0)
        q = (q / tf.reduce_sum(q, axis=1, keepdims=True))
        return q

    def target_distribution(self, q):
        p = q ** 2 / q.sum(axis=0)
        p = p / p.sum(axis=1, keepdims=True)
        return p

    def _kl_divergence(self, target, pred):
        return tf.reduce_mean(tf.reduce_sum(target * tf.log(target / (pred)), axis=1))



import keras.backend as K
import numpy as np
from keras.layers import Layer, Dense


class VLAD(Layer):

    def __init__(self, n_clusters, **kwargs):
        super(VLAD, self).__init__(**kwargs)
        self.n_clusters = n_clusters

    def build(self, input_shape):
        if K.image_data_format()=='channels_last':
            self.input_dim = input_shape[-1]
            print('input_dim', self.input_dim)
        else:
            print(K.image_data_format())
            print(input_shape[1])
            self.input_dim = input_shape[1]


        self.weights_clustering = self.add_weight(name="assign",
                                                  constraint=None,
                                                  regularizer=None,
                                                  trainable=True,
                                                  initializer="glorot_uniform",
                                                  shape=(1, 1, self.input_dim, self.n_clusters))
        print('weights_clustering', self.weights_clustering)
        if K.image_data_format() == 'channels_last':
            self.bias_clustering_shape = (1, 1, 1, self.n_clusters)
            print('bias_clustering_shape', self.bias_clustering_shape)
        else:
            self.bias_clustering_shape = (1, self.n_clusters, 1, 1)

        self.bias_clustering = self.add_weight(name='bias_assignment',
                                               constraint=None,
                                               regularizer=None,
                                               trainable=True,
                                               initializer="glorot_uniform",
                                               shape=self.bias_clustering_shape)
        print('bias_clustering', self.bias_clustering)

        self.centering = self.add_weight(name='centering',
                                         shape=(1, 1, 1, self.n_clusters, self.input_dim),
                                         initializer="glorot_uniform",
                                         regularizer=None,
                                         trainable=True,
                                         constraint=None)
        print('centering', self.centering)

        a = super(VLAD, self).build(input_shape)
        print('super(VLAD, self).build(input_shape)', a)

    def call(self, inputs, **kwargs):
        # Preparing tensors: Cluster assignment
        assign = K.conv2d(x=inputs,
                          kernel=self.weights_clustering,
                          strides=(1, 1),
                          padding="same",
                          data_format="channels_last",
                          dilation_rate=(1, 1))  # shape = bs x W x H x Nc
        assign += self.bias_clustering  # shape = bs x W x H x Nc
        if K.backend() == "tensorflow":
            assign = K.softmax(assign)  # shape = bs x W x H x Nc
        else:
            s = np.shape(assign)
            assign = K.reshape(assign, (s[0]*s[1]*s[2], s[3]))
            assign = K.softmax(assign)
            assign = K.reshape(assign, (s[0], s[1], s[2], s[3]))


        if K.image_data_format()=='channels_first':
            assign = K.permute_dimensions(assign, (0, 2, 3, 1))
            inputs = K.permute_dimensions(inputs, (0, 2, 3, 1))
        assign = K.expand_dims(assign, 4)  # shape = bs x W x H x Nc x 1

        # Preparing tensors: descriptors centering
        x = K.expand_dims(inputs, 3)  # shape = bs x W x H x 1 x D
        x = K.tile(x, (1, 1, 1, self.n_clusters, 1))  # shape = bs x W x H x Nc x D
        x -= self.centering  # shape = bs x W x H x Nc x D

        # broadcasting assignment over descriptors and dimensions:
        x = assign * x  # shape = bs x W x H x Nc x D
        # intra-normalisation:
        x = K.l2_normalize(x, axis=4)
        print(' x = K.l2_normalize', x)
        x = K.sum(x, axis=[1, 2])  # shape = bs x Nc x D
        print(' K.sum(x, axis=[1, 2])', x)
        # Computing representation + l2-normalisation
        x = K.reshape(x, (K.shape(inputs)[0], self.n_clusters * self.input_dim))
        print('K.reshape(x, (K.shape(inputs)[0], self.n_clusters * self.input_dim))', x)
        x = K.l2_normalize(x, axis=1)
        print('x  K.l2_normalize', x)
        return x

    def compute_output_shape(self, input_shape):
        """get the shape of the VLAD layer output

        Args:
            input_shape: A Tensor

        Returns:

        """
        if K.image_data_format() == 'channels_first':
            return input_shape[0], self.n_clusters * input_shape[1]
        else:
            return input_shape[0], self.n_clusters * input_shape[-1]

    def get_config(self):
        base_config = super(VLAD, self).get_config()
        config = {"n_clusters": self.n_clusters}

        return dict(list(base_config.items()) + list(config.items()))
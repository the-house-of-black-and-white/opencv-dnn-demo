import cv2
import numpy as np
from nets import nets_factory as slim_nets
import tensorflow as tf
slim = tf.contrib.slim


class TensorflowClassifier:

    def __init__(self, checkpoint):
        print("[INFO] loading model...")
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(self.sess, checkpoint)
        # Load net architecture
        self.X = tf.placeholder(dtype=tf.float32, shape=(None, 299, 299, 3), name='X')
        self.Y_age_pred, self.Y_eth_pred, self.Y_gender_pred, _, _ = cnn_architecture(self.X, is_training=False)

    def classify_all(self, image, boxes):
        # Create placeholders

        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float)
        img /= 255.0
        img = np.expand_dims(cv2.resize(img, (299, 299)), axis=0)

        Y_age_pred_v, Y_gender_pred_v, Y_eth_v = self.sess.run(
            [self.Y_age_pred, self.Y_gender_pred, self.Y_eth_pred], feed_dict={self.X: img})

        Y_age_pred_v = np.argmax(Y_age_pred_v, axis=1)[0]
        Y_gender_pred_v = np.argmax(Y_gender_pred_v, axis=1)[0]
        Y_eth_v = np.argmax(Y_eth_v, axis=1)[0]

        tf.logging.info("Age: {} Gender: {} Ethniticy: {}".format(Y_age_pred_v, Y_gender_pred_v, Y_eth_v))



def cnn_architecture(inputs, is_training, weight_decay=0.00004):
    """
    Return network architecture

    Args:
        inputs: Tensor
            Input Tensor
        is_training: bool
            Whether the network will be used to train or not. Used for dropout operation

    Returns:
        Logits for each demographic network branch
    """

    # tf.logging.info("is_training: {} network_name: {} endpoint: {}".format(is_training,network_name,endpoint))

    net_fn = slim_nets.get_network_fn(
        name="inception_v4", num_classes=None, is_training=is_training, weight_decay=weight_decay)

    _, endpoints = net_fn(inputs)

    # net_final = endpoints['Mixed_7d']
    # net_final = tf.layers.flatten(net_final)
    net_final = endpoints['global_pool']

    aux_logits = endpoints['Mixed_6h']

    # arg_scope = slim_nets.arg_scopes_map['inception_v4'](weight_decay=weight_decay)
    # with tf.contrib.framework.arg_scope(demographics_arg_scope(weight_decay=weight_decay)):
    # arg_scope = tf.contrib.framework.arg_scope
    # with arg_scope([tf.layers.conv2d,tf.layers.dense],kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay)):
    with tf.variable_scope("Demographics"):
        with tf.variable_scope("Aux_Logits"):
            aux_logits = tf.layers.average_pooling2d(
                aux_logits, pool_size=[5, 5], strides=3, padding='valid', name='AvgPool_1a_5x5')
            aux_logits = tf.layers.conv2d(
                aux_logits, 128, kernel_size=[1, 1],
                kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay), name='Conv2d_1b_1x1')
            aux_logits = tf.layers.conv2d(
                aux_logits, 768, aux_logits.get_shape()[1:3],
                kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay), name='Conv2d_2a')
            aux_logits = tf.layers.flatten(aux_logits)

            with tf.variable_scope("Aux_Age"):
                aux_net_age = tf.layers.dense(
                    aux_logits, 6, activation=None, name='Aux_FC1', trainable=True,
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay))

            with tf.variable_scope("Aux_Ethnicity"):
                aux_net_eth = tf.layers.dense(
                    aux_logits, 5, activation=None, name='Aux_FC1', trainable=True,
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay))

            with tf.variable_scope("Aux_Gender"):
                aux_net_gender = tf.layers.dense(
                    aux_logits, 2, activation=None, name='Aux_FC1', trainable=True,
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay))

        with tf.variable_scope("Logits"):
            net_final = tf.layers.batch_normalization(net_final, epsilon=1e-3, momentum=0.9997,
                                                      name='LogitBatchnorm', training=is_training)

            net_final = tf.layers.dropout(
                net_final, rate=0.5, name='Dropout_1b')
            net_final = tf.layers.flatten(net_final, name='PreLogitsFlatten')

            with tf.variable_scope("Age"):
                net_age = tf.layers.dense(
                    net_final, 6, activation=None, name='FC1', trainable=True,
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay))

            with tf.variable_scope("Ethnicity"):
                net_eth = tf.layers.dense(
                    net_final, 5, activation=None, name='FC1', trainable=True,
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay))

            with tf.variable_scope("Gender"):
                net_gender = tf.layers.dense(
                    net_final, 2, activation=None, name='FC1', trainable=True,
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay))

    return net_age, net_eth, net_gender, aux_net_age, aux_net_eth, aux_net_gender


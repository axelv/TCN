import tensorflow as tf
import numpy as np


def model_fn(features, labels, mode, params):
    TRAINING = (mode == tf.estimator.ModeKeys.TRAIN)

    def ResidualBlock(x, kernel_size, n_inputs, n_outputs, dilation_rate, activation = tf.nn.leaky_relu):


        y = x
        input_channels = n_inputs
        output_channels = n_outputs

        for i in range(3):
            y = tf.pad(y, [[0,0],[(kernel_size-1)*dilation_rate, 0], [0, 0]])
            y = tf.layers.conv1d(y,
                                 kernel_size=kernel_size,
                                 filters=output_channels, padding="same",
                                 dilation_rate=dilation_rate,
                                 kernel_constraint= lambda x: tf.nn.l2_normalize(x, axis=0))
            y = tf.slice(y, begin=[0, 0, 0], size=[-1, x.get_shape()[1], output_channels])
            y = activation(y)
            y = tf.layers.dropout(y, rate=0.3, training=TRAINING)

        x_downsampled = tf.layers.conv1d(y, kernel_size=1, filters=n_outputs)
        y = activation(y + x_downsampled)

        return y

    with tf.variable_scope("TCN"):

        with tf.variable_scope('InputLayer'):
            x = tf.feature_column.input_layer(features, params['feature_columns'])
            num_channels = params['num_channels']
            num_layers = len(num_channels)-1

        y = tf.reshape(x, shape=(-1, 200, 1))
        for i in range(num_layers):
            dilation_rate = 2 ** i
            in_channels = num_channels[i]
            out_channels = num_channels[i+1]
            with tf.variable_scope("ResidualBlock_"+str(i)):

                y = ResidualBlock(y,
                                  kernel_size=2,
                                  n_inputs=in_channels,
                                  n_outputs=out_channels,
                                  dilation_rate=dilation_rate)

        with tf.variable_scope('OutputLayer'):
            y = tf.layers.flatten(y)
            logits = tf.layers.dense(y, num_channels[-1])

        # Compute predictions.
        predicted_classes = tf.argmax(logits, 1)
        if mode == tf.estimator.ModeKeys.PREDICT:
            predictions = {
                'predictions': logits,
                'logits': logits,
            }
            return tf.estimator.EstimatorSpec(mode, predictions=predictions)

        # Compute loss.
        loss = tf.losses.mean_squared_error(labels=labels, predictions=logits)

        # Compute evaluation metrics.
        accuracy = tf.metrics.accuracy(labels=labels,
                                       predictions=logits,
                                       name='acc_op')

        metrics = {'accuracy': accuracy}
        tf.summary.scalar('accuracy', accuracy[1])

        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(
                mode, loss=loss, eval_metric_ops=metrics)

        optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)












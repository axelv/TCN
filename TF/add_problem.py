import tensorflow as tf
import numpy as np


def model_fn(features, labels, mode, params):
    TRAINING = (mode == tf.estimator.ModeKeys.TRAIN)
    STDEV = 0.1

    print(labels)

    def ResidualBlock(x, kernel_size, n_inputs, n_outputs, dilation_rate, activation = tf.nn.relu, dropout=0.0):

        y = x
        input_channels = n_inputs
        output_channels = n_outputs

        for i in range(2):
            with tf.variable_scope('Layer_'+str(i)):
                y = tf.pad(y, [[0,0],[(kernel_size-1)*dilation_rate, 0], [0, 0]])
                y = tf.layers.conv1d(y,
                                     kernel_size=[kernel_size],
                                     filters=output_channels, padding="valid",
                                     dilation_rate=dilation_rate,
                                     #kernel_constraint= lambda x: tf.nn.l2_normalize(x, axis=0),
                                     kernel_initializer=tf.random_normal_initializer(stddev=STDEV),
                                     use_bias=False,
                                     data_format="channels_last")
                if params['batch_norm']:
                    y = tf.layers.batch_normalization(y, training=TRAINING)
                y = activation(y)
                y = tf.layers.dropout(y, rate=dropout, training=TRAINING)

        x_downsampled = tf.layers.conv1d(x,
                                         padding="valid",
                                         kernel_size=1,
                                         filters=n_outputs,
                                         kernel_initializer=tf.random_normal_initializer(stddev=STDEV),
                                         use_bias=False,
                                         data_format="channels_last")
        y = activation(y + x_downsampled)

        return y

    with tf.variable_scope("TCN"):

        with tf.variable_scope('InputLayer'):
            x =features['x']
            num_channels = params['num_channels']
            kernel_size = params['kernel_size']
            seq_length = params['seq_length']
            num_layers = len(num_channels)-2

        y = tf.reshape(x, shape=(-1, seq_length, num_channels[0]))
        for i in range(num_layers):
            dilation_rate = 2 ** i
            in_channels = num_channels[i]
            out_channels = num_channels[i+1]
            with tf.variable_scope("ResidualBlock_"+str(i)):

                y = ResidualBlock(y,
                                  kernel_size=kernel_size,
                                  n_inputs=in_channels,
                                  n_outputs=out_channels,
                                  dilation_rate=dilation_rate,
                                  dropout=params['dropout'])

        with tf.variable_scope('OutputLayer'):
            y = tf.layers.flatten(y)
            outputs = tf.layers.dense(y,
                                      units=num_channels[-1],
                                      kernel_initializer=tf.random_normal_initializer(stddev=STDEV),
                                      use_bias=False)

        # Compute predictions.
        predicted_classes = tf.argmax(outputs, 1)
        if mode == tf.estimator.ModeKeys.PREDICT:
            predictions = {
                'predictions': outputs,
            }
            return tf.estimator.EstimatorSpec(mode, predictions=predictions)

        # Compute loss.
        loss = tf.losses.mean_squared_error(labels=labels, predictions=outputs)

        # Compute evaluation metrics.
        accuracy = tf.metrics.accuracy(labels=labels,
                                       predictions=outputs,
                                       name='acc_op')

        metrics = {'accuracy': accuracy}
        tf.summary.scalar('accuracy', accuracy[1])

        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(
                mode, loss=loss, eval_metric_ops=metrics)

        optimizer = tf.train.AdagradOptimizer(learning_rate=params['learning_rate'])

        grads = tf.gradients(loss, tf.trainable_variables())
        grads = list(zip(grads, tf.trainable_variables()))

        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

        # generate summaries
        for grad, var in grads:
            tf.summary.histogram(var.name+'/gradient', grad)
            tf.summary.histogram(var.name, var)

        print("* * *  MODEL CONSTRUCTED * * *")
        return tf.estimator.EstimatorSpec(mode,
                                          loss=loss,
                                          train_op=train_op)












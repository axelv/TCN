import os
import numpy as np
import tensorflow as tf
from TF.utils import get_run_dir, data_generator

STDEV = 0.01
BATCH_NORM = True
DROPOUT = 0


SEQ_LEN = 200
KERNEL_SIZE = 6
TRAIN_SAMPLES = 50000
TRAIN_BATCH = 100
EVAL_SAMPLES = 1000
LEARNING_RATE = 0.002
CHANNELS = 10
DROPOUT = 0.0
MODEL_PATH = get_run_dir(os.sep + os.path.join('tmp', 'adding_problem'))
NUM_STEPS = 3000

num_channels = [2, CHANNELS, CHANNELS, CHANNELS, CHANNELS, CHANNELS, CHANNELS, 1]

def weightnorm_conv1d(x, kernel_size, num_filters, dilation_rate):

    with tf.variable_scope("causal_conv1d"):
        # data based initialization of parameters
        x = tf.pad(x, [[0, 0], [(kernel_size - 1) * dilation_rate, 0], [0, 0]])
        V = tf.get_variable('V', [1, kernel_size, int(x.get_shape()[-1]), num_filters], tf.float32,
                            tf.random_normal_initializer(), trainable=True)

        V_norm = tf.nn.l2_normalize(V, [0, 1])

        x_expand = tf.expand_dims(x, axis=1)
        x_expand = tf.nn.atrous_conv2d(x_expand, V_norm, rate=dilation_rate, padding="VALID")
        g = tf.get_variable('g', shape=[1, 1, num_filters], dtype=tf.float32, initializer=tf.random_normal_initializer(stddev=STDEV), trainable=True)
        x_out = tf.squeeze(x_expand, axis=[1])
        return g*x_out

def ResidualBlock(x, training, kernel_size, n_inputs, n_outputs, dilation_rate, activation = tf.nn.relu, dropout=0.0):

    y = x
    input_channels = n_inputs
    output_channels = n_outputs

    for i in range(2):
        with tf.variable_scope('Layer_'+str(i)):
            #Alternative using standar layers

            y = tf.pad(y, [[0,0],[(kernel_size-1)*dilation_rate, 0], [0, 0]])
            y = tf.layers.conv1d(y,
                                 kernel_size=[kernel_size],
                                 filters=output_channels, padding="valid",
                                 dilation_rate=dilation_rate,
                                 #kernel_constraint= lambda x: tf.nn.l2_normalize(x, [0, 1]),
                                 kernel_initializer=tf.random_normal_initializer(stddev=STDEV),
                                 use_bias=False,
                                 data_format="channels_last")

            # g = tf.get_variable('g',
            #                     shape=[1, 1, output_channels], dtype=tf.float32,
            #                     initializer=tf.random_normal_initializer(stddev=STDEV),
            #                     trainable=True)
            #
            # y = g*y

            #y = weightnorm_conv1d(y, kernel_size=kernel_size, num_filters=output_channels, dilation_rate=dilation_rate)

            if BATCH_NORM:
                y = tf.layers.batch_normalization(y, training=training, scale=False)
            y = activation(y)

            if DROPOUT > 0:
                y = tf.layers.dropout(y, rate=dropout, training=training)

    x_downsampled = tf.layers.conv1d(x,
                                     padding="valid",
                                     kernel_size=1,
                                     filters=n_outputs,
                                     kernel_initializer=tf.random_normal_initializer(),
                                     use_bias=False,
                                     data_format="channels_last")
    y = activation(y + x_downsampled)

    return y

def model(x, training, num_channels, kernel_size, seq_length):

    with tf.variable_scope("TCN"):

        with tf.variable_scope('InputLayer'):

            num_layers = len(num_channels)-2

        y = tf.reshape(x, shape=(-1, seq_length, num_channels[0]))
        for i in range(num_layers):
            dilation_rate = 2 ** i
            in_channels = num_channels[i]
            out_channels = num_channels[i+1]
            with tf.variable_scope("ResidualBlock_"+str(i)):

                y = ResidualBlock(y,
                                  training = training,
                                  kernel_size=kernel_size,
                                  n_inputs=in_channels,
                                  n_outputs=out_channels,
                                  dilation_rate=dilation_rate,
                                  dropout=DROPOUT)

        with tf.variable_scope('OutputLayer'):
            y = tf.squeeze(tf.slice(y, begin=[0, SEQ_LEN-1, 0], size=[-1, 1, -1]), axis=[1])
            y = tf.layers.dense(y,
                                units=num_channels[-1],
                                kernel_initializer=tf.random_normal_initializer(stddev=STDEV),
                                use_bias=False)

        return y

if __name__ == "__main__":

    with tf.Graph().as_default():

        x = tf.placeholder(tf.float32, [None, SEQ_LEN, num_channels[0]])
        training = tf.placeholder(tf.bool, [])
        y = tf.placeholder(tf.float32, [None, 1])

        y_pred = model(x,
                       training,
                       num_channels=num_channels,
                       kernel_size=KERNEL_SIZE,
                       seq_length=SEQ_LEN)

        loss_op = tf.reduce_mean(tf.square(y-y_pred))
        tf.summary.scalar("MSE", loss_op)
        tf.summary.scalar("Variance Pred", tf.reduce_mean(tf.square(y_pred)))
        tf.summary.scalar("Variance Target", tf.reduce_mean(tf.square(y)))

        # Generate summaries of variables and gradients
        grads = tf.gradients(loss_op, tf.trainable_variables())
        grads = list(zip(grads, tf.trainable_variables()))

        for grad, var in grads:
            tf.summary.histogram(var.name + '/gradient', grad)
            tf.summary.histogram(var.name, var)

        optimizer = tf.train.AdagradOptimizer(learning_rate=LEARNING_RATE)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(loss_op)

        init_global_op = tf.global_variables_initializer()
        init_local_op = tf.local_variables_initializer()

        summary_op = tf.summary.merge_all()


        #Train & Eval session
        with tf.Session() as sess:

            sess.run([init_global_op, init_local_op])

            train_summary_writer = tf.summary.FileWriter(MODEL_PATH,
                                                         graph=tf.get_default_graph())
            eval_summary_writer = tf.summary.FileWriter(os.path.join(MODEL_PATH, "eval"),
                                                        graph=tf.get_default_graph())

            for batch_i in range(NUM_STEPS):

                x_train, y_train = data_generator(TRAIN_BATCH, SEQ_LEN)

                if batch_i % 10 == 0:
                    _, summary, loss = sess.run([train_op, summary_op, loss_op],
                                                          feed_dict={x: x_train, y: y_train, training: True})

                    train_summary_writer.add_summary(summary, batch_i)
                    print("Step: "+str(batch_i))
                    print("Train Loss: "+str(loss))
                    print("-------------------------------")
                else:
                    sess.run([train_op], feed_dict={x: x_train,
                                                    y: y_train,
                                                    training: True})


                if batch_i % 100 == 0:
                    x_eval, y_eval = data_generator(TRAIN_BATCH, SEQ_LEN)
                    summary, loss = sess.run([summary_op, loss_op], feed_dict={x: x_eval, y: y_eval, training: False})
                    eval_summary_writer.add_summary(summary, batch_i)
                    print("Eval Loss: " + str(loss))
                    print("-------------------------------")


            print("Training Finished")
            x_eval, y_eval = data_generator(TRAIN_BATCH, SEQ_LEN)
            summary, loss = sess.run([summary_op, loss_op], feed_dict={x: x_eval, y: y_eval, training: False})
            eval_summary_writer.add_summary(summary, NUM_STEPS)
            print("Final MSE: " + str(loss))
            print("-------------------------------")




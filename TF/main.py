import os
import numpy as np
import tensorflow as tf
from TF.utils import get_run_dir, data_generator

STDEV = 0.1
BATCH_NORM = True
DROPOUT = False


SEQ_LEN = 200
KERNEL_SIZE = 6
TRAIN_SAMPLES = 50000
TRAIN_BATCH = 50
EVAL_SAMPLES = 1000
LEARNING_RATE = 2e-2
DROPOUT = 0.0
MODEL_PATH = get_run_dir(os.sep + os.path.join('tmp', 'adding_problem'))
NUM_STEPS = 3000

num_channels = [2, 27, 27, 27, 27, 27, 27, 27, 1]

def ResidualBlock(x, training, kernel_size, n_inputs, n_outputs, dilation_rate, activation = tf.nn.relu, dropout=0.0):

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
                                 kernel_initializer=tf.random_normal_initializer(stddev=STDEV),
                                 use_bias=False,
                                 data_format="channels_last")
            if BATCH_NORM:
                y = tf.layers.batch_normalization(y, training=training)
            y = activation(y)
            y = tf.layers.dropout(y, rate=dropout, training=training)

    x_downsampled = tf.layers.conv1d(x,
                                     padding="valid",
                                     kernel_size=1,
                                     filters=n_outputs,
                                     kernel_initializer=tf.random_normal_initializer(stddev=STDEV),
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
            y = tf.layers.flatten(y)
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

        loss_op = tf.losses.mean_squared_error(predictions=y_pred, labels=y)
        tf.summary.scalar("MSE", loss_op)

        # Generate summaries of variables and gradients
        grads = tf.gradients(loss_op, tf.trainable_variables())
        grads = list(zip(grads, tf.trainable_variables()))

        for grad, var in grads:
            tf.summary.histogram(var.name + '/gradient', grad)
            tf.summary.histogram(var.name, var)

        optimizer = tf.train.AdagradOptimizer(learning_rate=LEARNING_RATE)
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




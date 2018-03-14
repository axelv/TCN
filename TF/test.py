import numpy as np
import tensorflow as tf
from TF.model import model_fn
from TF.adding_problem import data_generator

params = {'num_channels':
              [2, 2, 2, 2, 2, 1],
          'feature_columns': [tf.feature_column.numeric_column('channel', shape=(100,2))]}
x = dict()

x_, y = data_generator(20000, 100)

x['channel'] = np.transpose(x_, axes=(0,2,1))

print(x['channel'].shape)
train_fn = tf.estimator.inputs.numpy_input_fn(x,
                                              y=y,
                                              batch_size=128,
                                              num_epochs=1,
                                              shuffle=True,
                                              queue_capacity=1000,
                                              num_threads=1
                                              )

classifier = tf.estimator.Estimator(model_fn=model_fn, params=params)

classifier.train(input_fn=train_fn)
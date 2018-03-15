import os
import numpy as np
import tensorflow as tf

from TF.utils import get_run_dir
from TF.model import model_fn
from TF.adding_problem_data import data_generator

SEQ_LEN = 200
TRAIN_SAMPLES = 70000
TRAIN_BATCH = 200
EVAL_SAMPLES = 200
LEARNING_RATE = 4e-3
DROPOUT = 0.0
MODEL_PATH = get_run_dir(os.sep + os.path.join('tmp', 'adding_problem'))

params = {'num_channels':
              [2, 27, 27, 27, 27, 27, 27, 27, 1],
          'feature_columns': [tf.feature_column.numeric_column('channel', shape=(SEQ_LEN,2))],
          'kernel_size': 6,
          'seq_length': SEQ_LEN,
          'learning_rate': LEARNING_RATE,
          'dropout': DROPOUT}

## Setup Dataset

x_train = dict()
x_, y_train = data_generator(TRAIN_SAMPLES, SEQ_LEN)
x_train['channel'] = np.transpose(x_, axes=(0,2,1))

print("Train datashape: ")
print(x_train['channel'].shape)

train_input_fn = tf.estimator.inputs.numpy_input_fn(x=x_train,
                                                    y=y_train,
                                                    batch_size=TRAIN_BATCH,
                                                    num_epochs=5,
                                                    shuffle=False,
                                                    queue_capacity=1000,
                                                    num_threads=1
                                                    )
index = np.random.randint(0, TRAIN_SAMPLES)

x_eval = dict()
x_, y_eval = data_generator(EVAL_SAMPLES, SEQ_LEN)
x_eval['channel'] = np.transpose(x_, axes=(0,2,1))

print("Eval datashape: ")
print(x_eval['channel'].shape)


eval_input_fn = tf.estimator.inputs.numpy_input_fn(x=x_eval,
                                                   y=y_eval,
                                                   batch_size=EVAL_SAMPLES,
                                                   num_epochs=1,
                                                   shuffle=True,
                                                   queue_capacity=1000,
                                                   num_threads=1
                                                   )

# Construct Estimator
predictor = tf.estimator.Estimator(model_fn=model_fn,
                                   params=params,
                                   model_dir=MODEL_PATH)

train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=2000)
eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn, throttle_secs=300)

# Train + Evaluate
tf.logging.set_verbosity(tf.logging.INFO)
tf.estimator.train_and_evaluate(predictor, train_spec, eval_spec)
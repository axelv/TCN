import os
import numpy as np

def get_run_dir(model_path):
    try:
        last_run_dir = os.listdir(model_path)[-1]
        last_run_id = int(last_run_dir)

    except (IndexError, ValueError):
        last_run_id = -1

    except FileNotFoundError:
        os.mkdir(model_path)
        last_run_id = -1

    run_id = last_run_id + 1

    return os.path.join(model_path, str(run_id))

def data_generator(N, seq_length):
    """
    Args:
        seq_length: Length of the adding problem data
        N: # of data in the set
    """
    X_num = np.random.uniform(low=0,high=1,size=[N, 1, seq_length])
    X_mask = np.zeros([N, 1, seq_length])
    Y = np.zeros([N, 1])
    #positions = [20,60]
    for i in range(N):
        positions = np.random.choice(seq_length, size=2, replace=False)
        X_mask[i, 0, positions[0]] = 1
        X_mask[i, 0, positions[1]] = 1
        Y[i,0] = X_num[i, 0, positions[0]] + X_num[i, 0, positions[1]]
    X = np.concatenate((X_num, X_mask), axis=1)
    X = np.transpose(X, axes=(0,2,1))
    return X.astype(np.float32), Y.astype(np.float32)
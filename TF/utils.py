import os

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
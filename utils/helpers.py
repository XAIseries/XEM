import os
import shutil


def create_logger(log_filename, display=True):
    """Create a log file for the experiment"""
    f = open(log_filename, "a")
    counter = [0]

    def logger(text):
        if display:
            print(text)
        f.write(text + "\n")
        counter[0] += 1
        if counter[0] % 10 == 0:
            f.flush()
            os.fsync(f.fileno())

    return logger, f.close


def makedir(path):
    """If the directory does not exist, create it"""
    if not os.path.exists(path):
        os.makedirs(path)


def save_experiment(xp_dir, configuration):
    """Save files about the experiment"""
    makedir(xp_dir)
    shutil.copy(src=configuration, dst=xp_dir)

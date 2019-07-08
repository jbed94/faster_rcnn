import os

import tensorflow as tf
import tensorflow.contrib as tfc


class ExperimentHandler:

    def __init__(self, working_path, out_name, log_interval, model, optimizer) -> None:
        super().__init__()
        train_log_path = os.path.join(working_path, out_name, 'logs', 'train')
        val_log_path = os.path.join(working_path, out_name, 'logs', 'val')
        self.checkpoints_last_n_path = os.path.join(working_path, out_name, 'checkpoints', 'last_n')
        self.checkpoints_best_path = os.path.join(working_path, out_name, 'checkpoints', 'best')

        os.makedirs(train_log_path, exist_ok=True)
        os.makedirs(val_log_path, exist_ok=True)
        os.makedirs(self.checkpoints_last_n_path, exist_ok=True)
        os.makedirs(self.checkpoints_best_path, exist_ok=True)

        self.train_writer = tfc.summary.create_file_writer(train_log_path)
        self.val_writer = tfc.summary.create_file_writer(val_log_path)

        self.ckpt = tf.train.Checkpoint(optimizer=optimizer,
                                        model=model,
                                        optimizer_step=tf.train.get_or_create_global_step())
        self.log_interval = log_interval

    def log_training(self):
        self.train_writer.set_as_default()

    def log_validation(self):
        self.val_writer.set_as_default()

    def flush(self):
        self.train_writer.flush()
        self.val_writer.flush()

    def save_best(self):
        self.ckpt.save(self.checkpoints_best_path)

    def save_last(self):
        self.ckpt.save(self.checkpoints_last_n_path)

    def restore(self, path):
        self.ckpt.restore(path)

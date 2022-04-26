"""

Ordinary Logger

"""

import logging

def get_logger(log_path, mode='a', file_only=False):
    """Logger

    Args:
        log_path : Full path to log file.
        mode : read/write/append mode.
        file_only : Write to file only, and suppress stdout.

    Returns:
        Logger object.

    """
    # logger settings
    logger = logging.getLogger(logger_name)
    logger.setLevel(level=logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # write to file
    handler = logging.FileHandler(log_path, mode=mode)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    # stdout
    if not file_only:
        console = logging.StreamHandler()
        console.setFormatter(formatter)    
        logger.addHandler(console)
    return logger

"""

Tensorboard Logger

"""

import time
from datetime import datetime
from torch.utils.tensorbaord import SummaryWriter

class TensorboardLogger:
    def __init__(self, log_dir, start_iter=1):
        self.iteration = start_iter
        self.writer = self._get_tensorboard_writer(log_dir)

    @staticmethod
    def _get_tensorboard_writer(log_dir):
        timestamp = datetime.fromtimestamp(time.time()).strftime('%Y%m%d-%H:%M')
        tb_logger = SummaryWriter(f'{log_dir}/{timestamp}')
        return tb_logger

    def add_scalar(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.writer.add_scalar(k, v, self.iteration)

    def add_graph(self, model, data=(torch.randn(4, 32), torch.randn(4, 1, 128, 32))):
        self.writer.add_graph(model, input_to_model=data)

    def add_histogram(self, k, v):
        self.writer.add_histogram(k, v, self.iteration)

    def update_iter(self, iteration):
        self.iteration = iteration
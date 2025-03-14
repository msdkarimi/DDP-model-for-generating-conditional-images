import logging
import sys
import os
from datetime import datetime
from logging import StreamHandler
from termcolor import colored  # Assuming you're using termcolor for colored output

class StreamToLogger:
    """Redirects print statements to a logger."""
    def __init__(self, logger, level=logging.INFO):
        self.logger = logger
        self.level = level
        self.linebuf = ''

    def write(self, buf):
        # Handle each line of output
        for line in buf.rstrip().splitlines():
            self.logger.log(self.level, line.rstrip())

    def flush(self):
        # No-op: logger handles flushing internally
        pass

def build_logger(output_dir, name=''):
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # Create formatter
    fmt = '[%(asctime)s %(name)s] (%(filename)s %(lineno)d): %(levelname)s %(message)s'
    color_fmt = colored('[%(asctime)s %(name)s]', 'green') + \
                colored('(%(filename)s %(lineno)d)', 'yellow') + ': %(levelname)s %(message)s'

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(
        logging.Formatter(fmt=color_fmt, datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(console_handler)

    # Create file handlers
    os.makedirs(output_dir, exist_ok=True)

    current_datetime = datetime.now()
    formatted_datetime = current_datetime.strftime("%Y-%m-%d %H %M %S")
    _run_name = f'log_{formatted_datetime}'
    _run_file_name = f'log_{formatted_datetime}.txt'
    os.makedirs(os.path.join(output_dir, _run_name), exist_ok=True)

    file_handler = logging.FileHandler(os.path.join(output_dir, _run_name, _run_file_name), mode='a')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(file_handler)

    # Redirect print statements to logger
    sys.stdout = StreamToLogger(logger, logging.INFO)

    return logger, _run_name



















# import os
# import sys
# import logging
# from termcolor import colored
# from datetime import datetime
#
#
#
# def build_logger(output_dir, name=''):
#     # create logger
#     logger = logging.getLogger(name)
#     logger.setLevel(logging.DEBUG)
#     logger.propagate = False
#
#     # create formatter
#     fmt = '[%(asctime)s %(name)s] (%(filename)s %(lineno)d): %(levelname)s %(message)s'
#     color_fmt = colored('[%(asctime)s %(name)s]', 'green') + \
#                 colored('(%(filename)s %(lineno)d)', 'yellow') + ': %(levelname)s %(message)s'
#
#     # create console handlers for master process
#     console_handler = logging.StreamHandler(sys.stdout)
#     console_handler.setLevel(logging.DEBUG)
#     console_handler.setFormatter(
#         logging.Formatter(fmt=color_fmt, datefmt='%Y-%m-%d %H:%M:%S'))
#     logger.addHandler(console_handler)
#
#     # create file handlers
#     os.makedirs(output_dir, exist_ok=True)
#
#     current_datetime = datetime.now()
#     formatted_datetime = current_datetime.strftime("%Y-%m-%d %H %M %S")
#     _run_name = f'log_{formatted_datetime}'
#     _run_file_name = f'log_{formatted_datetime}.txt'
#     os.makedirs(os.path.join(output_dir, _run_name), exist_ok=True)
#
#     file_handler = logging.FileHandler(os.path.join(output_dir, _run_name, _run_file_name), mode='a')
#     file_handler.setLevel(logging.DEBUG)
#     file_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt='%Y-%m-%d %H:%M:%S'))
#     logger.addHandler(file_handler)
#
#     return logger, _run_name
# Code referenced from https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514
import os

import tensorflow as tf
import numpy as np
import scipy.misc

from tensorboardX import SummaryWriter

try:
    from StringIO import StringIO  # Python 2.7
except ImportError:
    from io import BytesIO         # Python 3.x

class Logger(object):

    def __init__(self, log_dir,step=0):
        """Create a summary writer logging to log_dir."""
        log_dir = os.path.join('log',log_dir)
        print('logging at: ', log_dir)
        self.writer = SummaryWriter(log_dir)
        self.step = step

    def incstep(self): self.step += 1

    def scalar_summary(self, data):
        """Log a dictionary of scalar variables."""

        for tag, value in data.items():
            self.writer.add_scalar(tag,value,self.step)

    def image_summary(self, data):
        """Log a dictionary of images."""

        for tag,img in data.items():
            self.writer.add_image(tag,img,self.step)

    def hist_summary(self, data):
        """Log a histogram of the tensor of values."""

        for tag,values in data.items():
            self.writer.add_histogram(tag,values,self.step,bins="auto")


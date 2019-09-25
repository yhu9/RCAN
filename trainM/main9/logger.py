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

    def __init__(self, log_dir):
        """Create a summary writer logging to log_dir."""
        log_dir = os.path.join('log',log_dir)
        if not os.path.exists(log_dir):
            print('log dir', log_dir)
            self.writer = SummaryWriter(log_dir)
        else:
            print('This training session name already exists. Please use a different name or delete it')
            quit()
        self.step = 0

    def incstep(self): self.step += 1

    def scalar_summary(self, data):
        """Log a scalar variable."""

        for tag, value in data.items():
            self.writer.add_scalar(tag,value,self.step)

    def image_summary(self, tag, images, step):
        """Log a list of images."""

        img_summaries = []
        for i, img in enumerate(images):
            # Write the image to a string
            try:
                s = StringIO()
            except:
                s = BytesIO()
            scipy.misc.toimage(img).save(s, format="png")

            # Create an Image object
            img_sum = tf.Summary.Image(encoded_image_string=s.getvalue(),
                                       height=img.shape[0],
                                       width=img.shape[1])
            # Create a Summary value
            img_summaries.append(tf.Summary.Value(tag='%s/%d' % (tag, i), image=img_sum))

        # Create and write Summary
        summary = tf.Summary(value=img_summaries)
        self.writer.add_summary(summary, step)

    def hist_summary(self, tag, values):
        """Log a histogram of the tensor of values."""

        self.writer.add_histogram(tag,values,self.step,bins="auto")


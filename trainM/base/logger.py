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
            #self.writer = tf.compat.v1.summary.FileWriter(log_dir)
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
            #summary = tf.compat.v1.Summary(value=[tf.compat.v1.Summary.Value(tag=tag, simple_value=value)])
            #self.writer.add_summary(summary, self.step)

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

    def hist_summary(self, tag, values, bins=5):
        """Log a histogram of the tensor of values."""

        '''
        # Create a histogram using numpy
        counts, bin_edges = np.histogram(values, bins=bins)

        # Fill the fields of the histogram proto
        hist = tf.compat.v1.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values**2))

        # Drop the start of the first bin
        bin_edges = bin_edges[1:]

        # Add bin edges and counts
        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for c in counts:
            hist.bucket.append(c)
        '''

        # Create and write Summary
        print('hello')
        self.writer.add_histogram(tag,values.clone().cpu().data.numpy(),self.step)
        print('hello')
        '''
        tf.compat.v1.summary.histogram(name=tag,values=values)
        summary = tf.summary.merge_all()
        print('hello')
        self.writer.add_summary(summary, self.step)
        print('hello')
        self.writer.flush()
        print('hello')
        '''


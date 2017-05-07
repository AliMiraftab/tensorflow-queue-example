from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import pdb
import sys
import argparse

import numpy as np
import tensorflow as tf

from cnn import CNN
from queue_loader import Queue_loader

def test(args):

    queue_loader = Queue_loader(batch_size=args.b_size, num_epochs=1, train=False)

    model = CNN(args.lr, args.b_size, queue_loader.num_batches)
    model.build(queue_loader.images)
    correct_op = tf.reduce_sum(tf.cast(tf.nn.in_top_k(model.logits, queue_loader.labels, 1), tf.int32))

    saver = tf.train.Saver()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))

    latest_ckpt = tf.train.latest_checkpoint('data_log')
    print ('Testing with model: {}'.format(latest_ckpt))
    saver.restore(sess, latest_ckpt)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    correct_all = 0
    try:
        while not coord.should_stop():
            correct = sess.run(correct_op)
            correct_all += correct
    except tf.errors.OutOfRangeError:
        print ('\nDone testing, accuracy: %.2f%%' % (correct_all * 100 / queue_loader.num_examples))
    finally:
        coord.request_stop()

    coord.join(threads)
    sess.close()

def train(args):

    queue_loader = Queue_loader(batch_size=args.b_size, num_epochs=args.ep)

    model = CNN(args.lr, args.b_size, queue_loader.num_batches)
    model.build(queue_loader.images)
    model.loss(queue_loader.labels)
    train_op = model.train()

    saver = tf.train.Saver()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))

    print ('Start training')
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
        ep = 0
        step = 1
        while not coord.should_stop():
            loss, _ = sess.run([model.loss_op, train_op])
            if step % 10 == 0:
                print ('epoch: %2d, step: %2d, loss: %.4f' % (ep+1, step, loss))

            if step % queue_loader.num_batches == 0:
                print ('epoch: %2d, step: %2d, loss: %.4f, epoch %2d done.' % (ep+1, step, loss, ep+1))
                checkpoint_path = os.path.join('data_log', 'stupid.ckpt')
                saver.save(sess, checkpoint_path, global_step=ep+1)
                step = 1
                ep += 1
            else:
                step += 1
    except tf.errors.OutOfRangeError:
        print ('\nDone training, epoch limit: %d reached.' % (args.ep))
    finally:
        coord.request_stop()

    coord.join(threads)
    sess.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', help='train.')
    parser.add_argument('--test', action='store_true', help='test.')
    parser.add_argument('--lr', metavar='', type=float, default=1e-3, help='learning rate.')
    parser.add_argument('--ep', metavar='', type=int, default=3, help='number of epochs.')
    parser.add_argument('--b_size', metavar='', type=int, default=512, help='batch size.')
    args, unparsed = parser.parse_known_args()
    if len(unparsed) != 0: sys.exit('Unknown argument: {}'.format(unparsed))
    if args.train:
        train(args)
    if args.test:
        test(args)
    if not args.train and not args.test:
        parser.print_help()

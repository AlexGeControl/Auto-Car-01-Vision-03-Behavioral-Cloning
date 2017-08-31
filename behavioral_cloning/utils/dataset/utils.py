# Set up session:
import argparse

import os
import sys
import time

from os.path import join

import numpy as np
import cv2
import tensorflow as tf

from .dataset import Dataset

def _int64_feature(value):
    """ Wrapper for Int64List
    """
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    """ Wrapper for BytesList
    """
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float32_feature(value):
    """ Wrapper for FloatList
    """
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _serialize_image(image):
    """ Convert image content and its size to bytes
    """
    # Format image:
    image = np.asarray(
        image,
        np.uint8
    )

    return image.tobytes()

def _create_example(image, label):
    """ Convert (image, label) pair to Tensorflow example
    """
    # Serialize image:
    image_serialized = _serialize_image(image)
    # Create new example:
    example = tf.train.Example(
        features = tf.train.Features(
            feature={
                'image': _bytes_feature(image_serialized),
                'label': _float32_feature(label),
            }
        )
    )

    return example

def show_progress_bar(
    iteration, total,
    prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'
    ):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    # Print New Line on Complete
    if iteration == total:
        print()
    # Else show current progress:
    else:
        percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
        filledLength = int(length * iteration // total)
        bar = fill * filledLength + '-' * (length - filledLength)
        print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')

def write_to_tfrecord(dataset_path):
    """ Write dataset to TFRecords
    """
    # Create TFRecords file:
    filename = join(dataset_path, 'dataset.tfrecords')
    writer = tf.python_io.TFRecordWriter(filename)

    # Convert (image, steering) pair to Tensorflow Example
    print("Convert {} to TFRecords...".format(dataset_path))
    start_time = time.time()

    # Initialize dataset:
    dataset = Dataset(args["dataset"])
    # Initialize progress bar status:
    i = 0
    N = dataset.N
    # Convert:
    for (images, steerings) in iter(dataset):
        for image, steering in zip(images, steerings):
            # Write example:
            example = _create_example(image, steering)
            writer.write(example.SerializeToString())
            # Update progress bar:
            i += 1
            show_progress_bar(
                i, 6*N,
                prefix = 'Progress:', suffix = 'Complete',
                length = 50
            )
    print("[Time Elapsed]: {:.2f} seconds".format(time.time() - start_time))
    print("TFRecords generated.")

    # Finally:
    writer.close()

    return filename

def read_from_tfrecord(
    filenames,
    image_size = [160, 320, 3],
    num_epochs = 10
):
    """ Read dataset from TFRecords
    """
    # Initialize reader:
    reader_queue = tf.train.string_input_producer(
        filenames,
        num_epochs = 10,
        name = "queue"
    )
    reader = tf.TFRecordReader()
    _, serialized = reader.read(reader_queue)

    # Label and image are stored as int64 or float64 values
    # in a serialized tf.Example protobuf.
    features = tf.parse_single_example(
        serialized,
        features={
            'image': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.float32),
        },
        name='features'
    )
    # Image should be decoded as uint8.
    image = tf.decode_raw(features['image'], tf.uint8)
    label = features['label']
    # Reconstruct the image by its shape:
    image = tf.reshape(image, image_size)

    return (image, label)

if __name__ == '__main__':
    # Parse command-line arguments:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--dataset",
        type=str,
        required=True,
        help="Path to behavioral cloning dataset."
    )
    args = vars(parser.parse_args())

    # Use case 01 -- write dataset as TFRecords:
    # filename = write_to_tfrecord(args["dataset"])

    # Use case 02 -- Read from TFRecords
    image, label = read_from_tfrecord(
        [
            join(args["dataset"], 'dataset.tfrecords')
        ],
        num_epochs=1
    )

    batch_size = 128
    capacity = 1024
    min_after_dequeue = 512

    images_batch, labels_batch = tf.train.shuffle_batch(
        [image, label], batch_size = batch_size,
        capacity = capacity, min_after_dequeue = min_after_dequeue
    )

    with tf.Session() as sess:
        # Initialize all variables:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        # Initialize coordinator & queue runner:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        try:
            while not coord.should_stop():
                images_batch_val, labels_batch_val = sess.run(
                    [images_batch, labels_batch]
                )
        except tf.errors.OutOfRangeError:
            print("Done training for {} steps.".format(iter_index))
        finally:
            print("Clear up session.")
            # When done, ask the threads to stop
            coord.request_stop()
            # Wait for threads to finish
            coord.join(threads)

    for index in np.random.choice(len(images_batch_val), 3):
        image, label = images_batch_val[index], labels_batch_val[index]

        cv2.imshow(
            "Steering: {:.2f}".format(label),
            image
        )
        cv2.waitKey(0)

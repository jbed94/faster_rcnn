import tensorflow as tf

from src.utils import coordinates_to_center_point

PADDED_SHAPES = {
    'image/height': [],
    'image/width': [],
    'image/filename': [],
    'image/source_id': [],
    'image/encoded': [None, None, None],
    'image/format': [],
    'image/num_objects': [],
    'image/object/bbox/xmin': [None],
    'image/object/bbox/xmax': [None],
    'image/object/bbox/ymin': [None],
    'image/object/bbox/ymax': [None],
    'image/object/class/text': [None],
    'image/object/class/label': [None],
    'image/object/bbox/roi': [None, 4]
}


def _get_size(filenames):
    size = 0
    for fn in filenames:
        for _ in tf.python_io.tf_record_iterator(fn):
            size += 1
    return size


def _parse_tensors(features):
    img = tf.image.decode_image(features['image/encoded'], 3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    features['image/encoded'] = img

    y_top = features['image/object/bbox/ymin']
    x_left = features['image/object/bbox/xmin']
    y_bottom = features['image/object/bbox/ymax']
    x_right = features['image/object/bbox/xmax']

    coords = tf.stack([y_top, x_left, y_bottom, x_right], 1)
    roi = coordinates_to_center_point(coords)

    features['image/object/bbox/roi'] = roi
    features['image/object/class/label'] = features['image/object/class/label'] - 1
    features['image/num_objects'] = tf.shape(roi)[0]
    return features


def clevr(path):
    filenames = [path]
    features_description = {
        'image/height': tf.FixedLenFeature([], tf.int64),
        'image/width': tf.FixedLenFeature([], tf.int64),
        'image/filename': tf.FixedLenFeature([], tf.string),
        'image/source_id': tf.FixedLenFeature([], tf.string),
        'image/encoded': tf.FixedLenFeature([], tf.string),
        'image/format': tf.FixedLenFeature([], tf.string),
        'image/object/bbox/xmin': tf.FixedLenSequenceFeature([], tf.float32, True),
        'image/object/bbox/xmax': tf.FixedLenSequenceFeature([], tf.float32, True),
        'image/object/bbox/ymin': tf.FixedLenSequenceFeature([], tf.float32, True),
        'image/object/bbox/ymax': tf.FixedLenSequenceFeature([], tf.float32, True),
        'image/object/class/text': tf.FixedLenSequenceFeature([], tf.string, True),
        'image/object/class/label': tf.FixedLenSequenceFeature([], tf.int64, True)
    }

    def _parse_function(example_proto):
        return tf.parse_single_example(example_proto, features_description)

    ds = tf.data.TFRecordDataset(filenames) \
        .map(_parse_function, 8) \
        .map(_parse_tensors, 8)

    return ds

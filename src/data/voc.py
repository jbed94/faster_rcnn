import tensorflow as tf

from ..utils.boxes import coordinates_to_center_point

NUM_CLASSES = 21


def as_object_detection(size=None, as_center_point=True):
    def select(data):
        image = tf.image.convert_image_dtype(data['image'], tf.float32)
        boxes = data['objects']['bbox']

        if size is not None:
            image = tf.image.resize(image, size)
        if as_center_point:
            boxes = coordinates_to_center_point(boxes)

        shape = tf.cast(tf.shape(image)[:2], tf.float32)
        boxes_norm = tf.concat([shape, shape], 0)[tf.newaxis]
        boxes = boxes * boxes_norm

        return {
            'image': image,
            'boxes': boxes,
            'labels': data['objects']['label'],
            'num_objects': tf.shape(data['objects']['bbox'])[0]
        }

    return select

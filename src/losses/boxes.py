import tensorflow as tf


def adjust_bbox(boxes1, boxes2):
    y1, x1, h1, w1 = tf.split(boxes1, 4, -1)
    y2, x2, h2, w2 = tf.split(boxes2, 4, -1)

    return tf.concat([
        (y1 - y2) / h2,
        (x1 - x2) / w2,
        tf.math.log(tf.maximum(h1 / h2, 1e-6)),
        tf.math.log(tf.maximum(w1 / w2, 1e-6))
    ], -1)


def boxes_loss(boxes, rois, anchors):
    t = adjust_bbox(rois, anchors)
    t_gt = adjust_bbox(boxes, anchors)
    return tf.losses.huber(t_gt, t)

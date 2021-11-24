import tensorflow as tf


def center_point_to_coordinates(bbox):
    """
    Converts bboxes in the form of [center_y, center_x, height, width] into
    form of [y_top, x_left, y_bottom, x_right]
    """
    y, x, h, w = tf.split(bbox, 4, -1)
    h2, w2 = tf.cast(h / 2, h.dtype), tf.cast(w / 2, w.dtype)
    return tf.concat([y - h2, x - w2, y + h2, x + w2], -1)


def coordinates_to_center_point(bbox):
    """
    Converts bboxes in the form of [y_top, x_left, y_bottom, x_right] into
    form of [center_y, center_x, height, width]
    """
    y_top, x_left, y_botton, x_right = tf.split(bbox, 4, -1)
    h, w = y_botton - y_top, x_right - x_left

    return tf.concat([y_top + tf.cast(h / 2, h.dtype), x_left + tf.cast(w / 2, w.dtype), h, w], -1)


def get_anchors(shape, anchor_base, downscale_rate, num_scales=3, ratios=[[1, 1], [2, 1], [1, 2]]):
    shape = tf.convert_to_tensor(shape, tf.int32)

    templates = tf.math.pow(2.0, tf.range(num_scales, dtype=tf.float32))
    templates = tf.concat([tf.stack([templates * anchor_base * r1, templates * anchor_base * r2], 1) for r1, r2 in ratios], 0)

    y = tf.linspace(0.5 * downscale_rate, (tf.cast(shape[0], tf.float32) - 0.5) * downscale_rate, shape[0])
    x = tf.linspace(0.5 * downscale_rate, (tf.cast(shape[1], tf.float32) - 0.5) * downscale_rate, shape[1])
    yx = tf.stack(tf.meshgrid(y, x), -1)
    yx = tf.transpose(yx, [1, 0, 2])
    yx = tf.tile(yx[:, :, tf.newaxis, :], [1, 1, tf.shape(templates)[0], 1])
    yx = tf.cast(yx, tf.float32)

    hw = tf.tile(templates[tf.newaxis, tf.newaxis], [shape[0], shape[1], 1, 1])
    yxhw = tf.concat([yx, hw], -1)

    return yxhw


def crop_anchors(anchors, norm):
    coords = center_point_to_coordinates(anchors)
    coords = tf.maximum(coords, 0.0)
    coords = tf.minimum(coords, norm)
    return coordinates_to_center_point(coords)


def filter_anchors(anchors, norm):
    coords = center_point_to_coordinates(anchors)
    return tf.reduce_all((coords >= 0.0) & (coords <= norm), -1)


def boxes_iou(boxes1, boxes2):
    boxes1 = center_point_to_coordinates(boxes1)
    boxes2 = center_point_to_coordinates(boxes2)

    y1_top, x1_left, y1_bottom, x1_right = (tf.squeeze(i, -1) for i in tf.split(boxes1, 4, -1))
    t2_top, x2_left, y2_bottom, x2_right = (tf.squeeze(i, -1) for i in tf.split(boxes2, 4, -1))

    xI1 = tf.maximum(x1_left, x2_left)
    xI2 = tf.minimum(x1_right, x2_right)

    yI1 = tf.minimum(y1_bottom, y2_bottom)
    yI2 = tf.maximum(y1_top, t2_top)

    inter_area = tf.maximum((xI2 - xI1), 0) * tf.maximum((yI1 - yI2), 0)

    bboxes1_area = (x1_right - x1_left) * (y1_bottom - y1_top)
    bboxes2_area = (x2_right - x2_left) * (y2_bottom - t2_top)

    union = (bboxes1_area + bboxes2_area) - inter_area
    return tf.math.divide_no_nan(inter_area, union)

import tensorflow as tf
from tensorflow.python.keras.losses import huber_loss


def rpn_loss(results, gt, rate=100):
    """
    Loss for Region Proposal Network:
    - logistic regression for all detections,
    - bounding box loss between roi (predicted) and bounding box (gt) based on anchor

    predictions: [N]
    rois: [N, 4]
    anchors: [N, 4]
    gt_bbox: [N, 4] (because it is nearest gt for each anchor)
    gt_detection: [N]
    """
    predictions, rois, anchors, _ = results
    gt_bbox, _, _ = gt

    d_loss = detection_loss(predictions, True)
    b_loss = bbox_loss(rois, anchors, gt_bbox)

    return d_loss + rate * b_loss


def frcnn_loss(results, gt, rate=100):
    """
    Loss for Faster R-CNN prediction (does not include RPN):
    - Cross-entropy for classification of the object,
    - bounding box loss between roi (predicted) and bounding box (gt) based on anchor

    predictions: [N, classes]
    rois: [N, 4]
    anchors: [N, 4]
    gt_bbox: [N, 4] (because it is nearest gt for each anchor)
    gt_label: [N]
    """
    predictions, rois, anchors, _ = results
    gt_bbox, gt_label, _ = gt

    d_loss = classification_loss(gt_label, predictions)
    b_loss = bbox_loss(rois, anchors, gt_bbox)

    return d_loss + rate * b_loss


def detection_loss(outputs, positive):
    """
    Logistic regression for detecting an object in one value
    :param outputs: [None] (not logits - sigmoid output)
    :param positive: whether detection was positive or not
    """
    labels = tf.ones_like(outputs) if positive else tf.zeros_like(outputs)
    return tf.reduce_mean(tf.keras.losses.binary_crossentropy(labels, outputs))


def classification_loss(labels, outputs):
    """
    Simple cross entropy between labels and outputs
    :param labels: [None]
    :param outputs: [None, classes] (logits)
    """
    return tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(labels, outputs, True))


def bbox_loss(rois, anchors, gt_bbox):
    """
    :param rois: predicted by network, [N, 4]
    :param anchors: handcrafted anchors, [N, 4]
    :param gt_bbox: ground truth bounding box (corresponding for each roi): [N, 4]
    :return:
    """

    bbox_rois = tf.split(rois, 4, -1)
    bbox_anchors = tf.split(anchors, 4, -1)
    bbox_gt = tf.split(gt_bbox, 4, -1)

    t = _adjust_bbox(bbox_rois, bbox_anchors)
    t_gt = _adjust_bbox(bbox_gt, bbox_anchors)

    return tf.reduce_mean(huber_loss(t_gt, t))


def _adjust_bbox(bbox1, bbox2):
    """
    Convert bounding boxes as in original paper
    """
    y1, x1, h1, w1 = bbox1
    y2, x2, h2, w2 = bbox2

    return tf.concat([
        (y1 - y2) / h2,
        (x1 - x2) / w2,
        tf.math.log(tf.maximum(h1 / h2, 1e-6)),
        tf.math.log(tf.maximum(w1 / w2, 1e-6))
    ], -1)

import tensorflow as tf


def rpn_loss(rpn_outputs, anchors, gt_outputs, rate=100, **kwargs):
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
    predictions, rois = rpn_outputs
    gt_bbox, gt_detection = gt_outputs

    d_loss = detection_loss(gt_detection, predictions)
    b_loss = bbox_loss(rois, anchors, gt_bbox, gt_detection[:, tf.newaxis])

    return d_loss + rate * b_loss


def faster_rcnn_loss(frcnn_outputs, anchors, gt_outputs, rate=100, **kwargs):
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
    predictions, rois = frcnn_outputs
    gt_bbox, gt_label, _ = gt_outputs

    d_loss = classification_loss(gt_label, predictions)
    b_loss = bbox_loss(rois, anchors, gt_bbox)

    return d_loss + rate * b_loss


def detection_loss(labels, outputs):
    """
    Logistic regression for detecting an object in one value
    :param labels: [None]
    :param outputs: [None]
    """
    return tf.losses.log_loss(labels, outputs)


def classification_loss(labels, outputs):
    """
    Simple cross entropy between labels and outputs
    :param labels: [None]
    :param outputs: [None, classes]
    """
    return tf.losses.sparse_softmax_cross_entropy(labels, outputs)


def bbox_loss(rois, anchors, gt_bbox, gt_detection=1.0):
    """
    :param rois: predicted by network, [N, 4]
    :param anchors: handcrafted anchors, [N, 4]
    :param gt_bbox: ground truth bounding box (corresponding for each roi): [N, 4]
    :param gt_detection: whether roi contains object or not [N], if not set, then all pairs equally counted
    :return:
    """

    bbox_rois = tf.split(rois, 4, -1)
    bbox_anchors = tf.split(anchors, 4, -1)
    bbox_gt = tf.split(gt_bbox, 4, -1)

    t = _adjust_bbox(bbox_rois, bbox_anchors)
    t_gt = _adjust_bbox(bbox_gt, bbox_anchors)

    return tf.losses.huber_loss(t_gt, t, gt_detection)


def _adjust_bbox(bbox1, bbox2):
    """
    Convert bounding boxes as in original paper
    """
    y1, x1, h1, w1 = bbox1
    y2, x2, h2, w2 = bbox2

    return tf.concat([
        (y1 - y2) / h2,
        (x1 - x2) / w2,
        tf.log(tf.maximum(h1 / h2, 1e-6)),
        tf.log(tf.maximum(w1 / w2, 1e-6))
    ], -1)

import tensorflow as tf

from ..utils.boxes import get_anchors, crop_anchors, filter_anchors, boxes_iou
from ..losses import boxes_loss


class RegionProposalNetwork(tf.keras.Model):

    def __init__(self, rpn_features, anchor_base, downscale_rate, num_scales, ratios=[[1, 1], [2, 1], [1, 2]], momentum=0.9, dropout=0.3):
        super().__init__()
        self.anchor_base = anchor_base
        self.downscale_rate = downscale_rate
        self.num_scales = num_scales
        self.ratios = ratios
        self.num_anchors = num_scales * len(ratios)

        self.extractor = tf.keras.Sequential([
            tf.keras.layers.Conv2D(rpn_features, 1, 1, 'same'),
            tf.keras.layers.BatchNormalization(momentum=momentum),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Conv2D(self.num_anchors * 5, 1, 1, 'same')
        ])

    def call(self, inputs, training=None, filter_outbound=True):
        # run rois refinement and detection on image features
        features = self.extractor(inputs, training=training)

        # reshape into form of [anchors_per_pixel x 5]
        features_shape = tf.concat([tf.shape(features)[:3], [self.num_anchors, 5]], 0)
        features = tf.reshape(features, features_shape)

        # spit last axis into detection and refinements parts
        predictions = features[:, :, :, :, 0]
        rois_refinements = features[:, :, :, :, 1:]

        # produce anchors for input features (based on downscale rate which should come from backbone)
        anchors = get_anchors(tf.shape(inputs)[1:3], self.anchor_base, self.downscale_rate, self.num_scales, self.ratios)
        anchors = tf.tile(anchors[tf.newaxis], (tf.shape(rois_refinements)[0], 1, 1, 1, 1))

        # produce assignments (says which anchor comes from which image) and flatten all data
        assignments = tf.reshape(tf.tile(tf.range(features_shape[0])[:, tf.newaxis], (1, tf.reduce_prod(features_shape[1:4]))), [-1])
        predictions = tf.reshape(predictions, [-1])
        rois_refinements = tf.reshape(rois_refinements, [-1, 4])
        anchors = tf.reshape(anchors, [-1, 4])

        norm = tf.cast(tf.concat([features_shape[1:3], features_shape[1:3]], 0)[tf.newaxis], tf.float32) * self.downscale_rate

        # filter or crop outliers
        if filter_outbound:
            mask = filter_anchors(anchors, norm)
            predictions = tf.boolean_mask(predictions, mask)
            anchors = tf.boolean_mask(anchors, mask)
            rois = anchors + tf.boolean_mask(rois_refinements, mask)
            assignments = tf.boolean_mask(assignments, mask)
        else:
            anchors = crop_anchors(anchors, norm)
            rois = anchors + rois_refinements

        return predictions, rois, anchors, assignments


def rpn_candidates(boxes, num_boxes, anchors, assignments, positive_threshold=0.7, negative_threshold=0.3):
    # calculates iou between boxes and anchors
    mask = tf.sequence_mask(tf.gather(num_boxes, assignments))
    iou = boxes_iou(tf.gather(boxes, assignments), anchors[:, tf.newaxis])

    # find best score for each box
    best_anchors = tf.reduce_max(tf.ragged.stack_dynamic_partitions(iou, assignments, tf.reduce_max(assignments) + 1), 1)
    best = iou == tf.gather(best_anchors, assignments)

    # find all positive matches (above threshold) and convert into list of indices
    highest = iou > positive_threshold
    positive = tf.where((best | highest) & mask)

    # find all negative anchors (below threshold)
    negative = tf.where(tf.reduce_max(iou, -1) < negative_threshold)[:, 0]

    return positive, negative


def rpn_positive_step(candidates, boxes, predictions, rois, anchors, assignments, num_samples):
    # sample N times from candidates and take appropriate anchors, rois and boxes
    samples = tf.gather(candidates, tf.random.uniform([num_samples], 0, tf.shape(candidates)[0], tf.int32))
    samples, samples_boxes = samples[:, 0], samples[:, 1]

    boxes = tf.gather_nd(boxes, tf.stack([tf.cast(tf.gather(assignments, samples), tf.int64), samples_boxes], 1))
    predictions = tf.gather(predictions, samples)[..., tf.newaxis]
    rois = tf.gather(rois, samples)
    anchors = tf.gather(anchors, samples)

    # calculate loss for detection and boxes and find accuracy
    p_loss = tf.losses.binary_crossentropy(tf.ones_like(predictions, tf.float32), predictions, from_logits=True)
    b_loss = boxes_loss(boxes, rois, anchors)
    hits = predictions[..., 0] > 0

    return p_loss, b_loss, hits


def rpn_negative_step(candidates, predictions, num_samples):
    # sample N times from candidates and take appropriate predictions
    samples = tf.gather(candidates, tf.random.uniform([num_samples], 0, tf.shape(candidates)[0], tf.int32))
    predictions = tf.gather(predictions, samples)[..., tf.newaxis]

    # calculate only detection loss and accuracy
    p_loss = tf.losses.binary_crossentropy(tf.zeros_like(predictions, tf.float32), predictions, from_logits=True)
    hits = predictions[..., 0] < 0

    return p_loss, hits


def rpn_step(boxes, num_boxes, predictions, rois, anchors, assignments, num_samples):
    # find positive matches and negative anchors
    p_candidates, n_candidates = rpn_candidates(boxes, num_boxes, anchors, assignments)

    # calculate loss for positive matches
    p_p_loss, p_b_loss, p_hits = rpn_positive_step(p_candidates, boxes, predictions, rois, anchors, assignments, num_samples)

    # calculate loss for negatice anchors
    n_p_loss, n_hits = rpn_negative_step(n_candidates, predictions, num_samples)

    # return loss and accuracy
    rpn_loss = tf.reduce_mean(p_p_loss + p_b_loss + n_p_loss)
    rpn_accuracy = tf.reduce_mean(tf.cast(tf.concat([p_hits, n_hits], 0), tf.float32))

    return rpn_loss, rpn_accuracy

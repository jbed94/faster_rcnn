import tensorflow as tf

from ..layers import ROIAlign, RegionProposalNetwork
from ..losses import boxes_loss
from ..utils.boxes import boxes_iou, center_point_to_coordinates


class FasterRCNN(tf.keras.Model):

    def __init__(self, num_classes, anchor_base=128, downscale_rate=32, num_scales=3, ratios=[[1, 1], [2, 1], [1, 2]], rpn_features=256, frcnn_features=256, align_shape=(7, 7),
                 align_samples=2, momentum=0.9, dropout=0.3, detection_threshold=0.7):
        super().__init__()

        self.downscale_rate = downscale_rate
        self.detection_threshold = detection_threshold
        self.cnn = tf.keras.applications.ResNet50V2(include_top=False)
        self.rpn = RegionProposalNetwork(rpn_features, anchor_base, self.downscale_rate, num_scales, ratios, momentum, dropout)
        self.roi_align = ROIAlign(align_shape, align_samples)
        self.extractor = tf.keras.Sequential([
            tf.keras.layers.Dense(frcnn_features),
            tf.keras.layers.BatchNormalization(momentum=momentum),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(4 + num_classes)
        ])

    def call(self, inputs, training=None, detector=None, filter_outbound=True):
        shape = tf.cast(tf.shape(inputs)[1:3], tf.float32)
        boxes_norm = tf.concat([shape, shape], 0)[tf.newaxis]

        inputs = inputs * 2.0 - 1.0
        images_features = self.cnn(inputs, training=training)
        rpn_predictions, rpn_rois, rpn_anchors, rpn_assignments = self.rpn(images_features, training=training, filter_outbound=filter_outbound)

        if detector is not None:
            indices = detector(rpn_predictions, rpn_rois, rpn_anchors, rpn_assignments)
        else:
            indices = tf.where(rpn_predictions > self.detection_threshold)

        predictions = tf.gather_nd(rpn_predictions, indices)
        rois = tf.gather_nd(rpn_rois, indices)
        anchors = tf.gather_nd(rpn_anchors, indices)
        assignments = tf.gather_nd(rpn_assignments, indices)

        anchors_shifted = center_point_to_coordinates(anchors) + tf.cast(assignments[:, tf.newaxis], tf.float32) * boxes_norm
        selected = tf.image.non_max_suppression(anchors_shifted, tf.nn.sigmoid(predictions), tf.shape(anchors_shifted)[0], 0.7)

        rois = tf.gather(rois, selected)
        anchors = tf.gather(anchors, selected)
        assignments = tf.gather(assignments, selected)

        features = tf.reduce_mean(self.roi_align(images_features, rois / boxes_norm, assignments), [1, 2])
        features = self.extractor(features, training=training)

        rois = rois + features[:, :4]
        predictions = features[:, 4:]

        return {
            'rpn': {
                'predictions': rpn_predictions,
                'rois': rpn_rois,
                'anchors': rpn_anchors,
                'assignments': rpn_assignments
            },
            'frcnn': {
                'predictions': predictions,
                'rois': rois,
                'anchors': anchors,
                'assignments': assignments,
            }
        }


def rpn_candidates(boxes, num_boxes, anchors, assignments, positive_threshold=0.7, negative_threshold=0.3):
    mask = tf.sequence_mask(tf.gather(num_boxes, assignments))
    iou = boxes_iou(tf.gather(boxes, assignments), anchors[:, tf.newaxis])
    best_anchors = tf.reduce_max(tf.ragged.stack_dynamic_partitions(iou, assignments, tf.reduce_max(assignments) + 1), 1)
    best = iou == tf.gather(best_anchors, assignments)
    highest = iou > positive_threshold

    positive = tf.where((best | highest) & mask)
    negative = tf.where(tf.reduce_max(iou, -1) < negative_threshold)[:, 0]

    return positive, negative


def rpn_positive_step(candidates, boxes, predictions, rois, anchors, assignments, num_samples):
    samples = tf.gather(candidates, tf.random.uniform([num_samples], 0, tf.shape(candidates)[0], tf.int32))
    samples, samples_boxes = samples[:, 0], samples[:, 1]

    boxes = tf.gather_nd(boxes, tf.stack([tf.cast(tf.gather(assignments, samples), tf.int64), samples_boxes], 1))
    predictions = tf.gather(predictions, samples)[..., tf.newaxis]
    rois = tf.gather(rois, samples)
    anchors = tf.gather(anchors, samples)

    p_loss = tf.losses.binary_crossentropy(tf.ones_like(predictions, tf.float32), predictions, from_logits=True)
    b_loss = boxes_loss(boxes, rois, anchors)
    hits = predictions[..., 0] > 0

    return p_loss, b_loss, hits


def rpn_negative_step(candidates, predictions, num_samples):
    samples = tf.gather(candidates, tf.random.uniform([num_samples], 0, tf.shape(candidates)[0], tf.int32))
    predictions = tf.gather(predictions, samples)[..., tf.newaxis]

    p_loss = tf.losses.binary_crossentropy(tf.zeros_like(predictions, tf.float32), predictions, from_logits=True)
    hits = predictions[..., 0] < 0

    return p_loss, hits


def rpn_step(boxes, num_boxes, predictions, rois, anchors, assignments, num_samples):
    p_candidates, n_candidates = rpn_candidates(boxes, num_boxes, anchors, assignments)
    p_p_loss, p_b_loss, p_hits = rpn_positive_step(p_candidates, boxes, predictions, rois, anchors, assignments, num_samples)
    n_p_loss, n_hits = rpn_negative_step(n_candidates, predictions, num_samples)
    rpn_loss = tf.reduce_mean(p_p_loss + p_b_loss + n_p_loss)
    rpn_accuracy = tf.reduce_mean(tf.cast(tf.concat([p_hits, n_hits], 0), tf.float32))
    return rpn_loss, rpn_accuracy


def frcnn_match(boxes, anchors, assignments):
    boxes = tf.gather(boxes, assignments)
    iou = boxes_iou(boxes, anchors[:, tf.newaxis])
    return tf.stack([tf.range(tf.shape(iou, tf.int64)[0]), tf.argmax(iou, -1)], 1)


def frcnn_step(boxes, labels, predictions, rois, anchors, assignments):
    matches = frcnn_match(boxes, anchors, assignments)
    labels = tf.gather_nd(labels, matches)
    boxes = tf.gather_nd(boxes, matches)

    p_loss = tf.losses.sparse_categorical_crossentropy(labels, predictions, from_logits=True)
    b_loss = boxes_loss(boxes, rois, anchors)

    loss = tf.reduce_mean(p_loss + b_loss)
    accuracy = tf.cast(tf.argmax(predictions, -1) == labels, tf.float32)

    return loss, accuracy


def prepare_query_and_train(frcnn, optimizer, num_samples):
    def inference(images, boxes, num_boxes, labels, training):
        def detector(predictions, rois, anchors, assignments):
            mask = tf.sequence_mask(tf.gather(num_boxes, assignments))
            iou = boxes_iou(tf.gather(boxes, assignments), anchors[:, tf.newaxis])
            best_anchors = tf.reduce_max(tf.ragged.stack_dynamic_partitions(iou, assignments, tf.reduce_max(assignments) + 1), 1)
            best = tf.reduce_any((iou == tf.gather(best_anchors, assignments)) & mask, -1)
            highest = tf.reduce_max(iou, 1) > 0.7
            return tf.where(best | highest)

        output = frcnn(images, training, detector)

        _rpn_loss, _rpn_accuracy = rpn_step(boxes, num_boxes, output['rpn']['predictions'], output['rpn']['rois'], output['rpn']['anchors'], output['rpn']['assignments'],
                                            num_samples)
        _frcnn_loss, _frcnn_accuracy = frcnn_step(boxes, labels, output['frcnn']['predictions'], output['frcnn']['rois'], output['frcnn']['anchors'],
                                                  output['frcnn']['assignments'])

        # total loss
        total_loss = _rpn_loss + _frcnn_loss

        return output, (total_loss, _rpn_accuracy, _frcnn_accuracy)

    @tf.function(input_signature=(
            tf.TensorSpec([None, None, None, 3], tf.float32),
            tf.TensorSpec([None, None, 4], tf.float32),
            tf.TensorSpec([None], tf.int32),
            tf.TensorSpec([None, None], tf.int64)
    ))
    def train(images, boxes, num_boxes, labels):
        with tf.GradientTape() as tape:
            frcnn_outputs, (loss, rpn_accuracy, frcnn_accuracy) = inference(images, boxes, num_boxes, labels, True)

        vars = frcnn.trainable_weights
        grads = tape.gradient(loss, vars)
        optimizer.apply_gradients(zip(grads, vars))

        return frcnn_outputs, (loss, rpn_accuracy, frcnn_accuracy)

    @tf.function(input_signature=(
            tf.TensorSpec([None, None, None, 3], tf.float32),
            tf.TensorSpec([None, None, 4], tf.float32),
            tf.TensorSpec([None], tf.int32),
            tf.TensorSpec([None, None], tf.int64)
    ))
    def query(images, boxes, num_boxes, labels):
        return inference(images, boxes, num_boxes, labels, False)

    return query, train

import tensorflow as tf

from ..layers import ROIAlign
from ..layers.rpn import RegionProposalNetwork, rpn_step
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
        # get input image shape and normalization tensor
        shape = tf.cast(tf.shape(inputs)[1:3], tf.float32)
        boxes_norm = tf.concat([shape, shape], 0)[tf.newaxis]

        # run pretrained backbone
        inputs = inputs * 2.0 - 1.0
        images_features = self.cnn(inputs, training=training)

        # run region proposal network
        rpn_predictions, rpn_rois, rpn_anchors, rpn_assignments = self.rpn(images_features, training=training, filter_outbound=filter_outbound)

        # decide if use external detector (e.g. during training) or simply take detection results
        if detector is not None:
            indices = detector(rpn_predictions, rpn_rois, rpn_anchors, rpn_assignments)
        else:
            indices = tf.where(rpn_predictions > self.detection_threshold)

        # get only positive anchors (with rois and predictions)
        predictions = tf.gather_nd(rpn_predictions, indices)
        rois = tf.gather_nd(rpn_rois, indices)
        anchors = tf.gather_nd(rpn_anchors, indices)
        assignments = tf.gather_nd(rpn_assignments, indices)

        # HACK:
        # batch non_max_suppression for batch processing (simply shift anchors according to assignment)
        anchors_shifted = center_point_to_coordinates(anchors) + tf.cast(assignments[:, tf.newaxis], tf.float32) * boxes_norm
        selected = tf.image.non_max_suppression(anchors_shifted, tf.nn.sigmoid(predictions), tf.shape(anchors_shifted)[0], 0.7)

        # get positive anchors and rois (without predictions!)
        rois = tf.gather(rois, selected)
        anchors = tf.gather(anchors, selected)
        assignments = tf.gather(assignments, selected)

        # get features for produced rois and perform roi refinement and classification
        features = tf.reduce_mean(self.roi_align(images_features, rois / boxes_norm, assignments), [1, 2])
        features = self.extractor(features, training=training)

        # get outpus
        rois = rois + features[:, :4]
        predictions = features[:, 4:]

        # return both results for training
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


def frcnn_match(boxes, anchors, assignments):
    boxes = tf.gather(boxes, assignments)
    iou = boxes_iou(boxes, anchors[:, tf.newaxis])
    # todo: check if there is proper assignments matching
    return tf.stack([assignments, tf.argmax(iou, -1, output_type=tf.int32)], 1)


def frcnn_step(boxes, labels, predictions, rois, anchors, assignments):
    # find closest boxes to anchors"
    matches = frcnn_match(boxes, anchors, assignments)

    # get corresponding boxes and labels
    labels = tf.gather_nd(labels, matches)
    boxes = tf.gather_nd(boxes, matches)

    # run boxes loss and classification loss
    b_loss = boxes_loss(boxes, rois, anchors)
    p_loss = tf.losses.sparse_categorical_crossentropy(labels, predictions, from_logits=True)

    # return loss and accuracy
    loss = tf.reduce_mean(p_loss + b_loss)
    accuracy = tf.cast(tf.argmax(predictions, -1) == labels, tf.float32)

    return loss, accuracy


def prepare_query_and_train(frcnn, optimizer, num_samples):
    def inference(images, boxes, num_boxes, labels, training):
        # define external detector for taking positive anchors
        def detector(predictions, rois, anchors, assignments):
            mask = tf.sequence_mask(tf.gather(num_boxes, assignments))
            iou = boxes_iou(tf.gather(boxes, assignments), anchors[:, tf.newaxis])
            best_anchors = tf.reduce_max(tf.ragged.stack_dynamic_partitions(iou, assignments, tf.reduce_max(assignments) + 1), 1)
            best = tf.reduce_any((iou == tf.gather(best_anchors, assignments)) & mask, -1)
            highest = tf.reduce_max(iou, 1) > 0.7
            return tf.where(best | highest)

        # run faster r-cnn
        output = frcnn(images, training, detector)

        # rpn and frcnn train steps
        _rpn_loss, _rpn_accuracy = rpn_step(boxes, num_boxes, output['rpn']['predictions'], output['rpn']['rois'], output['rpn']['anchors'], output['rpn']['assignments'],
                                            num_samples)
        _frcnn_loss, _frcnn_accuracy = frcnn_step(boxes, labels, output['frcnn']['predictions'], output['frcnn']['rois'], output['frcnn']['anchors'],
                                                  output['frcnn']['assignments'])

        # total loss
        total_loss = _rpn_loss + _frcnn_loss

        return output, (total_loss, _rpn_accuracy, _frcnn_accuracy)

    # @tf.function(input_signature=(
    #         tf.TensorSpec([None, None, None, 3], tf.float32),
    #         tf.TensorSpec([None, None, 4], tf.float32),
    #         tf.TensorSpec([None], tf.int32),
    #         tf.TensorSpec([None, None], tf.int64)
    # ))
    def train(images, boxes, num_boxes, labels):
        with tf.GradientTape() as tape:
            frcnn_outputs, (loss, rpn_accuracy, frcnn_accuracy) = inference(images, boxes, num_boxes, labels, True)

        vars = frcnn.trainable_weights
        grads = tape.gradient(loss, vars)
        optimizer.apply_gradients(zip(grads, vars))

        return frcnn_outputs, (loss, rpn_accuracy, frcnn_accuracy)

    # @tf.function(input_signature=(
    #         tf.TensorSpec([None, None, None, 3], tf.float32),
    #         tf.TensorSpec([None, None, 4], tf.float32),
    #         tf.TensorSpec([None], tf.int32),
    #         tf.TensorSpec([None, None], tf.int64)
    # ))
    def query(images, boxes, num_boxes, labels):
        return inference(images, boxes, num_boxes, labels, False)

    return query, train

import tensorflow as tf

from ..utils.boxes import get_anchors, crop_anchors, filter_anchors


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
        features = self.extractor(inputs, training=training)

        features_shape = tf.concat([tf.shape(features)[:3], [self.num_anchors, 5]], 0)
        features = tf.reshape(features, features_shape)

        predictions = features[:, :, :, :, 0]
        rois_refinements = features[:, :, :, :, 1:]

        anchors = get_anchors(tf.shape(inputs)[1:3], self.anchor_base, self.downscale_rate, self.num_scales, self.ratios)
        anchors = tf.tile(anchors[tf.newaxis], (tf.shape(rois_refinements)[0], 1, 1, 1, 1))

        assignments = tf.reshape(tf.tile(tf.range(features_shape[0])[:, tf.newaxis], (1, tf.reduce_prod(features_shape[1:4]))), [-1])
        predictions = tf.reshape(predictions, [-1])
        rois_refinements = tf.reshape(rois_refinements, [-1, 4])
        anchors = tf.reshape(anchors, [-1, 4])

        norm = tf.cast(tf.concat([features_shape[1:3], features_shape[1:3]], 0)[tf.newaxis], tf.float32) * self.downscale_rate

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

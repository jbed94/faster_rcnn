import tensorflow as tf

from .utils import _sample_many, center_point_to_coordnates


class RegionProposalNetwork(tf.keras.Model):
    """
    Region Proposal Network takes as an input features (e.g. output of some features extractor like
    ResNet50, VGG16, etc.) and produces RoIs for detected objects.
    Algorithm consists of following steps:
    1. Extract additional features (256, 512 or other arbitrary chosen number of features),
    2. For each "pixel" and each **anchor** predicts whether object is detected and in what RoI,
    3. **Anchor** is a handcrafted RoI with particular size (scale and aspect ratio). E.g we can have 9
    anchor templates (128x128, 128x256, 256x128, etc.) and we apply them to each position in features tensor.
    Each position corresponds to the position in original image (e.g. if original image: 256x256, and
    features shape: 64x64 then each "shift" in features step is equal to 4 "shifts" in original image.
    This way we get 9 anchors [y, x, height, width] for every position in features tensor.
    4. Remove all anchors which cross boundary,
    5. To determine whether particular anchor contains an object we compare all of anchors with ground truth,
    calculating IoU (our score; additionally score = 1 for each closest anchor to the gt),
    6. Again filter anchors according to the non_max_non_max_suppression,
    7. Sample sane number of **positive** (detected object) and **negative** anchors and perform
    optimization step,

    Remarks:
    - in inference instead of IoU we take predicted value from network,
    - to classify anchor as "positive" score need to be >= 0.7,
    - to classify anchor ad "negatie" score need to be <= 0.3,
    - anchors with score between (0.3, 0.7) are not used in training,
    - in inference we take as output only anchors with score >= 0.7,
    - in inference we only crop cross boundary anchors, not remove
    """

    def __init__(self, rpn_features,
                 anchor_num_scales=3,
                 total_anchor_overlap_rate=0.9,
                 non_max_suppression_iou_threshold=0.7,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.rpn_features = rpn_features
        self.anchor_num_scales = anchor_num_scales
        self.total_anchor_overlap_rate = total_anchor_overlap_rate
        self.non_max_suppression_iou_threshold = non_max_suppression_iou_threshold

        self.anchor_cross_boundary_filter = CrossBoundaryAnchorFilter()
        self.anchor_cross_boundary_crop = CrossBoundaryAnchorCrop()
        self.anchor_non_max_suppression_filter = NonMaxSuppressionAnchorFilter(self.non_max_suppression_iou_threshold)

        self.extractor = tf.keras.Sequential([
            tf.keras.layers.Conv2D(self.rpn_features, 3, 1, 'same', activation='relu'),
            tf.keras.layers.Conv2D(self.anchor_num_scales * 3 * 5, 1, 1, 'same')
        ])

    def call(self, inputs, training=None, original_shape=None, mask=None):
        # get batch size for tile fo anchors and image assignments
        batch_size = tf.shape(inputs)[0]

        # run features extraction and output conv consisting of 5 filters (1 for classification and 4 for bbox)
        features = self.extractor(inputs)
        # return also features shape to know how many samples (anchors) do we need
        features_shape = tf.shape(features).numpy()[1:3]

        # get detection (1 vals - logical regression), and bbox (4 vals)
        f_shape = tf.shape(features)
        features = tf.reshape(features, [f_shape[0], f_shape[1], f_shape[2], -1, 5])
        predictions = tf.sigmoid(tf.reshape(features[:, :, :, :, :1], [-1]))
        rois_refinements = tf.reshape(features[:, :, :, :, 1:], [-1, 4])

        # build all possible anchors
        # get anchor templates (total number = 3 * num_scales, rates per scale: 2:2, 1:2, 2:1)
        anchor_templates = _prepare_anchor_templates(original_shape, self.anchor_num_scales,
                                                     self.total_anchor_overlap_rate)

        # for particular features (or rather features shape) prepare all possible anchors (y, x, height, width)
        anchors = _prepare_features_anchors(features_shape, original_shape, anchor_templates)
        anchors = tf.tile(anchors, [batch_size, 1])
        rois = anchors + rois_refinements

        # produce image assignments for each anchor -> image
        image_assignments = tf.range(0, batch_size)[:, tf.newaxis, tf.newaxis, tf.newaxis]
        image_assignments = tf.tile(image_assignments, [1, f_shape[1], f_shape[2], self.anchor_num_scales * 3])
        image_assignments = tf.reshape(image_assignments, [-1])

        # run filtering / cropping
        if training:
            selected_anchors = self.anchor_cross_boundary_filter(anchors, original_shape)
            predictions, rois, anchors, image_assignments = \
                _sample_many(selected_anchors, predictions, rois, anchors, image_assignments)
        else:
            anchors = self.anchor_cross_boundary_crop(anchors, original_shape)

        return predictions, rois, anchors, image_assignments

    def filter(self, anchors, scores, image_assignments, original_shape, to_filter):
        # first find interesting anchors
        selected_anchors = self.anchor_non_max_suppression_filter(anchors, scores, image_assignments, original_shape)

        # then gather corresponding positions from every tensor in to_filter list
        return _sample_many(selected_anchors, *to_filter)


class CrossBoundaryAnchorFilter(tf.keras.layers.Layer):
    """
    Cross Boundary Anchor Filter remove all anchors that cross boundary (any point of bbox) from given tensor
    """

    # noinspection PyMethodOverriding
    def call(self, anchors, original_shape, **kwargs):
        # 1. filter anchors which cross boundary
        y, x, h, w = tf.split(anchors, 4, -1)
        top, bottom, left, right = y - h, y + h, x - w, x + w
        tb = tf.logical_and(top > 0.0, bottom < original_shape[0])
        lr = tf.logical_and(left > 0.0, right < original_shape[1])
        inner_anchors = tf.squeeze(tf.logical_and(tb, lr), 1)

        # 2.1 first gather current filtered anchors and push them to non_max_suppression to increase efficiency
        selected_anchors = tf.squeeze(tf.where(inner_anchors), 1)

        return selected_anchors


class CrossBoundaryAnchorCrop(tf.keras.layers.Layer):
    """
    Cross Boundary Anchor Crop do similar as CrossBoundaryAnchorFilter but instead of removing only
    crops anchors which cross boundary
    """

    # noinspection PyMethodOverriding
    def call(self, anchors, original_shape, **kwargs):
        h, w = original_shape
        anchors = center_point_to_coordnates(anchors, True)
        anchors = tf.maximum(anchors, 0)
        anchors = tf.minimum(anchors, [[h, w, h, w]])
        anchors = center_point_to_coordnates(anchors, True)
        return anchors


class NonMaxSuppressionAnchorFilter(tf.keras.layers.Layer):
    """
    Compare every pair of anchors in a given set and remove those which overlaps with iou >= iou_threshold
    and have smaller score
    """

    def __init__(self, iou_threshold, **kwargs):
        super().__init__(False, None, tf.float32, **kwargs)
        self.iou_threshold = iou_threshold

    # noinspection PyMethodOverriding
    def call(self, anchors, scores, image_assignments, original_shape, **kwargs):
        h, w = original_shape
        # filter anchors according to the non_max_suppression
        anchors_coords = center_point_to_coordnates(anchors)

        # multibatch trick:
        shifts = tf.cast(image_assignments[:, tf.newaxis], tf.float32) * tf.convert_to_tensor([[h, w, h, w]],
                                                                                              tf.float32)
        anchors_coords += shifts

        selected_anchors = tf.image.non_max_suppression(anchors_coords, scores, tf.shape(anchors)[0],
                                                        self.iou_threshold)

        return selected_anchors


def _prepare_anchor_templates(original_shape, num_scales, total_overlap_rate=0.9):
    """
    Function for anchor templates preparation. For each scale we prepare 3 anchors in ratios 2:2, 2:1, 1:2.
    :param original_shape: shape of the original image
    :param num_scales: number of scales for anchors, each next scale is previous one multiplied by 2
    :param total_overlap_rate: how big should be biggest anchor - it is a multiplier of original_shape, so 0.9 means that biggest anchor will be size of 0.9 * original_shape
    :return: flatten list of anchors ([num_anchors, 2], 2 values for width and height)
    """
    # size of the biggest anchor have to be smaller than image size, otherwise will always cross boundary
    min_shape = min(original_shape)
    max_multiplier = float(2 ** (num_scales - 1))
    base_shape = min_shape * total_overlap_rate / max_multiplier

    # push anchors to the list (ratios: 2:2, 2:1, 1:2
    anchors = []
    for i in range(num_scales):
        anchors.append([base_shape * (2 ** i) / 2, base_shape * (2 ** i) / 2])
        anchors.append([base_shape * (2 ** i), base_shape * (2 ** i) / 2])
        anchors.append([base_shape * (2 ** i) / 2, base_shape * (2 ** i)])

    # after all we want to have it in a tensor (convert to float32 because of gather gpu implementation)
    return tf.convert_to_tensor(anchors, tf.float32)


def _prepare_features_anchors(output_shape, original_shape, anchor_templates):
    """
    Function for constructing all possible anchors (with real y, x, height and width) for particular features tensor
    :param output_shape: shape of the features tensor (an output from VGG/ ResNet or similar)
    :param original_shape: shape of the original image
    :param anchor_templates: templates of the anchors made by f.e. prepare_anchor_templates function
    :return: all possible anchors for particular feature tensor
    """
    # prepare meshgrid representing y and x position of one anchor per (super) pixel
    block_half_height = (original_shape[0] / float(output_shape[0])) / 2.0
    block_half_width = (original_shape[1] / float(output_shape[1])) / 2.0
    y = tf.linspace(block_half_height, original_shape[0] - block_half_height, output_shape[0])
    x = tf.linspace(block_half_width, original_shape[1] - block_half_width, output_shape[1])

    # shape: [f_shape[0], f_shape[1], 2]
    yx = tf.stack(tf.meshgrid(y, x), -1)

    # prepare y and x for many anchors per (super) pixel
    yx = tf.tile(yx[:, :, tf.newaxis, :], [1, 1, tf.shape(anchor_templates)[0], 1])
    yx = tf.cast(yx, tf.float32)

    # for each y, x prepare height and width of all the anchor templates (broadcast)
    hw = tf.ones_like(yx) * anchor_templates[tf.newaxis, tf.newaxis, :]

    # concat yx and hw
    yxhw = tf.concat([yx, hw], -1)
    yxhw = tf.reshape(yxhw, [-1, 4])

    return yxhw

from .losses import frcnn_loss, rpn_loss, detection_loss
from .roi import ROIAlign
from .rpn import RegionProposalNetwork, NonMaxSuppressionAnchorFilter
from .utils import *


class FasterRCNN(tf.keras.Model):
    """
    Faster R-CNN implementation in Keras (adjusted to eager execution).
    Model consists of 3 parts:
    - features extraction (like VGG, ResNet, etc. might be easily replaced with any network)
    - ROI Align algorithm,
    - prediction header - convolution (stride 2) -> Global Average Pooling -> prediction (class, bbox)

    Prediction header is executed on the output from ROI Align.

    Faster R-CNN pipeline differs when it's training, validation or simple inference.
    In both training and validation as a score we use IoU between anchors and ground truth bboxes whereas
    during simple inference object detection (0 - no object, 1 - object) is used as score.

    Additionally, in training or validation mode the model outputs additionally ground-truth bboxes and labels
    (so they have to be passed to call function for later detections assignments) and sampler
    function for RPN training. This sampler returns N randomly sampled positive samples (positice = detection),
    and M negative samples. Can be executed as follows: batch = sampler(N, M).
    Where batch is a tuple of (predictions, rois, anchors, gt_rois, gt_prediction).
    """

    def __init__(self,
                 num_classes,
                 frcnn_features=256,
                 rpn_features=256,
                 anchor_num_scales=3,
                 total_anchor_overlap_rate=0.9,
                 non_max_suppression_iou_threshold=0.7,
                 roi_align_output_size=(7, 7),
                 roi_align_samples=2,
                 detection_upper_threshold=0.7,
                 image_size=None,
                 optimizer=None):
        """
        :param num_classes: number of classes for classification detections,
        :param frcnn_features: convolution size made on output of ROI Align before final prediction
        :param rpn_features: convolution size made on extracted features before rpn prediction
        :param anchor_num_scales: number of anchor scales (anchors are generated automatically)
        :param total_anchor_overlap_rate: size of the biggest anchor, e.g. 0.9 will results with anchor of size min(image_width, image_height) * 0.9
        :param non_max_suppression_iou_threshold: threshold for filtering overlapping detected anchors
        :param filter_cross_boundary: whether filter cross-boundary rois or simply crop
        :param roi_align_output_size: size of the sample of the features for particular object, e.g. each detected object will have object_features [1, roi_align_output_size[0], roi_align_output_size[1], F]
        :param roi_align_samples: how many bilinear samples make for each output point (and the size of pooling)
        :param detection_upper_threshold: threshold for detecting object (made on scores)
        :param detection_lower_threshold: threshold for detecting not-object (made on scores)
        """
        super().__init__()

        self.optimizer = optimizer
        self.num_classes = num_classes
        self.frcnn_features = frcnn_features
        self.rpn_features = rpn_features
        self.anchor_num_scales = anchor_num_scales
        self.total_anchor_overlap_rate = total_anchor_overlap_rate
        self.non_max_suppression_iou_threshold = non_max_suppression_iou_threshold
        self.detection_upper_threshold = detection_upper_threshold
        self.image_size = image_size

        # features extraction network
        self.cnn = tf.keras.applications.ResNet50(weights=None, include_top=False, input_shape=self.image_size)
        self.rpn = RegionProposalNetwork(self.rpn_features,
                                         self.anchor_num_scales,
                                         self.total_anchor_overlap_rate,
                                         self.image_size,
                                         self.cnn.output_shape[1:])

        self.roi = ROIAlign(roi_align_output_size, roi_align_samples)

        # for creating one feature vector for context (output of CNN) and object (output of extractor)
        self.gap = tf.keras.layers.GlobalAveragePooling2D()

        # final features extraction and fast r-cnn predictions
        self.extractor = tf.keras.Sequential([
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(self.frcnn_features),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Dropout(0.5)
        ])
        self.predict_class = tf.keras.layers.Dense(self.num_classes)
        self.predict_roi = tf.keras.layers.Dense(4)

        self.anchor_non_max_suppression_filter = NonMaxSuppressionAnchorFilter(self.non_max_suppression_iou_threshold)

    def call(self, inputs, training=False, scores=None):
        # get a shape of the input image

        # extract features using cnn, if fine-tune then pass training argument,
        # otherwise always false (because of batch normalization adjustment in eager execution)
        context_features = self.cnn(inputs, training=training)
        scene_features = self.gap(context_features)

        # run Region Proposal Network with first filtering of the cross-boundary filter
        rpn_predictions, rpn_rois, rpn_anchors, rpn_ia = self.rpn(context_features, training=training)

        scores = rpn_predictions if scores is None else scores

        # filter according to the scores (removing overlapping anchors)
        rpn_active_anchors = tf.range(0, tf.shape(scores)[0])
        rpn_predictions, rpn_rois, rpn_anchors, rpn_ia, rpn_active_anchors, scores = \
            self.anchor_non_max_suppression_filter.filter(
                rpn_rois, scores, rpn_ia, self.image_size,
                [rpn_predictions, rpn_rois, rpn_anchors, rpn_ia, rpn_active_anchors, scores])
        positive_samples = tf.squeeze(tf.where(scores >= self.detection_upper_threshold), 1)

        # after RPN we need only positive detections
        frcnn_rois, frcnn_anchors, frcnn_ia, frcnn_active_anchors = \
            sample_many(positive_samples, rpn_rois, rpn_anchors, rpn_ia, rpn_active_anchors)

        # normalize rois to have in range between [0,1]
        h, w, _ = self.image_size
        norm_rois = frcnn_rois / tf.convert_to_tensor([[h, w, h, w]], tf.float32)

        # run roi align to extract object_features
        # size (example) [rois, 7, 7, X]]
        object_features = self.roi(context_features, norm_rois, box_indices=frcnn_ia)
        object_features = self.gap(object_features)

        # run final features extraction and predict classes and rois
        features = self.extractor(object_features, training=training)
        frcnn_predictions = self.predict_class(features)
        frcnn_rois_refinements = self.predict_roi(features)
        frcnn_rois = frcnn_rois_refinements + frcnn_rois

        # calculate visual representation for each object if necessary
        object_features = tf.concat([
            tf.gather(scene_features, frcnn_ia),
            object_features
        ], -1)

        frcnn_result = (frcnn_predictions, frcnn_rois, frcnn_anchors, frcnn_ia)
        rpn_result = (rpn_predictions, rpn_rois, rpn_anchors, rpn_ia)
        features = (scene_features, object_features)
        active_anchors = (frcnn_active_anchors, rpn_active_anchors)

        return frcnn_result, rpn_result, features, active_anchors

    @tf.function
    def call_supervised(self, inputs, training, scores):
        return self(inputs, training, scores)

    @tf.function
    def call_unsupervised(self, inputs, training):
        return self(inputs, training)

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, None, None, 3], dtype=tf.float32),
        tf.TensorSpec(shape=[None, None, 4], dtype=tf.float32),
        tf.TensorSpec(shape=[None, None], dtype=tf.int64),
        tf.TensorSpec(shape=[None], dtype=tf.int32)
    ])
    def train_step(self, images, object_bbox, object_label, num_objects):
        with tf.GradientTape() as tape:
            anchors = tf.tile(self.rpn.anchors_filtered, [tf.shape(images)[0], 1])
            gt_data = get_gt_data(anchors, object_bbox, object_label, num_objects, self.detection_upper_threshold)

            frcnn_result, rpn_result, features, active_anchors = self(images, training=True, scores=gt_data[-1])

            frcnn_gt = sample_many(active_anchors[0], *gt_data)
            rpn_gt = sample_many(active_anchors[1], *gt_data)

            rpn_p_results, rpn_p_gt, rpn_n_results, rpn_n_gt = rpn_sample(rpn_result, rpn_gt)

            frcnn_l = frcnn_loss(frcnn_result, frcnn_gt)
            rpn_p_l = rpn_loss(rpn_p_results, rpn_p_gt)
            rpn_n_l = detection_loss(rpn_n_results[0], False)

            model_loss = frcnn_l + rpn_p_l + rpn_n_l

        grads = tape.gradient(model_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        frcnn_accuracy = tf.keras.metrics.sparse_categorical_accuracy(frcnn_gt[1], frcnn_result[0])
        frcnn_accuracy = tf.reduce_mean(frcnn_accuracy)
        rpn_accuracy = tf.keras.metrics.binary_accuracy(
            tf.concat([tf.ones_like(rpn_p_results[0]), tf.zeros_like(rpn_n_results[0])], 0),
            tf.concat([rpn_p_results[0], rpn_n_results[0]], 0)
        )
        rpn_accuracy = tf.reduce_mean(rpn_accuracy)
        model_result = frcnn_result, rpn_result, features

        return model_loss, frcnn_accuracy, rpn_accuracy, model_result

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, None, None, 3], dtype=tf.float32),
        tf.TensorSpec(shape=[None, None, 4], dtype=tf.float32),
        tf.TensorSpec(shape=[None, None], dtype=tf.int64),
        tf.TensorSpec(shape=[None], dtype=tf.int32)
    ])
    def val_step(self, images, object_bbox, object_label, num_objects):
        anchors = tf.tile(self.rpn.anchors_filtered, [tf.shape(images)[0], 1])
        gt_data = get_gt_data(anchors, object_bbox, object_label, num_objects, self.detection_upper_threshold)

        frcnn_result, rpn_result, features, active_anchors = self(images, training=False, scores=gt_data[-1])

        frcnn_gt = sample_many(active_anchors[0], *gt_data)
        rpn_gt = sample_many(active_anchors[1], *gt_data)

        rpn_p_results, rpn_p_gt, rpn_n_results, rpn_n_gt = rpn_sample(rpn_result, rpn_gt)

        frcnn_l = frcnn_loss(frcnn_result, frcnn_gt)
        rpn_p_l = rpn_loss(rpn_p_results, rpn_p_gt)
        rpn_n_l = detection_loss(rpn_n_results[0], False)

        model_loss = frcnn_l + rpn_p_l + rpn_n_l

        frcnn_accuracy = tf.keras.metrics.sparse_categorical_accuracy(frcnn_gt[1], frcnn_result[0])
        frcnn_accuracy = tf.reduce_mean(frcnn_accuracy)
        rpn_accuracy = tf.keras.metrics.binary_accuracy(
            tf.concat([tf.ones_like(rpn_p_results[0]), tf.zeros_like(rpn_n_results[0])], 0),
            tf.concat([rpn_p_results[0], rpn_n_results[0]], 0)
        )
        rpn_accuracy = tf.reduce_mean(rpn_accuracy)
        model_result = frcnn_result, rpn_result, features

        return model_loss, frcnn_accuracy, rpn_accuracy, model_result

    @staticmethod
    def std_spec(num_classes, image_size):
        return {
            'num_classes': num_classes,
            'frcnn_features': 512,
            'rpn_features': 512,
            'anchor_num_scales': 3,
            'total_anchor_overlap_rate': 0.9,
            'non_max_suppression_iou_threshold': 0.4,
            'roi_align_output_size': (7, 7),
            'roi_align_samples': 2,
            'detection_upper_threshold': 0.7,
            'image_size': image_size
        }

    @staticmethod
    def clevr_spec():
        return {
            'num_classes': 96,
            'frcnn_features': 512,
            'rpn_features': 512,
            'anchor_num_scales': 3,
            'total_anchor_overlap_rate': 0.9,
            'non_max_suppression_iou_threshold': 0.4,
            'roi_align_output_size': (7, 7),
            'roi_align_samples': 2,
            'detection_upper_threshold': 0.7,
            'image_size': [320, 480, 3]
        }

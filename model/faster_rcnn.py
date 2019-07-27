import tensorflow as tf

from .roi import ROIAlign
from .rpn import RegionProposalNetwork
from .utils import get_gt_data, get_data_sampler, _sample_many


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
                 detection_lower_threshold=0.3,
                 fine_tune_features_extraction=False,
                 *args, **kwargs):
        """
        :param num_classes: number of classes for classification detections,
        :param frcnn_features: convolution size made on output of ROI Align before final prediction
        :param rpn_features: convolution size made on extracted features before rpn prediction
        :param anchor_num_scales: number of anchor scales (anchors are generated automatically)
        :param total_anchor_overlap_rate: size of the biggest anchor, e.g. 0.9 will results with anchor of size min(image_width, image_height) * 0.9
        :param non_max_suppression_iou_threshold: threshold for filtering overlapping detected anchors
        :param roi_align_output_size: size of the sample of the features for particular object, e.g. each detected object will have object_features [1, roi_align_output_size[0], roi_align_output_size[1], F]
        :param roi_align_samples: how many bilinear samples make for each output point (and the size of pooling)
        :param detection_upper_threshold: threshold for detecting object (made on scores)
        :param detection_lower_threshold: threshold for detecting not-object (made on scores)
        :param fine_tune_features_extraction: whether fine-tune features extraction network
        """
        super().__init__(*args, **kwargs)

        self.num_classes = num_classes
        self.frcnn_features = frcnn_features
        self.rpn_features = rpn_features
        self.anchor_num_scales = anchor_num_scales
        self.total_anchor_overlap_rate = total_anchor_overlap_rate
        self.non_max_suppression_iou_threshold = non_max_suppression_iou_threshold
        self.detection_upper_threshold = detection_upper_threshold
        self.detection_lower_threshold = detection_lower_threshold
        self.fine_tune_features_extraction = fine_tune_features_extraction

        # features extraction network
        self.cnn = tf.keras.applications.ResNet50(include_top=False)
        # if not fine-tune then set parameters as not trainable
        if not self.fine_tune_features_extraction:
            for layer in self.cnn.layers:  # Freeze layers in pretrained model
                layer.trainable = False

        self.rpn = RegionProposalNetwork(self.rpn_features,
                                         self.anchor_num_scales,
                                         self.total_anchor_overlap_rate,
                                         self.non_max_suppression_iou_threshold)

        self.roi = ROIAlign(roi_align_output_size, roi_align_samples)

        # for creating one feature vector for context (output of CNN) and object (output of extractor)
        self.gap = tf.keras.layers.GlobalAveragePooling2D()

        # final features extraction and fast r-cnn predictions
        self.extractor = tf.keras.Sequential([
            tf.keras.layers.Dense(self.frcnn_features),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Dropout(0.2)
        ])
        self.predict_class = tf.keras.layers.Dense(self.num_classes)
        self.predict_roi = tf.keras.layers.Dense(4)

    def call(self, inputs, training=None, inference=False, gt_object_bbox=None, gt_object_label=None,
             gt_num_objects=None, return_visual_representations=False, mask=None):
        # get a shape of the input image
        original_shape = tf.shape(inputs).numpy()[1:3]

        # extract features using cnn, if fine-tune then pass training argument,
        # otherwise always false (because of batch normalization adjustment in eager execution)
        context_features = self.cnn(inputs, training=training and self.fine_tune_features_extraction)

        # run Region Proposal Network with first filtering of the cross-boundary filter
        rpn_predictions, rpn_rois, anchors, image_assignments = \
            self.rpn(context_features, training=training, original_shape=original_shape)

        # if training or validation then get ground-truth and calculate scores according to the IoU between
        # each anchor and ground truth
        rpn_sampler = None
        gt_outputs = None
        if training or not inference:
            # when training or validation additional ground-truth arguments need to be passed to network (call params)
            assert gt_object_bbox is not None
            assert gt_object_label is not None

            # get ground truth data (and assign nearest ground truth to each anchor)
            anchors_gt_bbox, anchors_gt_detection, scores, gt_object_label = \
                get_gt_data(anchors, gt_object_bbox, gt_object_label, image_assignments, gt_num_objects,
                            self.detection_upper_threshold)

            # run filtering according to the scores
            # here all overlapping anchors will be removed (respecting scores)
            tensors = [rpn_predictions, rpn_rois, anchors, scores, anchors_gt_bbox, anchors_gt_detection,
                       gt_object_label, image_assignments]
            tensors = self.rpn.filter(rpn_rois, scores, image_assignments, original_shape, tensors)
            rpn_predictions, rpn_rois, anchors, scores, anchors_gt_bbox, anchors_gt_detection, gt_object_label, image_assignments = tuple(
                tensors)

            # construct sampler for easy "positive" and "negative" anchors sampling
            # we do this, because there are a lot of negatives and only few positives,
            # so f.e. if we have |positive| = 3 and |negative| = 200 we can sample f.e. 32 times from
            # positive (one object may be sampled multiple times), and 32 times from negative
            rpn_sampler = get_data_sampler(
                scores, rpn_predictions, rpn_rois, anchors, anchors_gt_bbox, anchors_gt_detection, image_assignments,
                self.detection_upper_threshold,
                self.detection_lower_threshold)
            # output will additionally contains gt bboxes and object label (like tree, ball, etc)
            gt_outputs = [anchors_gt_bbox, gt_object_label, anchors_gt_detection]
            positive_predictions = tf.squeeze(tf.where(scores >= self.detection_upper_threshold), 1)
            gt_outputs = _sample_many(positive_predictions, *gt_outputs)
        else:
            # filter according to the scores (removing overlapping anchors)
            rpn_predictions, rpn_rois, anchors, image_assignments = \
                self.rpn.filter(rpn_rois, rpn_predictions, image_assignments, original_shape,
                                [rpn_predictions, rpn_rois, anchors, image_assignments])
            positive_predictions = tf.squeeze(tf.where(rpn_predictions >= self.detection_upper_threshold), 1)

        # after RPN we need only positive detections
        rpn_predictions = tf.gather(rpn_predictions, positive_predictions)
        rpn_rois = tf.gather(rpn_rois, positive_predictions)
        anchors = tf.gather(anchors, positive_predictions)
        image_assignments = tf.gather(image_assignments, positive_predictions)

        # normalize rois to have in range between [0,1]
        h, w = original_shape[0], original_shape[1]
        norm_rois = rpn_rois / tf.convert_to_tensor([[h, w, h, w]], tf.float32)

        # run roi align to extract object_features
        # size (example) [rois, 7, 7, 256]]
        object_features = self.roi(context_features, norm_rois, box_indices=image_assignments)
        object_features = self.gap(object_features)

        # run final features extraction and predict classes and rois
        features = self.extractor(object_features, training=training)
        frcnn_predictions = self.predict_class(features)
        frcnn_rois_refinements = self.predict_roi(features)
        frcnn_rois = frcnn_rois_refinements + rpn_rois

        # calculate visual representation for each object if necessary
        visual_representations = None
        if return_visual_representations:
            context_features = self.gap(context_features)
            context_features = tf.gather(context_features, image_assignments)
            visual_representations = tf.concat([context_features, object_features], -1)

        rpn_output = (rpn_predictions, rpn_rois)
        frcnn_output = (frcnn_predictions, frcnn_rois)
        anchors = anchors
        gt_outputs = gt_outputs

        return {
            'rpn_outputs': rpn_output,
            'frcnn_outputs': frcnn_output,
            'anchors': anchors,
            'gt_outputs': gt_outputs,
            'rpn_sampler': rpn_sampler,
            'visual_representations': visual_representations,
            'image_assignments': image_assignments
        }

    @staticmethod
    def std_spec(num_classes, fine_tune=False):
        return {
            'num_classes': num_classes,
            'frcnn_features': 512,
            'rpn_features': 512,
            'anchor_num_scales': 3,
            'total_anchor_overlap_rate': 0.9,
            'non_max_suppression_iou_threshold': 0.7,
            'roi_align_output_size': (7, 7),
            'roi_align_samples': 2,
            'detection_upper_threshold': 0.7,
            'detection_lower_threshold': 0.3,
            'fine_tune_features_extraction': fine_tune
        }

    @staticmethod
    def clevr_spec(fine_tune=False):
        return {
            'num_classes': 96,
            'frcnn_features': 512,
            'rpn_features': 512,
            'anchor_num_scales': 3,
            'total_anchor_overlap_rate': 0.9,
            'non_max_suppression_iou_threshold': 0.7,
            'roi_align_output_size': (7, 7),
            'roi_align_samples': 2,
            'detection_upper_threshold': 0.7,
            'detection_lower_threshold': 0.3,
            'fine_tune_features_extraction': fine_tune
        }

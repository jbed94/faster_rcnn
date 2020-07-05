import tensorflow as tf

from .utils import center_point_to_coordnates


class ROIAlign(tf.keras.layers.Layer):
    """
    ROI Align is a method of extracting features (or simply image) according to the bounding box.
    Bounding boxes are normalized to 1.0
    1. we need to define what is an intended output size (f.e. 7x7 for each roi,
    2. construct "bins" (e.g. 7x7) from rois,
    3. calculate width and height of the bin,
    4. decide how many "samples" do we want to have in each bin (f.e. 2x2, 4x4, etc) - it's regular grid inside bin
    5. calculate most top-left and right-bottom samples' coordinates
    6. run bilinear sampling with intended output size equal to summary number of samples in height and width
    7. for each bin run max pooling (or any other pooling method), so in fact kernel = samples per bin, stride = samples
    """

    def __init__(self, output_size, samples_rate, **kwargs):
        super().__init__(False, 'ROIAlign', tf.float32, **kwargs)
        self.output_size = output_size
        self.samples_rate = samples_rate
        self.pooling = tf.keras.layers.AveragePooling2D(
            (self.samples_rate, self.samples_rate), (self.samples_rate, self.samples_rate))

    # noinspection PyMethodOverriding
    def call(self, inputs, boxes, box_indices, **kwargs):
        boxes = center_point_to_coordnates(boxes)

        # calculate bin width and height
        y1, x1, y2, x2 = tf.split(boxes, 4, axis=1)
        bin_height = (y2 - y1) / self.output_size[0]
        bin_width = (x2 - x1) / self.output_size[1]

        # calculate most left-top coordinate
        grid_center_y1 = (y1 + 0.5 * bin_height / self.samples_rate)
        grid_center_x1 = (x1 + 0.5 * bin_width / self.samples_rate)

        # calculate most bottom right coordinate
        grid_center_y2 = (y2 - 0.5 * bin_height / self.samples_rate)
        grid_center_x2 = (x2 - 0.5 * bin_width / self.samples_rate)

        # construct new bounding box prepared for bilinear sampling
        new_boxes = tf.concat([grid_center_y1, grid_center_x1, grid_center_y2, grid_center_x2], axis=1)

        # type summary number of samples in height and width
        crop_size = tf.constant([self.output_size[0] * self.samples_rate, self.output_size[1] * self.samples_rate])

        # box_indices should be range (0, batch_size)
        sampled = tf.image.crop_and_resize(inputs, new_boxes, box_indices=box_indices, crop_size=crop_size,
                                           method='bilinear')

        # run pooling (may be 'valid' because kernel is adjusted to size
        roi_align = self.pooling(sampled)
        return roi_align

import tensorflow as tf

from ..utils.boxes import center_point_to_coordinates


class ROIAlign(tf.keras.layers.Layer):

    def __init__(self, output_size, samples_rate):
        super().__init__()
        self.output_size = output_size
        self.samples_rate = samples_rate
        self.pooling = tf.keras.layers.AveragePooling2D((self.samples_rate, self.samples_rate), (self.samples_rate, self.samples_rate))

    def call(self, inputs, boxes, assignments):
        input_size = tf.cast(tf.shape(inputs), tf.float32)
        boxes = center_point_to_coordinates(boxes)

        y1, x1, y2, x2 = tf.split(boxes, 4, axis=1)
        bin_height = (y2 - y1) / self.output_size[0]
        bin_width = (x2 - x1) / self.output_size[1]

        grid_center_y1 = (y1 + 0.5 * bin_height / self.samples_rate)
        grid_center_x1 = (x1 + 0.5 * bin_width / self.samples_rate)
        grid_center_y2 = (y2 - 0.5 * bin_height / self.samples_rate)
        grid_center_x2 = (x2 - 0.5 * bin_width / self.samples_rate)

        new_boxes = tf.concat([grid_center_y1 / input_size[1], grid_center_x1 / input_size[2], grid_center_y2 / input_size[1], grid_center_x2 / input_size[2]], axis=1)
        crop_size = tf.constant([self.output_size[0] * self.samples_rate, self.output_size[1] * self.samples_rate])

        sampled = tf.image.crop_and_resize(inputs, new_boxes, box_indices=tf.cast(assignments, tf.int32), crop_size=crop_size, method='bilinear')

        return self.pooling(sampled)

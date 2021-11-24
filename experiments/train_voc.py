import tensorflow as tf
import tensorflow_datasets as tfds

import src

num_epochs = 35
batch_size = 32
num_classes = 20
target_size = (256, 256, 3)
anchor_base = 32
num_samples = 32
momentum = 0.8
dropout = 0.5
learning_rate = 1e-4

(voc_train, voc_val), voc_info = tfds.load(
    'voc/2007',
    split=['train', 'validation'],
    with_info=True
)

voc_train = voc_train \
    .shuffle(256) \
    .map(src.data.voc.as_object_detection(target_size[:2], True), num_parallel_calls=32) \
    .padded_batch(batch_size) \
    .prefetch(8)

voc_val = voc_val \
    .map(src.data.voc.as_object_detection(target_size[:2], True), num_parallel_calls=32) \
    .padded_batch(batch_size) \
    .prefetch(8)

frcnn = src.models.FasterRCNN(src.data.voc.NUM_CLASSES, anchor_base=anchor_base, momentum=momentum, dropout=dropout)
optimizer = tf.optimizers.Adam(learning_rate)

query, train = src.models.faster_rcnn.prepare_query_and_train(frcnn, optimizer, num_samples)
epoch_rpn_accuracy, epoch_frcnn_accuracy = tf.metrics.Mean(), tf.metrics.Mean()


def reset_state():
    epoch_rpn_accuracy.reset_state()
    epoch_frcnn_accuracy.reset_state()


def update_state(rpn_accuracy, frcnn_accuracy):
    epoch_rpn_accuracy.update_state(rpn_accuracy)
    epoch_frcnn_accuracy.update_state(frcnn_accuracy)


for epoch in range(num_epochs):
    reset_state()
    for data in voc_train:
        outputs, (loss, rpn_accuracy, frcnn_accuracy) = train(data['image'], data['boxes'], data['num_objects'], data['labels'])
        update_state(rpn_accuracy, frcnn_accuracy)

    print(f'Train epoch: {epoch} | RPN Accuracy: {epoch_rpn_accuracy.result()} | FRCNN Accuracy: {epoch_frcnn_accuracy.result()}')

    reset_state()
    for data in voc_val:
        outputs, (loss, rpn_accuracy, frcnn_accuracy) = query(data['image'], data['boxes'], data['num_objects'], data['labels'])
        update_state(rpn_accuracy, frcnn_accuracy)

    print(f'Validation epoch: {epoch} | RPN Accuracy: {epoch_rpn_accuracy.result()} | FRCNN Accuracy: {epoch_frcnn_accuracy.result()}')

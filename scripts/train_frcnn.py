import inspect
import os
import sys

from tqdm import tqdm

# add parent (root) to pythonpath
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from argparse import ArgumentParser

import tensorflow as tf
import tensorflow.contrib as tfc
from scripts import ExperimentHandler

from datasets.clevr import clevr, PADDED_SHAPES
from src.faster_rcnn import FasterRCNN
from src.losses import rpn_loss, faster_rcnn_loss
from src.utils import center_point_to_coordnates

tf.enable_eager_execution()

_tqdm = lambda t, s, i: tqdm(
    ncols=80,
    total=s,
    bar_format='%s epoch %d | {l_bar}{bar} | Remaining: {remaining}' % (t, i))


def _ds(title, ds, ds_size, i, batch_size):
    with _tqdm(title, ds_size, i) as pbar:
        for i, data in enumerate(ds):
            yield (i,) + (data,)
            pbar.update(batch_size)


def image_bbox(image, bbox, norm, assignments):
    first_indices = tf.squeeze(tf.where(tf.equal(assignments, 0)), 1)
    bbox = tf.gather(bbox, first_indices)
    bbox = center_point_to_coordnates(bbox[tf.newaxis] / norm)

    return tf.image.draw_bounding_boxes(image[:1], bbox)


def main(args):
    # 1. Get datasets
    train_ds = clevr(os.path.join(args.dataset_path, 'train', 'scenes_objects.tfrecord'))
    val_ds = clevr(os.path.join(args.dataset_path, 'val', 'scenes_objects.tfrecord'))

    train_ds = train_ds \
        .shuffle(50) \
        .padded_batch(args.batch_size, padded_shapes=PADDED_SHAPES)
    val_ds = val_ds \
        .shuffle(50) \
        .padded_batch(args.batch_size, padded_shapes=PADDED_SHAPES)

    # 2. Define model
    model = FasterRCNN(**FasterRCNN.std_spec(96, args.fine_tune_cnn))

    # 3. Optimization
    eta = tfc.eager.Variable(args.eta)
    eta_f = tf.train.exponential_decay(
        args.eta,
        tf.train.get_or_create_global_step(),
        args.eta_decay_steps,
        args.train_beta)
    eta.assign(eta_f())
    optimizer = tf.train.AdamOptimizer(eta)
    l2_reg = tf.keras.regularizers.l2(1e-4)

    # 4. Restore, Log & Save
    experiment_handler = ExperimentHandler(args.working_path, args.out_name, args.log_interval, model, optimizer)

    # 5. Run everything
    train_step, val_step = 0, 0
    for epoch in range(args.num_epochs):

        # 5.1. Training Loop
        experiment_handler.log_training()
        for i, image_spec in _ds('Train', train_ds, args.train_size, epoch, args.batch_size):
            image = image_spec['image/encoded']
            gt_object_bbox = image_spec['image/object/bbox/roi']
            gt_object_label = image_spec['image/object/class/label']
            gt_num_objects = image_spec['image/num_objects']
            h, w = tf.shape(image).numpy()[1:3]
            norm = tf.convert_to_tensor([[[h, w, h, w]]], tf.float32)

            # 5.1.1. Make inference of the model, calculate losses and record gradients
            with tf.GradientTape() as tape:
                model_outputs = model(
                    image, True,
                    inference=False,
                    gt_object_bbox=gt_object_bbox,
                    gt_object_label=gt_object_label,
                    gt_num_objects=gt_num_objects,
                    return_visual_representations=False)
                rpn_samples = model_outputs['rpn_sampler'](args.rpn_positive_samples, args.rpn_negative_samples)

                loss_rpn = rpn_loss(**rpn_samples)
                loss_faster_rcnn = faster_rcnn_loss(**model_outputs)
                loss_reg = tfc.layers.apply_regularization(l2_reg, model.trainable_variables)
                total_loss = loss_rpn + loss_faster_rcnn + loss_reg

            # 5.1.2 Take gradients (if necessary apply regularization like clipping),
            grads = tape.gradient(total_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables),
                                      global_step=tf.train.get_or_create_global_step())

            # 5.1.3 get outputs for stats
            obj_detection_logits, obj_rpn_rois = model_outputs['rpn_outputs']
            obj_label_logits, obj_frcnn_rois = model_outputs['frcnn_outputs']
            gt_bbox, gt_label, gt_detection = model_outputs['gt_outputs']
            ia = model_outputs['image_assignments']
            s_obj_detection_logits, s_obj_rpn_rois = rpn_samples['rpn_outputs']
            s_gt_bbox, s_gt_detection = rpn_samples['gt_outputs']

            # 5.1.4 Calculate statistics
            obj_detection = tf.cast(obj_detection_logits > 0.5, tf.float32)
            obj_label = tf.argmax(obj_label_logits, -1)
            s_obj_detection = tf.cast(s_obj_detection_logits > 0.5, tf.float32)

            detection_accuracy = tf.reduce_mean(tf.cast(tf.equal(obj_detection, gt_detection), tf.float32))
            s_detection_accuracy = tf.reduce_mean(tf.cast(tf.equal(s_obj_detection, s_gt_detection), tf.float32))
            label_accuracy = tf.reduce_mean(tf.cast(tf.equal(obj_label, gt_label), tf.float32))

            # 5.1.5 Save logs for particular interval
            with tfc.summary.record_summaries_every_n_global_steps(args.log_interval, train_step):
                # losses
                tfc.summary.scalar('metrics/rpn_loss', loss_rpn, step=train_step)
                tfc.summary.scalar('metrics/faster_rcnn_loss', loss_faster_rcnn, step=train_step)
                tfc.summary.scalar('metrics/reg_loss', loss_reg, step=train_step)
                # accuracy
                tfc.summary.scalar('metrics/detection_accuracy', detection_accuracy, step=train_step)
                tfc.summary.scalar('metrics/sampled_detection_accuracy', s_detection_accuracy, step=train_step)
                tfc.summary.scalar('metrics/label_accuracy', label_accuracy, step=train_step)
                # images
                tfc.summary.image('images/rpn_detections', image_bbox(image, obj_rpn_rois, norm, ia), step=train_step)
                tfc.summary.image('images/frcnn_detections', image_bbox(image, obj_frcnn_rois, norm, ia),
                                  step=train_step)
                tfc.summary.image('images/gt', image_bbox(image, gt_bbox, norm, ia), step=train_step)
                # eta
                tfc.summary.scalar('other/eta', eta, step=train_step)

            # 5.1.6 Update meta variables
            eta.assign(eta_f())
            train_step += 1

            # 5.1.7 Save model after some interations
            if i > 0 and i % args.save_interval == 0:
                experiment_handler.save_last()
                experiment_handler.flush()

        # 5.2. Validation Loop
        experiment_handler.log_validation()
        for i, image_spec in _ds('Validation', val_ds, args.val_size, epoch, args.batch_size):
            image = image_spec['image/encoded'][tf.newaxis]
            gt_object_bbox = image_spec['image/object/bbox/roi']
            gt_object_label = image_spec['image/object/class/label']
            h, w = tf.shape(image).numpy()[1:3]
            norm = tf.convert_to_tensor([[[h, w, h, w]]], tf.float32)

            # 5.2.1 Make inference of the model for validation and calculate losses
            model_outputs = model(
                image, False,
                inference=False,
                gt_object_bbox=gt_object_bbox,
                gt_object_label=gt_object_label,
                return_visual_representations=False)
            rpn_samples = model_outputs['rpn_sampler'](args.rpn_positive_samples, args.rpn_negative_samples)

            loss_rpn = rpn_loss(**rpn_samples)
            loss_faster_rcnn = faster_rcnn_loss(**model_outputs)

            # 5.2.3 get outputs for stats
            obj_detection_logits, obj_rpn_rois = model_outputs['rpn_outputs']
            obj_label_logits, obj_frcnn_rois = model_outputs['frcnn_outputs']
            gt_bbox, gt_label, gt_detection = model_outputs['gt_outputs']
            ia = model_outputs['image_assignments']
            s_obj_detection_logits, s_obj_rpn_rois = rpn_samples['rpn_outputs']
            s_gt_bbox, s_gt_detection = rpn_samples['gt_outputs']

            # 5.2.4 Calculate statistics
            obj_detection = tf.cast(obj_detection_logits > 0.5, tf.float32)
            obj_label = tf.argmax(obj_label_logits, -1)
            s_obj_detection = tf.cast(s_obj_detection_logits > 0.5, tf.float32)

            detection_accuracy = tf.reduce_mean(tf.cast(tf.equal(obj_detection, gt_detection), tf.float32))
            s_detection_accuracy = tf.reduce_mean(tf.cast(tf.equal(s_obj_detection, s_gt_detection), tf.float32))
            label_accuracy = tf.reduce_mean(tf.cast(tf.equal(obj_label, gt_label), tf.float32))

            # 5.2.5 Save logs for particular interval
            with tfc.summary.record_summaries_every_n_global_steps(args.log_interval, train_step):
                # losses
                tfc.summary.scalar('metrics/rpn_loss', loss_rpn, step=train_step)
                tfc.summary.scalar('metrics/faster_rcnn_loss', loss_faster_rcnn, step=train_step)
                tfc.summary.scalar('metrics/reg_loss', loss_reg, step=train_step)
                # accuracy
                tfc.summary.scalar('metrics/detection_accuracy', detection_accuracy, step=train_step)
                tfc.summary.scalar('metrics/sampled_detection_accuracy', s_detection_accuracy, step=train_step)
                tfc.summary.scalar('metrics/label_accuracy', label_accuracy, step=train_step)
                # images
                tfc.summary.image('images/rpn_detections', image_bbox(image, obj_rpn_rois, norm, ia), step=train_step)
                tfc.summary.image('images/frcnn_detections', image_bbox(image, obj_frcnn_rois, norm, ia),
                                  step=train_step)
                tfc.summary.image('images/gt', image_bbox(image, gt_bbox, norm, ia), step=train_step)

            # 5.2.5 Update meta variables
            val_step += 1


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset-path', type=str, required=True)
    parser.add_argument('--working-path', type=str, default='./working_dir')
    parser.add_argument('--num-epochs', type=int, required=True)
    parser.add_argument('--save-interval', type=int, default=5000)
    parser.add_argument('--log-interval', type=int, default=5)
    parser.add_argument('--batch-size', type=int, required=True)
    parser.add_argument('--out-name', type=str, required=True)
    parser.add_argument('--eta', type=float, default=5e-4)
    parser.add_argument('--eta-decay-steps', type=int, default=2000)
    parser.add_argument('--train-beta', type=float, default=0.99)
    parser.add_argument('--rpn-positive-samples', type=int, default=256)
    parser.add_argument('--rpn-negative-samples', type=int, default=256)
    parser.add_argument('--train-size', type=int, default=70000)
    parser.add_argument('--val-size', type=int, default=15000)
    parser.add_argument('--fine-tune-cnn', action='store_true', default=False)

    args, _ = parser.parse_known_args()
    main(args)

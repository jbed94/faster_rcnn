import tensorflow as tf


def reduce_bbox_iou(bboxes1, bboxes2):
    """
    Helper function for calculation IoU between each pair from two bbox sets.
    :param bboxes1: [batch_size, b1, 4]
    :param bboxes2: [batch_size, b2, 4]
    :return: iou [batch_size, b1, b2]
    """
    # convert center point to coordinates
    y_top1, x_left1, y_bottom1, x_right1 = center_point_to_coordnates(bboxes1, reduce=False)
    y_top2, x_left2, y_bottom2, x_right2 = center_point_to_coordnates(bboxes2, reduce=False)

    xI1 = tf.maximum(x_left1, tf.transpose(x_left2, [0, 2, 1]))
    xI2 = tf.minimum(x_right1, tf.transpose(x_right2, [0, 2, 1]))

    yI1 = tf.minimum(y_bottom1, tf.transpose(y_bottom2, [0, 2, 1]))
    yI2 = tf.maximum(y_top1, tf.transpose(y_top2, [0, 2, 1]))

    inter_area = tf.maximum((xI2 - xI1), 0) * tf.maximum((yI1 - yI2), 0)

    bboxes1_area = (x_right1 - x_left1) * (y_bottom1 - y_top1)
    bboxes2_area = (x_right2 - x_left2) * (y_bottom2 - y_top2)

    union = (bboxes1_area + tf.transpose(bboxes2_area, [0, 2, 1])) - inter_area

    # some invalid boxes should have iou of 0 instead of NaN
    # If inter_area is 0, then this result will be 0; if inter_area is
    # not 0, then union is not too, therefore adding a epsilon is OK.
    iou = inter_area / (union + 0.0001)
    return iou


def center_point_to_coordnates(bbox, reduce=True):
    """
    Converts bboxes in the form of [center_y, center_x, height, width] into
    form of [y_top, x_left, y_bottom, x_right]
    """
    y, x, h, w = tf.split(bbox, 4, -1)
    h2, w2 = h / 2.0, w / 2.0
    bbox = [y - h2, x - w2, y + h2, x + w2]
    if reduce:
        return tf.concat(bbox, -1)
    return tuple(bbox)


def coordinates_to_center_point(bbox, reduce=True):
    """
    Converts bboxes in the form of [y_top, x_left, y_bottom, x_right] into
    form of [center_y, center_x, height, width]
    """
    y_top, x_left, y_botton, x_right = tf.split(bbox, 4, -1)
    h, w = y_botton - y_top, x_right - x_left
    h2, w2 = h / 2, w / 2

    bbox = [y_top + h2, x_left + w2, h, w]
    if reduce:
        return tf.concat(bbox, -1)
    return tuple(bbox)


def _pad_anchors(anchors, y, c, batch_size):
    """
    Anchors are flattened. Tu run everything in batch we need to have padded tensor.
    E.g. when first example has 10 anchors and second 7, then we want to have a tensor with shape of
    [batch_size, 10, 4] where for second example remaining 3 anchors are padded with zeros
    :param anchors: anchors to pad (flattened)
    :param y: index of example in batch
    :param c: number of anchors for each example in batch
    :param batch_size:
    :return:
    - **active_anchors** - are those, which are real (in above example, for batch idx = 0, {0, 1, ..., 9},
    for batch idx = 1 {0, 1, ..., 6}
    - **padded_anchors** - tensor of shape [batch_size, max_anchors, 4]
    """
    # init variables
    # max_count - maximum size of anchors in batch
    # current_p - helper cumulative variable
    # active_anchors - "active" means not padded, inactive - padded (used to gather at the end of processing)
    # real_anchors_pointer - the same as active anchors but exact pointers in tensor (used to build padded tensor)
    max_count, current_p, active_anchors, real_anchors_pointer = max(c), 0, [], []

    # for each pair ("example_idx", "count of anchors")
    for y_i, c_i in zip(y, c):
        real_anchors_pointer.append(tf.stack([tf.tile([y_i], [c_i]), tf.range(0, c_i)], 1))
        active_anchors.append(tf.range(current_p, current_p + c_i))
        current_p += max_count

    # concat results from loop - run everything in one operation
    real_anchors_pointer = tf.concat(real_anchors_pointer, 0)
    active_anchors = tf.concat(active_anchors, 0)

    # create tensor filled with real values using scatter_nd
    shape = [batch_size, max_count, 4]
    padded_anchors = tf.scatter_nd(real_anchors_pointer, anchors, shape)

    return active_anchors, padded_anchors


def _get_object_presence(gt_num_objects):
    """
    Since we are working on batch and each image have different number of objects we need to distinguish which
    object is really on image and which is padded one.
    :return indicator tensor, which says whether object is real (1) or padded (0)
    """
    max_num = tf.reduce_max(gt_num_objects)
    obj_presence = tf.map_fn(lambda x: tf.cast(tf.range(max_num) < x, tf.int32), gt_num_objects)
    return tf.cast(tf.reshape(obj_presence, [-1]), tf.float32)


def get_gt_data(anchors, gt_object_bbox, gt_label, gt_num_objects, score_threshold=0.7):
    """
    Helper function which assigns ground truth data to each detected anchor.
    :param anchors: [None, 4] - detected anchors at the image,
    :param gt_object_bbox: [None, max_gt, 4] - ground truth bounding boxes of the objects,
    :param gt_label: [None, max_gt] - ground truth labels of the objects,
    :param gt_num_objects: number of objects (ground truth) in each example in batch,
    :param score_threshold: then to say that anchor (according to the IoU) contains object or not.
    :return: ground truth data (bbox, object label, detection, scores) for each input anchor
    """
    # get some constant values like batch size, number of anchors for each input image
    batch_size = tf.shape(gt_object_bbox)[0]
    anchors = tf.reshape(anchors, [batch_size, -1, 4])

    # we assume, that each image in batch may have different number of anchors (f.e. if different anchors filtering
    # will be executed, at the beginning), so we need to calculate all sparse indices to "active" detected objects
    # and ground truth data (we pad both GROUND TRUTH and DETECTED ANCHORS)
    object_presence = _get_object_presence(gt_num_objects)

    # calculate iou between all anchors and gt (also between padded ones !!!)
    iou = reduce_bbox_iou(anchors, gt_object_bbox)

    # Find best anchor for each ground truth (closest in IoU score)
    # There might be situation where closes anchor has IoU < threshold, but we need to decide which is the best
    best_anchors = tf.argmax(iou, 1, output_type=tf.int32)
    batch_idx = tf.tile(tf.range(0, batch_size)[:, tf.newaxis], [1, tf.shape(best_anchors)[1]])
    best_anchors = tf.stack([tf.reshape(batch_idx, [-1]), tf.reshape(best_anchors, [-1])], 1)

    # Assign scores to found ones (here scores are not IoU yet)
    best_scores = tf.scatter_nd(best_anchors, object_presence, [batch_size, tf.shape(anchors)[1]])
    best_scores = tf.minimum(best_scores, 1.0)

    # WARNING: for opposite situation for the best anchor for particular ground truth does not have to occur
    # best gt for each anchor can be different
    fix_indices = tf.tile(tf.range(0, tf.shape(gt_object_bbox)[1])[:, tf.newaxis], [tf.shape(gt_object_bbox)[0], 1])
    fix_indices = tf.concat([best_anchors, fix_indices], -1)
    iou_fix = tf.scatter_nd(fix_indices, object_presence, tf.shape(iou))
    iou_fix = tf.minimum(iou_fix, 1.0)

    # now highest iou (hopefully, because there always can be more "1"s, but then don't care),
    # will be for anchor's best fit
    iou = tf.maximum(iou, iou_fix)

    # Find best ground truth for each anchor
    scores = tf.reduce_max(iou, 2)
    gt_idx = tf.argmax(iou, 2, output_type=tf.int32)
    batch_idx = tf.tile(tf.range(0, batch_size)[:, tf.newaxis], [1, tf.shape(gt_idx)[1]])
    gt_idx = tf.stack([tf.reshape(batch_idx, [-1]), tf.reshape(gt_idx, [-1])], 1)

    # gather closest gt bbox and gt label for each detected anchor
    gt_object_bbox = tf.gather_nd(gt_object_bbox, gt_idx)
    gt_label = tf.gather_nd(gt_label, gt_idx)

    # Merge scores (score != best_score)
    scores = tf.reshape(tf.maximum(scores, best_scores), [-1])

    return gt_object_bbox, gt_label, scores


def rpn_sample(results, gt, upper_threshold=0.7, lower_threshold=0.3, num_positive=256, num_negative=256):
    """
    Prepares a sampler which returns given number of positive and negative (randomly sampled with returning) examples
    """
    scores = gt[-1]

    # as positive we take only those with score >= 0.7, for negative <= 0.3 and the rest is ignored
    positive = tf.squeeze(tf.where(scores >= upper_threshold), 1)
    negative = tf.squeeze(tf.where(scores <= lower_threshold), 1)

    # from positive indices sample (with returning) num_positive samples
    # to the same for negative
    positive_idx = tf.random.uniform([num_positive], 0, tf.shape(positive)[0], tf.int32)
    negative_idx = tf.random.uniform([num_negative], 0, tf.shape(negative)[0], tf.int32)
    positive_idx = tf.gather(positive, positive_idx)
    negative_idx = tf.gather(negative, negative_idx)

    # sample both positive and negative
    positive_batch_results = sample_many(positive_idx, *results)
    positive_batch_gt = sample_many(positive_idx, *gt)

    negative_batch_results = sample_many(negative_idx, *results)
    negative_batch_gt = sample_many(negative_idx, *gt)

    return positive_batch_results, positive_batch_gt, negative_batch_results, negative_batch_gt


def sample_many(indices, *tensors):
    """
    Simple helper for gathering the same indices from a given list of tensors
    """
    return tuple([tf.gather(tensor, indices) for tensor in tensors])


def allow_memory_growth():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

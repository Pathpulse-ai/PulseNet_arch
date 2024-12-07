"""
Assign closest ground-truth bounding boxes to anchor boxes.
One or more than one overlapping anchor boxes are assigned to each bounding boxes depending 
upon the given iou_threshhold 
"""

def assign_anchor_to_bbox(ground_truth, anchors, iou_threshold=0.4):
    
    num_anchors, num_gt_boxes = anchors.shape[0], ground_truth.shape[0]
    # Element x_ij in the i-th row and j-th column is the IoU of the anchor
    # box i and the ground-truth bounding box j
    jaccard = box_iou(anchors, ground_truth)
    # print('jaccard: ', jaccard)
    # Initialize the tensor to hold the assigned ground-truth bounding box for
    # each anchor
    anchors_bbox_map = tf.fill((num_anchors,), -1)
    # print('anchors_bbox_map: ', anchors_bbox_map)
    # Assign ground-truth bounding boxes according to the threshold
    # print(tf.math.reduce_max(jaccard, axis=1))
    
    max_ious = tf.math.reduce_max(jaccard, axis=1)
    indices = tf.math.argmax(jaccard, axis=1)
    # print('max_ious, indices: ', max_ious, indices)
    anc_i = tf.where(max_ious >= iou_threshold).reshape(-1)
    # print('anc_i: ', anc_i)
    box_j = indices[max_ious >= iou_threshold]
    # print('box_j: ', box_j)
    anchors_bbox_map_np = np.array(anchors_bbox_map)
    anchors_bbox_map_np[np.array(anc_i[:])] = box_j
    # print('anchors_bbox_map: ', anchors_bbox_map_np)
    col_discard = tf.fill((num_anchors,), -1)
    row_discard = tf.fill((num_gt_boxes,), -1)
    # print('row_discard, col_discard: ', row_discard, col_discard)
    jaccard = np.array(jaccard)
    for i in range(num_gt_boxes):
        max_idx = tf.math.argmax(jaccard[:, i])  # Find the largest IoU
        # print('max_idx: ', max_idx)
        box_idx = i
        anc_idx = max_idx
        # print('box_idx, anc_idx: ', box_idx, anc_idx)
        anchors_bbox_map_np[anc_idx] = box_idx
        jaccard[:, box_idx] = col_discard
        jaccard[anc_idx, :] = row_discard
    # print('jaccard: ', jaccard)
    return tf.convert_to_tensor(anchors_bbox_map_np)
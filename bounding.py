"""
Converts bounding box from (xmin, ymin, xmax, ymax) format
to (cx, cy, width, height) format
"""

def box_corner_to_center(cords):
    try:
        ret = cords.numpy()
    except:
        ret = np.copy(cords)
    ret[:, 0] = cords[:, 0] + (cords[:, 2] - cords[:, 0]) / 2
    ret[:, 1] = cords[:, 1] + (cords[:, 3] - cords[:, 1]) / 2
    ret[:, 2] = cords[:, 2] - cords[:, 0]
    ret[:, 3] = cords[:, 3] - cords[:, 1]
    return tf.convert_to_tensor(ret)

    """
Converts bounding box from (cx, cy, width, height) format
to (xmin, ymin, xmax, ymax) format
"""

def box_center_to_corner(cords):
    try:
        ret = cords.numpy()
    except:
        ret = np.copy(cords)
    ret[:, 0] = cords[:, 0] - cords[:, 2] / 2
    ret[:, 1] = cords[:, 1] - cords[:, 3] / 2
    ret[:, 2] = cords[:, 2] + ret[:, 0]
    ret[:, 3] = cords[:, 3] + ret[:, 1]
    return tf.convert_to_tensor(ret)

    """
Calculates offsets of actual bounding boxes from predefined anchors
"""

def offset_boxes(anchors, assigned_bb, eps=1e-6):
    # print("offset_boxes", anchors.shape, assigned_bb.shape)
    # print(assigned_bb, assigned_bb)
    """Transform for anchor box offsets."""
    # change anchors to yolo form (cx, cy, width, height)
    c_anc = box_corner_to_center(anchors)
    # print('c_anc: ', c_anc)
    # change bbox to yolo form (cx, cy, width, height)
    c_assigned_bb = box_corner_to_center(assigned_bb)
    # print('c_assigned_bb: ', c_assigned_bb)
    offset_xy = 10 * (c_assigned_bb[:, :2] - c_anc[:, :2]) / c_anc[:, 2:]
    offset_wh = 5 * tf.math.log(eps + c_assigned_bb[:, 2:] / c_anc[:, 2:])
    offset = tf.concat([offset_xy, offset_wh], axis=1)
    return offset

    """
One of the most important function/step for Object detection

Input: 
1. pre generated anchors boxes
2. the class labels from training input
3. the bounding box labels from training input

The following things takes place in this function:
---------------------------------------------------

for each batch of input,
1.  a anchors_bbox_map is created of size 2140 which tells that out of 2140 anchor boxes
    which are those anchor boxes active or overlaps with input labels

2.  bbox_mask of shape 8560 is created which copies over the active anchors_box_map 
    4 times to match with output bounding boxes of shape (batch_size, 8560)
    It has values 0 or 1 which signifies active or inactive

3.  new class_labels and assigned_bb (bounding box) are created initialized as 
    empty arrays of shape(2140, 16) and (2140, 4) respectively

4.  these new labels and bb are fed with values from input labels with respect to
    anchors indexes (print and see yourself for in depth clarification)

5.  offsets are created for the new bb with respect to prdefined anchors. During training, inference
    those are offsets what are predicted by the model, they go through offset_inverse
    for plotting and iou calculations 


At the end after completeng all steps for every batch the new bbox_offset, bbox_masks,
class_labels ( initialised as one hot encoded class for active anchors and one hot encoded background 
for background anchors ), all these values are returned to caller function (loss function for
loss calculation)

"""

def multibox_target(anchors, cls_labels, box_labels):
    """Label anchor boxes using ground-truth bounding boxes."""
    batch_size, anchors, n_classes = len(cls_labels), anchors[0], cls_labels.shape[-1]
    batch_offset, batch_mask, batch_class_labels = [], [], []
    num_anchors = anchors.shape[0]
    for i in range(batch_size):
        label = box_labels[i]
        c_label = cls_labels[i]
        # print('c_label: ', c_label)
        anchors_bbox_map = assign_anchor_to_bbox(label, anchors)
        # print('anchors_bbox_map: ', anchors_bbox_map)
        bbox_mask = (anchors_bbox_map >= 0).numpy().repeat(4, 0).reshape(-1, 4).astype('float16')
        # print('bbox_mask: ', bbox_mask)

        # Initialize class labels and assigned bounding box coordinates with
        # zeros
        class_labels = np.zeros((num_anchors, 16))

        # make sure to mark the last column as 1 by default so that it signifies 
        #background class
        class_labels[:, 15] = np.ones(num_anchors)
        # print(class_labels[:, 15])

        assigned_bb = np.zeros((num_anchors, 4))
        # print('class_labels, assigned_bb: ', class_labels, assigned_bb)

        # Label classes of anchor boxes using their assigned ground-truth
        # bounding boxes. If an anchor box is not assigned any, we label its
        # class as background (the value remains one)
        indices_true = tf.where(anchors_bbox_map >= 0).numpy()
        # print('indices_true: ', indices_true)
        bb_idx = np.array(anchors_bbox_map[indices_true])
        # print('bb_idx: ', bb_idx)
 
        # print(bb_idx.shape, indices_true.shape)
        # print(class_labels.shape, cls_labels[bb_idx[:, 0]].shape)
        
        class_labels[indices_true] = c_label[bb_idx] #np.argmax(class_labels[i][bb_idx]) + 1 #label[bb_idx, 0] + 1 
        class_labels[indices_true, 15] = 0
        # assigned_bb = assigned_bb.numpy()
       
        assigned_bb[indices_true] = label[bb_idx]
        
        # print('class_labels, assigned_bb after: ', class_labels, assigned_bb)
        # Offset transformation
        offset = offset_boxes(anchors, assigned_bb) * bbox_mask
        # print('offset: ', offset)
        batch_offset.append(offset.reshape(-1))
        batch_mask.append(bbox_mask.reshape(-1))
        batch_class_labels.append(class_labels)
    bbox_offset = tf.stack(batch_offset)
    bbox_mask = tf.stack(batch_mask)
    class_labels = tf.stack(batch_class_labels)
    return (bbox_offset, bbox_mask, class_labels)

    """
Converts offsets back to actual bounding boxes of form (xmin, ymin, xmax, ymax)
with respect to predefined anchors
"""

def offset_inverse(anchors, offset_preds):
    anc = box_corner_to_center(anchors)
    pred_bbox_xy = (offset_preds[:, :2] * anc[:, 2:] / 10) + anc[:, :2]
    pred_bbox_wh = tf.math.exp(offset_preds[:, 2:] / 5) * anc[:, 2:]
    pred_bbox = tf.concat((pred_bbox_xy, pred_bbox_wh), axis=1)
    predicted_bbox = box_center_to_corner(pred_bbox)
    return predicted_bbox

    """
Major function

Creates anchors boxes for our model.
For our model we need anchors boxes of size
"""

def create_anchors():
    
    sizes = [[0.2, 0.272], [0.37, 0.447], [0.54, 0.619], [0.71, 0.79], [0.88, 0.961]]
    input_shapes = [[20, 20, 3], [10, 10, 3], [5, 5, 3], [3, 3, 3], [1, 1, 3]]
    ratios = [0.5, 1, 2]
    output_anchors = multibox_prior(np.zeros(input_shapes[0]), sizes=sizes[0], ratios=ratios)
    
    for i in range(1,5):
        anchors_t = multibox_prior(np.zeros(input_shapes[i]), sizes=[0.75, 0.5], ratios=[1, 2, 0.5])
        output_anchors = tf.concat([output_anchors, anchors_t], axis=1)
        
    return output_anchors
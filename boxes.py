"""
Computes IoU of one list of boxes over other
box format: np.array([[xmin, ymin, xmax, ymax], ...])
returns ious of size [len(boxes1), len(boxes2)]
"""
def box_iou(boxes1, boxes2):
    """Compute pairwise IoU across two lists of anchor or bounding boxes."""
    box_area = lambda boxes: ((boxes[:, 2] - boxes[:, 0]) *
                              (boxes[:, 3] - boxes[:, 1]))
    # Shape of `boxes1`, `boxes2`, `areas1`, `areas2`: (no. of boxes1, 4),
    # (no. of boxes2, 4), (no. of boxes1,), (no. of boxes2,)
    areas1 = box_area(boxes1)
    areas2 = box_area(boxes2)
    # print('curr area: ', areas1, '\n')
    # Shape of `inter_upperlefts`, `inter_lowerrights`, `inters`: (no. of
    # boxes1, no. of boxes2, 2)
    inter_upperlefts = tf.math.maximum(boxes1[:, None, :2], boxes2[:, :2])
    inter_lowerrights = tf.math.minimum(boxes1[:, None, 2:], boxes2[:, 2:])
    inters = (inter_lowerrights - inter_upperlefts).numpy().clip(min=0)
    # print(inters)
    # Shape of `inter_areas` and `union_areas`: (no. of boxes1, no. of boxes2)
    inter_areas = inters[:, :, 0] * inters[:, :, 1]
    union_areas = areas1[:, None] + areas2 - inter_areas
    return inter_areas / union_areas

    """
Compute if a new object in an Image is overlapping with alraedy existing object's boxes.
"""

def overlap(boxes1, boxes2):
    box_area = lambda boxes: ((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]))

    areas1 = box_area(boxes1)
    areas2 = box_area(boxes2)

    inter_upperlefts = tf.math.maximum(boxes1[:, None, :2], boxes2[:, :2])
    inter_lowerrights = tf.math.minimum(boxes1[:, None, 2:], boxes2[:, 2:])
    inters = (inter_lowerrights - inter_upperlefts).numpy().clip(min=0)

    inter_areas = inters[:, :, 0] * inters[:, :, 1]
    union_areas = areas1[:, None] + areas2 - inter_areas
 
    if np.where(((inter_areas / union_areas) * areas1)[0] > 0.01)[0].shape[0] > 0:
        return True
   

    return False
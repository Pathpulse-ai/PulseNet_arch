"""
Generates anchor boxes for a given image size, size of anchors
and ratios of anchors
"""

def multibox_prior(X, sizes, ratios):
    in_height, in_width = X.shape[-3:-1]
    num_sizes, num_ratios = len(sizes), len(ratios)
    boxes_per_pixel = (num_sizes + num_ratios - 1)
    size_tensor = tf.convert_to_tensor(sizes)
    ratio_tensor = tf.convert_to_tensor(ratios)
    # Offsets are required to move the anchor to the center of a pixel. Since
    # a pixel has height=1 and width=1, we choose to offset our centers by 0.5
    offset_h, offset_w = 0.5, 0.5
    steps_h = 1.0 / in_height  # Scaled steps in y axis
    steps_w = 1.0 / in_width  # Scaled steps in x axis
    # print("steps_h, steps_w", steps_h, steps_w)

    # Generate all center points for the anchor boxes
    center_h = (tf.range(in_height, dtype=tf.float32) + offset_h) * steps_h
    center_w = (tf.range(in_width, dtype=tf.float32) + offset_w) * steps_w
    shift_y, shift_x = tf.meshgrid(center_h, center_w, indexing='ij')
    shift_y, shift_x = shift_y.reshape(-1), shift_x.reshape(-1)
    # print("center_h, center_w", center_h, center_w)
    # print("shift_y, shift_x", shift_y, shift_x)

    # Generate `boxes_per_pixel` number of heights and widths that are later
    # used to create anchor box corner coordinates (xmin, xmax, ymin, ymax)
    w = tf.concat((size_tensor * tf.sqrt(ratio_tensor[0]), sizes[0] * tf.sqrt(ratio_tensor[1:])), axis=0) * in_height / in_width  # Handle rectangular inputs
    h = tf.concat((size_tensor / tf.sqrt(ratio_tensor[0]), sizes[0] / tf.sqrt(ratio_tensor[1:])), axis=0)
    # print(w, h)
    # Divide by 2 to get half height and half width
    # print(type(tf.stack((-w, -h, w, h)).T.numpy()))
    # return
    anchor_manipulations = np.tile(tf.stack((-w, -h, w, h), axis=0).numpy(), in_height * in_width) / 2
    anchor_manipulations = anchor_manipulations.T
    # print('anchor_manipulations: ', anchor_manipulations.shape)
    # print(anchor_manipulations)
    # Each center point will have `boxes_per_pixel` number of anchor boxes, so
    # generate a grid of all anchor box centers with `boxes_per_pixel` repeats
    # print('outgrid: ', tf.stack([shift_x, shift_y, shift_x, shift_y], axis=1).numpy().repeat(boxes_per_pixel, 0))
    out_grid = tf.stack([shift_x, shift_y, shift_x, shift_y], axis=1).numpy().repeat(boxes_per_pixel, 0)
    output = out_grid + anchor_manipulations
    print('output: ', output.shape)
    # print(output)
    return tf.expand_dims(output, 0)
"""
Model Architecture
-------------------

5 class heads of shapes:

20 x 20 x 64, 
10 x 10 x 64, 
5 x 5 x 64,
3 x 3 x 64,
1 x 1 x 64
-----------
Total : 535 X 64

This 64 output channel of each head will be rehaped to 4 X 16
 [4 for each anchor box and 16 for classes]
and then stacked to get final class output shape of (2140 X 16)


5 box heads of shapes:

20 x 20 x 16, 
10 x 10 x 16, 
5 x 5 x 16,
3 x 3 x 16,
1 x 1 x 16
-----------
Total : 535 X 16

This 16 channels will be rehaped to a 4 channel i.e 4 anchors X 4 boxes anchors being stacked atop 
each other, so the total output shape will be (2140 X 4).



  2140 X 20 where first 4 colums are bboxes and last 16 colums are class probabilities (not in probabilistic form)

"""
input_ = Input(shape=(320, 320, 3), name='image')


conv2d_1 = Conv2D(16, 3, activation='relu', name='conv2d_1', padding='same')(input_)
maxpool2d_1 = MaxPool2D(2, name='maxpool2d_1')(conv2d_1)
batchnorm_1 = BatchNormalization(name='batchnorm_1')(maxpool2d_1)


conv2d_2 = Conv2D(32, 3, activation='relu', name='conv2d_2', padding='same')(batchnorm_1)
maxpool2d_2 = MaxPool2D(2, name='maxpool2d_2')(conv2d_2)
batchnorm_2 = BatchNormalization(name='batchnorm_2')(maxpool2d_2)


conv2d_3 = Conv2D(64, 3, activation='relu', name='conv2d_3', padding='same')(batchnorm_2)
maxpool2d_3 = MaxPool2D(2, name='maxpool2d_3')(conv2d_3)
batchnorm_3 = BatchNormalization(name='batchnorm_3')(maxpool2d_3)


conv2d_4 = Conv2D(128, 3, activation='relu', name='conv2d_4', padding='same')(batchnorm_3)
maxpool2d_4 = MaxPool2D(2, name='maxpool2d_4')(conv2d_4) # 20x20x128
batchnorm_4 = BatchNormalization(name='batchnorm_4')(maxpool2d_4)


conv2d_5 = Conv2D(256, 3, activation='relu', name='conv2d_5', padding='same')(batchnorm_4)
maxpool2d_5 = MaxPool2D(2, name='maxpool2d_5')(conv2d_5) # 10x10x128
batchnorm_5 = BatchNormalization(name='batchnorm_5')(maxpool2d_5)


conv2d_6 = Conv2D(256, 3, activation='relu', name='conv2d_6', padding='same')(batchnorm_5)
maxpool2d_6 = MaxPool2D(2, name='maxpool2d_6')(conv2d_6) # 5x5x128 output
batchnorm_6 = BatchNormalization(name='batchnorm_6')(maxpool2d_6)
  
    
conv2d_7 = Conv2D(256, 3, activation='relu', name='conv2d_7')(batchnorm_6) # 3x3x128 output
conv2d_8 = Conv2D(512, 3, activation='relu', name='conv2d_8')(conv2d_7) # 1x1x256 output
    

class_20x20 = Conv2D(64, 3, name='class_20x20', activation='linear', padding='same')(maxpool2d_4)
class_20x20_reshape = Reshape((-1, 16), name='class_20x20_reshape')(class_20x20)
# class_20x20_reshape_sm = tf.keras.layers.Softmax(name='class_20x20_reshape_softmax')(class_20x20_reshape)
box_20x20 = Conv2D(16, 3, name='box_20x20', padding='same')(maxpool2d_4)
box_20x20_reshape = Reshape((-1, 4), name='box_20x20_reshape')(box_20x20)


class_10x10 = Conv2D(64, 3, name='class_10x10', activation='linear', padding='same')(maxpool2d_5)
class_10x10_reshape = Reshape((-1, 16), name='class_10x10_reshape')(class_10x10)
# class_10x10_reshape_sm = tf.keras.layers.Softmax(name='class_10x10_reshape_softmax')(class_10x10_reshape)
box_10x10 = Conv2D(16, 3, name='box_10x10', padding='same')(maxpool2d_5)
box_10x10_reshape = Reshape((-1, 4), name='box_10x10_reshape')(box_10x10)


class_5x5 = Conv2D(64, 3, name='class_5x5', activation='linear', padding='same')(maxpool2d_6)
class_5x5_reshape = Reshape((-1, 16), name='class_5x5_reshape')(class_5x5)
# class_5x5_reshape_sm = tf.keras.layers.Softmax(name='class_5x5_reshape_softmax')(class_5x5_reshape)
box_5x5 = Conv2D(16, 3, name='box_5x5', padding='same')(maxpool2d_6)
box_5x5_reshape = Reshape((-1, 4), name='box_5x5_reshape')(box_5x5)


class_3x3 = Conv2D(64, 3, name='class_3x3', activation='linear', padding='same')(conv2d_7)
class_3x3_reshape = Reshape((-1, 16), name='class_3x3_reshape')(class_3x3)
# class_3x3_reshape_sm = tf.keras.layers.Softmax(name='class_3x3_reshape_softmax')(class_3x3_reshape)
box_3x3 = Conv2D(16, 3, name='box_3x3', padding='same')(conv2d_7)
box_3x3_reshape = Reshape((-1, 4), name='box_3x3_reshape')(box_3x3)


class_1x1 = Conv2D(64, 3, name='class_1x1', activation='linear', padding='same')(conv2d_8)
class_1x1_reshape = Reshape((-1, 16), name='class_1x1_reshape')(class_1x1)
# class_1x1_reshape_sm = tf.keras.layers.Softmax(name='class_1x1_reshape_softmax')(class_1x1_reshape)
box_1x1 = Conv2D(16, 3, name='box_1x1', padding='same')(conv2d_8)
box_1x1_reshape = Reshape((-1, 4), name='box_1x1_reshape')(box_1x1)


    
  
# class_out = Concatenate(axis=1, name='class_out')([class_20x20_reshape_sm, class_10x10_reshape_sm, class_5x5_reshape_sm, class_3x3_reshape_sm, class_1x1_reshape_sm])

class_out = Concatenate(axis=1, name='class_out')([class_20x20_reshape, class_10x10_reshape, class_5x5_reshape, class_3x3_reshape, class_1x1_reshape])
box_out = Concatenate(axis=1, name='box_out')([box_20x20_reshape, box_10x10_reshape, box_5x5_reshape, box_3x3_reshape, box_1x1_reshape])
    
# box_out_reshape = Reshape((-1, 4), name='box_out_reshape')(box_out)
final_output = Concatenate(axis=2, name='final_output')([box_out, class_out])


# model = tf.keras.models.Model(input_, [class_out, box_out])
model = tf.keras.models.Model(input_, final_output)
model.summary()
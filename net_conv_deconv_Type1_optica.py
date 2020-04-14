from __future__ import division

import six
import math
from keras.models import Model
from keras.layers import (
    Input,
    Activation,
    Merge,
    Dense,
    Permute,
    GRU,
    Reshape,
    Flatten,
    merge,
    Dropout
)
from keras.layers.convolutional import (
    Convolution2D,
    AtrousConvolution2D,
    Deconvolution2D,
    MaxPooling2D,
    AveragePooling2D
)
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras import backend as K
#from keras.utils.visualize_util import plot


def _bn_relu(input):
    """Helper to build a BN -> relu block
    """
    norm = BatchNormalization(axis=CHANNEL_AXIS)(input)
    return Activation("relu")(norm)


def _deconv_bn_relu(**conv_params):
    """Helper to build a conv -> BN -> relu block
    """
    nb_filter = conv_params["nb_filter"]
    nb_row = conv_params["nb_row"]
    nb_col = conv_params["nb_col"]
    output_shape = conv_params["output_shape"]
    subsample = conv_params.setdefault("subsample", (1, 1))
    init = conv_params.setdefault("init", "he_normal")
    border_mode = conv_params.setdefault("border_mode", "valid")
    W_regularizer = conv_params.setdefault("W_regularizer", l2(1.e-4))

    def f(input):
        conv = Deconvolution2D(nb_filter=nb_filter, nb_row=nb_row, nb_col=nb_col, output_shape=output_shape, subsample=subsample,
                             init=init, border_mode=border_mode, W_regularizer=W_regularizer)(input)
        return _bn_relu(conv)

    return f

def _conv_bn_relu(**conv_params):
    """Helper to build a conv -> BN -> relu block
    """
    nb_filter = conv_params["nb_filter"]
    nb_row = conv_params["nb_row"]
    nb_col = conv_params["nb_col"]
    subsample = conv_params.setdefault("subsample", (1, 1))
    init = conv_params.setdefault("init", "he_normal")
    border_mode = conv_params.setdefault("border_mode", "same")
    W_regularizer = conv_params.setdefault("W_regularizer", l2(1.e-4))

    def f(input):
        conv = Convolution2D(nb_filter=nb_filter, nb_row=nb_row, nb_col=nb_col, subsample=subsample,
                             init=init, border_mode=border_mode, W_regularizer=W_regularizer)(input)
        return _bn_relu(conv)

    return f

def _altconv_bn_relu(**conv_params):
    """Helper to build a conv -> BN -> relu block
    """
    nb_filter = conv_params["nb_filter"]
    nb_row = conv_params["nb_row"]
    nb_col = conv_params["nb_col"]
    subsample = conv_params.setdefault("subsample", (1, 1))
    init = conv_params.setdefault("init", "he_normal")
    border_mode = conv_params.setdefault("border_mode", "same")
    W_regularizer = conv_params.setdefault("W_regularizer", l2(1.e-4))
    input_shape = conv_params.setdefault("input_shape",(None,1024,1024,1))
    atrous_rate  = conv_params.setdefault("atrous_rate",(2,2))

    def f(input):
        conv = AtrousConvolution2D(nb_filter=nb_filter, nb_row=nb_row, nb_col=nb_col, input_shape=input_shape, subsample=subsample,
                             init=init, border_mode=border_mode, W_regularizer=W_regularizer,atrous_rate=atrous_rate)(input)
        return _bn_relu(conv)

    return f

def _bn_relu_altconv(**conv_params):
    """Helper to build a conv -> BN -> relu block
    """
    nb_filter = conv_params["nb_filter"]
    nb_row = conv_params["nb_row"]
    nb_col = conv_params["nb_col"]
    subsample = conv_params.setdefault("subsample", (1, 1))
    init = conv_params.setdefault("init", "he_normal")
    border_mode = conv_params.setdefault("border_mode", "same")
    W_regularizer = conv_params.setdefault("W_regularizer", l2(1.e-4))
    input_shape = conv_params.setdefault("input_shape",(None,1024,1024,1))
    atrous_rate  = conv_params.setdefault("atrous_rate",(2,2))

    def f(input):
        activation = _bn_relu(input)
        return AtrousConvolution2D(nb_filter=nb_filter, nb_row=nb_row, nb_col=nb_col, input_shape=input_shape, subsample=subsample,
                                   init=init, border_mode=border_mode, W_regularizer=W_regularizer,atrous_rate=atrous_rate)(activation)

    return f


def _convfirst_bn_relu(**conv_params):
    """Helper to build a conv -> BN -> relu block
    """
    nb_filter = conv_params["nb_filter"]
    nb_row = conv_params["nb_row"]
    nb_col = conv_params["nb_col"]
    subsample = conv_params.setdefault("subsample", (1, 1))
    init = conv_params.setdefault("init", "he_normal")
    border_mode = conv_params.setdefault("border_mode", "same")
    W_regularizer = conv_params.setdefault("W_regularizer", l2(1.e-4))
    input_shape = conv_params.setdefault("input_shape",(None,1024,1024,1))


    def f(input):
        conv = Convolution2D(nb_filter=nb_filter, nb_row=nb_row, nb_col=nb_col, input_shape=input_shape, subsample=subsample,
                             init=init, border_mode=border_mode, W_regularizer=W_regularizer)(input)
        return _bn_relu(conv)

    return f


def _bn_relu_conv(**conv_params):
    """Helper to build a BN -> relu -> conv block.
    This is an improved scheme proposed in http://arxiv.org/pdf/1603.05027v2.pdf
    """
    nb_filter = conv_params["nb_filter"]
    nb_row = conv_params["nb_row"]
    nb_col = conv_params["nb_col"]
    subsample = conv_params.setdefault("subsample", (1,1))
    init = conv_params.setdefault("init", "he_normal")
    border_mode = conv_params.setdefault("border_mode", "same")
    W_regularizer = conv_params.setdefault("W_regularizer", l2(1.e-4))

    def f(input):
        activation = _bn_relu(input)
        return Convolution2D(nb_filter=nb_filter, nb_row=nb_row, nb_col=nb_col, subsample=subsample,
                             init=init, border_mode=border_mode, W_regularizer=W_regularizer)(activation)

    return f

def _bn_relu_deconv(**conv_params):
    """Helper to build a BN -> relu -> conv block.
    This is an improved scheme proposed in http://arxiv.org/pdf/1603.05027v2.pdf
    """
    nb_filter = conv_params["nb_filter"]
    nb_row = conv_params["nb_row"]
    nb_col = conv_params["nb_col"]
    output_shape = conv_params["output_shape"]
    subsample = conv_params.setdefault("subsample", (1,1))
    init = conv_params.setdefault("init", "he_normal")
    border_mode = conv_params.setdefault("border_mode", "same")
    W_regularizer = conv_params.setdefault("W_regularizer", l2(1.e-4))

    def f(input):
        activation = _bn_relu(input)
        return Deconvolution2D(nb_filter=nb_filter, nb_row=nb_row, nb_col=nb_col, output_shape=output_shape, subsample=subsample,
                             init=init, border_mode=border_mode, W_regularizer=W_regularizer)(activation)

    return f


def _shortcut(input, residual, init_subsample, output_shape):
    """Adds a shortcut between input and residual block and merges them with "sum"
    """
    # Expand channels of shortcut to match residual.
    # Stride appropriately to match residual (width, height)
    # Should be int if network architecture is correctly configured.

    stride_width=init_subsample[0]
    stride_height=init_subsample[1]
    shortcut = input
    # 1 X 1 conv if shape is different. Else identity.
    if stride_width > 1 or stride_height > 1:
        shortcut = Deconvolution2D(nb_filter=output_shape[CHANNEL_AXIS],
                                 nb_row=2, nb_col=2, output_shape=output_shape,
                                 subsample=init_subsample,
                                 init="he_normal", border_mode="valid",
                                 W_regularizer=l2(0.0001))(input)

    return merge([shortcut, residual], mode="sum")

def _shortcutconv(input, residual,init_subsample):
    """Adds a shortcut between input and residual block and merges them with "sum"
    """
    # Expand channels of shortcut to match residual.
    # Stride appropriately to match residual (width, height)
    # Should be int if network architecture is correctly configured.


    stride_width = init_subsample[0]
    stride_height = init_subsample[1]
    input_shape = K.int_shape(input)
    residual_shape = K.int_shape(residual)
    equal_channels = input_shape[CHANNEL_AXIS] == residual_shape[CHANNEL_AXIS]

    shortcut = input
    # 1 X 1 conv if shape is different. Else identity.
    if stride_width > 1 or stride_height > 1 or not equal_channels:
        shortcut = Convolution2D(nb_filter=residual_shape[CHANNEL_AXIS],
                                 nb_row=1, nb_col=1,
                                 subsample=init_subsample,
                                 init="he_normal", border_mode="valid",
                                 W_regularizer=l2(0.0001))(input)

    return merge([shortcut, residual], mode="sum")


def _residual_block(block_function, nb_filter, repetitions, output_shape, is_first_layer=False):
    """Builds a residual block with repeating bottleneck blocks.
    """
    def f(input):
        for i in range(repetitions):
            init_subsample = (1, 1)
            if i == 0:
                init_subsample = (2, 2)
            input = block_function(nb_filter=nb_filter,output_shape=output_shape, init_subsample=init_subsample, 
                                   is_first_block_of_first_layer=(is_first_layer and i == 0))(input)
        return input

    return f



def _residual_blockconv(block_function, nb_filter, repetitions, is_first_layer=False):
    """Builds a residual block with repeating bottleneck blocks.
    """
    def f(input):
        for i in range(repetitions):
            init_subsample = (1, 1)
            if i == 0:
                init_subsample = (2, 2)
            input = basic_blockconv(nb_filter=nb_filter, init_subsample=init_subsample,
                                   is_first_block_of_first_layer=(is_first_layer and i == 0))(input)
        return input

    return f

def _residual_blockconvalt(block_function, nb_filter, repetitions, input_shapess=(None, 256, 256,1), atrous_rate=(2,2), is_first_layer=False):
    """Builds a residual block with repeating bottleneck blocks.
    """
    def f(input):
        for i in range(repetitions):
            init_subsample = (1, 1)
            if i == 0:
                init_subsample = (2, 2)
            input = basic_blockconvalt(nb_filter=nb_filter, init_subsample=init_subsample, input_shapes=input_shapess, atrous_rate=atrous_rate,
                                   is_first_block_of_first_layer=(is_first_layer and i == 0))(input)
        return input

    return f


def _residual_blockconv_final(block_function, nb_filter, repetitions, is_first_layer=False):
    """Builds a residual block with repeating bottleneck blocks.
    """
    def f(input):
        for i in range(repetitions):
            init_subsample = (1, 1)
            input = basic_blockconv(nb_filter=nb_filter, init_subsample=init_subsample,
                                   is_first_block_of_first_layer=(is_first_layer and i == 0))(input)
        return input

    return f

def basic_block(nb_filter, output_shape, init_subsample=(1, 1), is_first_block_of_first_layer=False):
    """Basic 3 X 3 convolution blocks for use on resnets with layers <= 34.
    Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf
    """
    def f(input):

        if is_first_block_of_first_layer:
            # don't repeat bn->relu since we just did bn->relu->maxpool
            print('amhere')
            conv1 = Deconvolution2D(nb_filter=nb_filter,
                                 nb_row=2, nb_col=2, output_shape=output_shape, 
                                 subsample=init_subsample,
                                 init="he_normal", border_mode="same",
                                 W_regularizer=l2(0.0001))(input)
            print('doneamhere')
        else:
            if init_subsample==(2,2):
                conv1 = _bn_relu_deconv(nb_filter=nb_filter, nb_row=2, nb_col=2, output_shape=output_shape,
                                  subsample=init_subsample)(input)               
            else:
                conv1 = _bn_relu_conv(nb_filter=nb_filter, nb_row=3, nb_col=3,
                                  subsample=init_subsample)(input)

        residual = _bn_relu_conv(nb_filter=nb_filter, nb_row=3, nb_col=3)(conv1)
        return _shortcut(input, residual, init_subsample, output_shape)

    return f

def basic_blockconv(nb_filter, init_subsample=(1, 1), is_first_block_of_first_layer=False):
    """Basic 3 X 3 convolution blocks for use on resnets with layers <= 34.
    Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf
    """
    def f(input):

        if is_first_block_of_first_layer:
            # don't repeat bn->relu since we just did bn->relu->maxpool
            conv1 = Convolution2D(nb_filter=nb_filter,
                                 nb_row=3, nb_col=3,
                                 subsample=init_subsample,
                                 init="he_normal", border_mode="same",
                                 W_regularizer=l2(0.0001))(input)
        else:
            conv1 = _bn_relu_conv(nb_filter=nb_filter, nb_row=3, nb_col=3,
                                  subsample=init_subsample)(input)

        residual = _bn_relu_conv(nb_filter=nb_filter, nb_row=3, nb_col=3)(conv1)
        return _shortcutconv(input, residual, init_subsample)

    return f

def basic_blockconvalt(nb_filter, init_subsample=(1, 1), input_shapes=(None, 256, 256,1), atrous_rate=(2,2), is_first_block_of_first_layer=False):
    """Basic 3 X 3 convolution blocks for use on resnets with layers <= 34.
    Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf
    """
    def f(input):

        if is_first_block_of_first_layer:
            # don't repeat bn->relu since we just did bn->relu->maxpool
            conv1 = Convolution2D(nb_filter=nb_filter,
                                 nb_row=3, nb_col=3,
                                 subsample=init_subsample,
                                 init="he_normal", border_mode="same",
                                 W_regularizer=l2(0.0001))(input)
        else:
            if init_subsample==(1,1):
                conv1 = _bn_relu_altconv(nb_filter=nb_filter, nb_row=3, nb_col=3, input_shape=input_shapes, atrous_rate=atrous_rate,
                                  subsample=init_subsample)(input)
            else:
                conv1 = _bn_relu_conv(nb_filter=nb_filter, nb_row=3, nb_col=3,
                                  subsample=init_subsample)(input)
                

        residual = _bn_relu_altconv(nb_filter=nb_filter, nb_row=3, nb_col=3, input_shape=input_shapes, atrous_rate=atrous_rate)(conv1)
        return _shortcutconv(input, residual, init_subsample)

    return f




def _handle_dim_ordering():
    global ROW_AXIS
    global COL_AXIS
    global CHANNEL_AXIS
    if K.image_dim_ordering() == 'tf':
        ROW_AXIS = 1
        COL_AXIS = 2
        CHANNEL_AXIS = 3
    else:
        CHANNEL_AXIS = 1
        ROW_AXIS = 2
        COL_AXIS = 3


def _get_block(identifier):
    if isinstance(identifier, six.string_types):
        res = globals().get(identifier)
        if not res:
            raise ValueError('Invalid {}'.format(identifier))
        return res
    return identifier


class ResnetBuilder(object):
    @staticmethod
    def build_plain(input_shapeaux, type2, shape_add=1, unet=True,imginput=True):
        """Builds a custom ResNet like architecture.
        Args:
            input_shape: The input shape in the form (nb_channels, nb_rows, nb_cols)
            num_outputs: The number of outputs at final softmax layer
            block_fn: The block function to use. This is either `basic_block` or `bottleneck`.
                The original paper used basic_block for layers < 50
            repetitions: Number of repetitions of various block units.
                At each block unit, the number of filters are doubled and the input size is halved
        Returns:
            The keras `Model`.
        """
        _handle_dim_ordering()
        dr_rate=0.02
        dr_rate2=0.02
       


        # Load function from str if needed.
        block_fn = _get_block(basic_block)
        if type2 and not imginput:
            aux_input1 = Input(shape=(1,1,1), name='aux_input1')
            permute01=Permute((2,3,1))(aux_input1)
            conv11 = _conv_bn_relu(nb_filter=16, nb_row=1, nb_col=1, subsample=(1, 1))(permute01)

            ifl=True
            for kk in range(int(math.log(shape_add,2))):
                conv11 = _residual_block(block_fn, nb_filter=80-8*kk, repetitions=2, output_shape=(None,2**(kk+1),2**(kk+1),80-8*kk), is_first_layer=ifl)(conv11)
                ifl=False


        nb_filtermat=[16, 24, 32, 48, 64, 96, 128, 192, 256, 384]
        #nb_filtermat=[16, 32, 48, 64, 96, 128, 192, 256, 384, 512]

        aux_input = Input(shape=input_shapeaux, name='aux_input')        
        permute0=Permute((2,3,1))(aux_input)
        
        
        if type2 and imginput:
            aux_input2 = Input(shape=(1,1024,1024), name='aux_input2')
            permute01=Permute((2,3,1))(aux_input2)
            conv11=permute01

            for kk in range(10-int(math.log(shape_add,2))):
                conv11 = _conv_bn_relu(nb_filter=nb_filtermat[kk], nb_row=3, nb_col=3, subsample=(2, 2))(conv11)
                conv11  = Dropout((dr_rate2))(conv11)
               
     

        
        
        #nb_filtermat[9-int(math.log(shape_add,2))]=nb_filtermat[9-int(math.log(shape_add,2))]-(80-8*int(math.log(shape_add,2)))

        if shape_add==1024 and type2 and imginput:
                permute0 = merge([permute0, conv11], mode='concat', concat_axis=CHANNEL_AXIS)
        
        convalt0 = _altconv_bn_relu(nb_filter=nb_filtermat[0], nb_row=3, nb_col=3, input_shape=(None, 1024, 1024,1), subsample=(1, 1),atrous_rate=(2,2))(permute0) #512
        convalt1 = _altconv_bn_relu(nb_filter=nb_filtermat[0], nb_row=3, nb_col=3, input_shape=(None, 1024, 1024,1),  subsample=(1, 1),atrous_rate=(2,2))(convalt0) #512
        convalt2 = _altconv_bn_relu(nb_filter=nb_filtermat[0], nb_row=3, nb_col=3, input_shape=(None, 1024, 1024,1), subsample=(1, 1),atrous_rate=(2,2))(convalt1) #512
        convalt3 = _altconv_bn_relu(nb_filter=nb_filtermat[0], nb_row=3, nb_col=3, input_shape=(None, 1024, 1024,1), subsample=(1, 1),atrous_rate=(2,2))(convalt2) #512
        convalt4 = _altconv_bn_relu(nb_filter=nb_filtermat[0], nb_row=3, nb_col=3, input_shape=(None, 1024, 1024,1), subsample=(1, 1),atrous_rate=(2,2))(convalt3) #512
        
        conv0 = _conv_bn_relu(nb_filter=nb_filtermat[0], nb_row=3, nb_col=3, subsample=(2, 2))(convalt4) #512
        conv0  = Dropout((dr_rate))(conv0)
        if shape_add==512 and type2:
                conv0 = merge([conv0, conv11], mode='concat', concat_axis=CHANNEL_AXIS)
        conv1 = _conv_bn_relu(nb_filter=nb_filtermat[1], nb_row=3, nb_col=3, subsample=(2, 2))(conv0) #256
        conv1  = Dropout((dr_rate))(conv1)
        if shape_add==256 and type2:
                conv1 = merge([conv1, conv11], mode='concat', concat_axis=CHANNEL_AXIS)
        conv2 = _conv_bn_relu(nb_filter=nb_filtermat[2], nb_row=3, nb_col=3, subsample=(2, 2))(conv1) #128
        conv2  = Dropout((dr_rate))(conv2)
        if shape_add==128 and type2:
                conv2 = merge([conv2, conv11], mode='concat', concat_axis=CHANNEL_AXIS)
        conv3 = _conv_bn_relu(nb_filter=nb_filtermat[3], nb_row=3, nb_col=3, subsample=(2, 2))(conv2) #64
        conv3  = Dropout((dr_rate))(conv3)
        if shape_add==64 and type2:
                conv3 = merge([conv3, conv11], mode='concat', concat_axis=CHANNEL_AXIS)
        conv4 = _conv_bn_relu(nb_filter=nb_filtermat[4], nb_row=3, nb_col=3, subsample=(2, 2))(conv3) #32
        conv4  = Dropout((dr_rate))(conv4)
        if shape_add==32 and type2:
                conv4 = merge([conv4, conv11], mode='concat', concat_axis=CHANNEL_AXIS)
        conv5 = _conv_bn_relu(nb_filter=nb_filtermat[5], nb_row=3, nb_col=3, subsample=(2, 2))(conv4) #16
        conv5  = Dropout((dr_rate))(conv5)
##        if shape_add==16 and type2:
##                conv5 = merge([conv5, conv11], mode='concat', concat_axis=CHANNEL_AXIS)
##        conv6 = _conv_bn_relu(nb_filter=nb_filtermat[6], nb_row=3, nb_col=3, subsample=(2, 2))(conv5) #8
##        conv6  = Dropout((dr_rate))(conv6)
##        if shape_add==8 and type2:
##                conv6 = merge([conv6, conv11], mode='concat', concat_axis=CHANNEL_AXIS)
##        conv7 = _conv_bn_relu(nb_filter=nb_filtermat[7], nb_row=3, nb_col=3, subsample=(2, 2))(conv6) #4
##        conv7  = Dropout((dr_rate))(conv7)
##        if shape_add==4 and type2:
##                conv7 = merge([conv7, conv11], mode='concat', concat_axis=CHANNEL_AXIS)
##        conv8 = _conv_bn_relu(nb_filter=nb_filtermat[8], nb_row=3, nb_col=3, subsample=(2, 2))(conv7) #2
##        conv8  = Dropout((dr_rate))(conv8)
##        if shape_add==2 and type2:
##                conv8 = merge([conv8, conv11], mode='concat', concat_axis=CHANNEL_AXIS)
##        conv9 = _conv_bn_relu(nb_filter=nb_filtermat[9], nb_row=3, nb_col=3, subsample=(2, 2))(conv8) #1
##        conv9  = Dropout((dr_rate))(conv9)
##        if shape_add==1 and type2:
##                conv9 = merge([conv9, conv11], mode='concat', concat_axis=CHANNEL_AXIS)


            

        if unet:

            print('I reacher here.......................................................')
##            deconv1 = _deconv_bn_relu(nb_filter=nb_filtermat[8], nb_row=2, nb_col=2, output_shape=(None, 2, 2, nb_filtermat[8]), subsample=(2, 2))(conv9) #2
##            deconv1  = Dropout((dr_rate))(deconv1)
##            deconv1 = merge([conv8, deconv1], mode='concat', concat_axis=CHANNEL_AXIS)
##            
##            deconv2 = _deconv_bn_relu(nb_filter=nb_filtermat[7]*2, nb_row=2, nb_col=2, output_shape=(None, 4, 4,nb_filtermat[7]*2), subsample=(2, 2))(deconv1) #4
##            deconv2  = Dropout((dr_rate))(deconv2)
##            deconv2 = merge([conv7, deconv2], mode='concat', concat_axis=CHANNEL_AXIS)
##
##            deconv3 = _deconv_bn_relu(nb_filter=nb_filtermat[6]*2, nb_row=2, nb_col=2, output_shape=(None, 8, 8,nb_filtermat[6]*2), subsample=(2, 2))(deconv2) #8
##            deconv3  = Dropout((dr_rate))(deconv3)
##            deconv3 = merge([conv6, deconv3], mode='concat', concat_axis=CHANNEL_AXIS)
##
##            deconv4 = _deconv_bn_relu(nb_filter=nb_filtermat[5]*2, nb_row=2, nb_col=2, output_shape=(None, 16, 16,nb_filtermat[5]*2), subsample=(2, 2))(deconv3) #16
##            deconv4  = Dropout((dr_rate))(deconv4)
##            deconv4 = merge([conv5, deconv4], mode='concat', concat_axis=CHANNEL_AXIS)

            deconv5 = _deconv_bn_relu(nb_filter=nb_filtermat[4]*2, nb_row=2, nb_col=2, output_shape=(None, 32, 32,nb_filtermat[4]*2), subsample=(2, 2))(conv5) #32
            deconv5  = Dropout((dr_rate))(deconv5)
            deconv5 = merge([conv4, deconv5], mode='concat', concat_axis=CHANNEL_AXIS)

            deconv6 = _deconv_bn_relu(nb_filter=nb_filtermat[3]*2, nb_row=2, nb_col=2, output_shape=(None, 64, 64,nb_filtermat[3]*2), subsample=(2, 2))(deconv5) #64
            deconv6  = Dropout((dr_rate))(deconv6)
            deconv6 = merge([conv3, deconv6], mode='concat', concat_axis=CHANNEL_AXIS)

            deconv7 = _deconv_bn_relu(nb_filter=nb_filtermat[2]*2, nb_row=2, nb_col=2, output_shape=(None, 128, 128,nb_filtermat[2]*2), subsample=(2, 2))(deconv6) #128
            deconv7  = Dropout((dr_rate))(deconv7)
            deconv7 = merge([conv2, deconv7], mode='concat', concat_axis=CHANNEL_AXIS)

            deconv8 = _deconv_bn_relu(nb_filter=nb_filtermat[1]*2, nb_row=2, nb_col=2, output_shape=(None, 256, 256,nb_filtermat[1]*2), subsample=(2, 2))(deconv7) #256
            deconv8  = Dropout((dr_rate))(deconv8)
            deconv8 = merge([conv1, deconv8], mode='concat', concat_axis=CHANNEL_AXIS)

            deconv9 = _deconv_bn_relu(nb_filter=nb_filtermat[0]*2, nb_row=2, nb_col=2, output_shape=(None, 512, 512,nb_filtermat[0]*2), subsample=(2, 2))(deconv8) #256
            deconv9  = Dropout((dr_rate))(deconv9)
            deconv9 = merge([conv0, deconv9], mode='concat', concat_axis=CHANNEL_AXIS)
            
        else:
            deconv1 = _deconv_bn_relu(nb_filter=nb_filtermat[8], nb_row=2, nb_col=2, output_shape=(None, 2, 2,nb_filtermat[8]), subsample=(2, 2))(conv9) #2
            deconv1  = Dropout((dr_rate))(deconv1)
            
            deconv2 = _deconv_bn_relu(nb_filter=nb_filtermat[7], nb_row=2, nb_col=2, output_shape=(None, 4, 4,nb_filtermat[7]), subsample=(2, 2))(deconv1) #4
            deconv2  = Dropout((dr_rate))(deconv2)

            deconv3 = _deconv_bn_relu(nb_filter=nb_filtermat[6], nb_row=2, nb_col=2, output_shape=(None, 8, 8,nb_filtermat[6]), subsample=(2, 2))(deconv2) #8
            deconv3  = Dropout((dr_rate))(deconv3)

            deconv4 = _deconv_bn_relu(nb_filter=nb_filtermat[5], nb_row=2, nb_col=2, output_shape=(None, 16, 16,nb_filtermat[5]), subsample=(2, 2))(deconv3) #16
            deconv4  = Dropout((dr_rate))(deconv4)

            deconv5 = _deconv_bn_relu(nb_filter=nb_filtermat[4], nb_row=2, nb_col=2, output_shape=(None, 32, 32,nb_filtermat[4]), subsample=(2, 2))(deconv4) #32
            deconv5  = Dropout((dr_rate))(deconv5)

            deconv6 = _deconv_bn_relu(nb_filter=nb_filtermat[3], nb_row=2, nb_col=2, output_shape=(None, 64, 64,nb_filtermat[3]), subsample=(2, 2))(deconv5) #64
            deconv6  = Dropout((dr_rate))(deconv6)

            deconv7 = _deconv_bn_relu(nb_filter=nb_filtermat[2], nb_row=2, nb_col=2, output_shape=(None, 128, 128,nb_filtermat[2]), subsample=(2, 2))(deconv6) #128
            deconv7  = Dropout((dr_rate))(deconv7)

            deconv8 = _deconv_bn_relu(nb_filter=nb_filtermat[1], nb_row=2, nb_col=2, output_shape=(None, 256, 256,nb_filtermat[1]), subsample=(2, 2))(deconv7) #256
            deconv8  = Dropout((dr_rate))(deconv8)

            deconv9 = _deconv_bn_relu(nb_filter=nb_filtermat[0], nb_row=2, nb_col=2, output_shape=(None, 512, 512,nb_filtermat[0]), subsample=(2, 2))(deconv8) #256
            deconv9  = Dropout((dr_rate))(deconv9)

        
        deconv10 = _bn_relu_conv(nb_filter=1, nb_row=3, nb_col=3, subsample=(1, 1))(deconv9) #1
        deconv10 = Activation("relu")(deconv10)
        
        permute1=Permute((3,1,2))(deconv10)
        print('all done')

        if type2 and not imginput:
            model = Model(input=[aux_input,aux_input1], output=permute1)
        elif type2 and imginput:
            model = Model(input=[aux_input,aux_input2], output=permute1)
        else:
            model = Model(input=aux_input, output=permute1)

        plot(model, to_file='%s.png' % 'Type2_net', show_shapes=True, show_layer_names=True)

            
        return model
    
    @staticmethod
    def build_res(input_shapeaux, block_fn, repetitions, type2, shape_add=1, unet=True,imginput=True):
        """Builds a custom ResNet like architecture.
        Args:
            input_shape: The input shape in the form (nb_channels, nb_rows, nb_cols)
            num_outputs: The number of outputs at final softmax layer
            block_fn: The block function to use. This is either `basic_block` or `bottleneck`.
                The original paper used basic_block for layers < 50
            repetitions: Number of repetitions of various block units.
                At each block unit, the number of filters are doubled and the input size is halved
        Returns:
            The keras `Model`.
        """
        _handle_dim_ordering()
        dr_rate=0.05
        dr_rate2=0.05
       


        # Load function from str if needed.
        block_fn = _get_block(basic_block)
        rep=repetitions
        if type2 and not imginput:
            aux_input1 = Input(shape=(1,1,1), name='aux_input1')
            permute01=Permute((2,3,1))(aux_input1)
            conv11 = _conv_bn_relu(nb_filter=16, nb_row=1, nb_col=1, subsample=(1, 1))(permute01)

            ifl=True
            for kk in range(int(math.log(shape_add,2))):
                conv11 = _residual_block(block_fn, nb_filter=80-8*kk, repetitions=2, output_shape=(None,2**(kk+1),2**(kk+1),80-8*kk), is_first_layer=ifl)(conv11)
                ifl=False


        #nb_filtermat=[16, 24, 32, 48, 64, 96, 128, 192, 256, 384]
        nb_filtermat=[16, 32, 48, 64, 96, 128, 192, 256, 384, 512]

        aux_input = Input(shape=input_shapeaux, name='aux_input')        
        permute0=Permute((2,3,1))(aux_input)
        
        
        if type2 and imginput:
            aux_input2 = Input(shape=(1,1024,1024), name='aux_input2')
            permute01=Permute((2,3,1))(aux_input2)
            conv11=permute01
            ifl=True
            for kk in range(10-int(math.log(shape_add,2))):
                print(kk)
                conv11 = _residual_blockconv(block_fn, nb_filter=nb_filtermat[kk], repetitions=rep[0], is_first_layer=ifl)(conv11)
                conv11  = Dropout((dr_rate2))(conv11)
                ifl=False
               
     

        


        if shape_add==1024 and type2:
            if imginput:
                permute0 = merge([permute0, conv11], mode='concat', concat_axis=CHANNEL_AXIS)

        

        
        
        conv0 = _residual_blockconv(block_fn, nb_filter=nb_filtermat[0], repetitions=rep[0], is_first_layer=False)(permute0)  #256
        conv0  = Dropout((dr_rate))(conv0)

        if shape_add==512 and type2:
                conv0 = merge([conv0, conv11], mode='concat', concat_axis=CHANNEL_AXIS)
        conv1 = _residual_blockconvalt(block_fn, nb_filter=nb_filtermat[1], repetitions=rep[1], is_first_layer=False, input_shapess=(None, 256, 256,1), atrous_rate=(2,2))(conv0)  #256
        conv1  = Dropout((dr_rate))(conv1)
        if shape_add==256 and type2:
                conv1 = merge([conv1, conv11], mode='concat', concat_axis=CHANNEL_AXIS)
                
        conv2 = _residual_blockconvalt(block_fn, nb_filter=nb_filtermat[2], repetitions=rep[2], is_first_layer=False, input_shapess=(None, 128, 128,1), atrous_rate=(2,2))(conv1)  #128
        conv2  = Dropout((dr_rate))(conv2)
        if shape_add==128 and type2:
                conv2 = merge([conv2, conv11], mode='concat', concat_axis=CHANNEL_AXIS)
                
        conv3 = _residual_blockconvalt(block_fn, nb_filter=nb_filtermat[3], repetitions=rep[3], is_first_layer=False, input_shapess=(None, 64, 64,1), atrous_rate=(2,2))(conv2)  #64
        conv3  = Dropout((dr_rate))(conv3)
        if shape_add==64 and type2:
                conv3 = merge([conv3, conv11], mode='concat', concat_axis=CHANNEL_AXIS)
        conv4 = _residual_blockconv(block_fn, nb_filter=nb_filtermat[4], repetitions=rep[4], is_first_layer=False)(conv3)  #32
        conv4  = Dropout((dr_rate))(conv4)
        if shape_add==32 and type2:
                conv4 = merge([conv4, conv11], mode='concat', concat_axis=CHANNEL_AXIS)
        conv5 = _residual_blockconv(block_fn, nb_filter=nb_filtermat[5], repetitions=rep[5], is_first_layer=False)(conv4)  #16
        conv5  = Dropout((dr_rate))(conv5)
        if shape_add==16 and type2:
                conv5 = merge([conv5, conv11], mode='concat', concat_axis=CHANNEL_AXIS)
        conv6 = _residual_blockconv(block_fn, nb_filter=nb_filtermat[6], repetitions=rep[6], is_first_layer=False)(conv5)  #8
        conv6  = Dropout((dr_rate))(conv6)

                
        

            

        if unet:
            
            deconv4 = _residual_block(block_fn, nb_filter=nb_filtermat[5]*2, repetitions=rep[7], output_shape=(None, 16, 16, nb_filtermat[5]*2), is_first_layer=False)(conv6) #16
            deconv4  = Dropout((dr_rate))(deconv4)
            deconv4 = merge([conv5, deconv4], mode='concat', concat_axis=CHANNEL_AXIS)

            deconv5 = _residual_block(block_fn, nb_filter=nb_filtermat[4]*2, repetitions=rep[7], output_shape=(None, 32, 32, nb_filtermat[4]*2), is_first_layer=False)(deconv4) #32
            deconv5  = Dropout((dr_rate))(deconv5)
            deconv5 = merge([conv4, deconv5], mode='concat', concat_axis=CHANNEL_AXIS)

            deconv6 = _residual_block(block_fn, nb_filter=nb_filtermat[3]*2, repetitions=rep[7], output_shape=(None, 64, 64, nb_filtermat[3]*2), is_first_layer=False)(deconv5) #64
            deconv6  = Dropout((dr_rate))(deconv6)
            deconv6 = merge([conv3, deconv6], mode='concat', concat_axis=CHANNEL_AXIS)

            deconv7 = _residual_block(block_fn, nb_filter=nb_filtermat[2]*2, repetitions=rep[7], output_shape=(None, 128, 128, nb_filtermat[2]*2), is_first_layer=False)(deconv6) #128
            deconv7  = Dropout((dr_rate))(deconv7)
            deconv7 = merge([conv2, deconv7], mode='concat', concat_axis=CHANNEL_AXIS)

            deconv8 = _residual_block(block_fn, nb_filter=nb_filtermat[1]*2, repetitions=rep[7], output_shape=(None, 256, 256, nb_filtermat[1]*2), is_first_layer=False)(deconv7) #256
            deconv8  = Dropout((dr_rate))(deconv8)
            deconv8 = merge([conv1, deconv8], mode='concat', concat_axis=CHANNEL_AXIS)

            deconv9 = _residual_block(block_fn, nb_filter=nb_filtermat[0]*2, repetitions=rep[7], output_shape=(None, 512, 512, nb_filtermat[0]*2), is_first_layer=False)(deconv8) #256
            deconv9  = Dropout((dr_rate))(deconv9)
            deconv9 = merge([conv0, deconv9], mode='concat', concat_axis=CHANNEL_AXIS)
            
        else:

            #deconv1 = _deconv_bn_relu(nb_filter=nb_filtermat[8], nb_row=2, nb_col=2, output_shape=(None, 2, 2,nb_filtermat[8]), subsample=(2, 2))(conv9) #2
            deconv1 = _residual_block(block_fn, nb_filter=nb_filtermat[8], repetitions=rep[7], output_shape=(None, 2, 2, nb_filtermat[8]), is_first_layer=False)(conv9)
            deconv1  = Dropout((dr_rate))(deconv1)

            deconv2 = _residual_block(block_fn, nb_filter=nb_filtermat[7], repetitions=rep[7], output_shape=(None, 4, 4, nb_filtermat[7]), is_first_layer=False)(deconv1) #4
            deconv2  = Dropout((dr_rate))(deconv2)

            deconv3 = _residual_block(block_fn, nb_filter=nb_filtermat[6], repetitions=rep[7], output_shape=(None, 8, 8, nb_filtermat[6]), is_first_layer=False)(deconv2) #8
            deconv3  = Dropout((dr_rate))(deconv3)

            deconv4 = _residual_block(block_fn, nb_filter=nb_filtermat[5], repetitions=rep[7], output_shape=(None, 16, 16, nb_filtermat[5]), is_first_layer=False)(deconv3) #16
            deconv4  = Dropout((dr_rate))(deconv4)

            deconv5 = _residual_block(block_fn, nb_filter=nb_filtermat[4], repetitions=rep[7], output_shape=(None, 32, 32, nb_filtermat[4]), is_first_layer=False)(deconv4) #32
            deconv5  = Dropout((dr_rate))(deconv5)

            deconv6 = _residual_block(block_fn, nb_filter=nb_filtermat[3], repetitions=rep[7], output_shape=(None, 64, 64, nb_filtermat[3]), is_first_layer=False)(deconv5) #64
            deconv6  = Dropout((dr_rate))(deconv6)

            deconv7 = _residual_block(block_fn, nb_filter=nb_filtermat[2], repetitions=rep[7], output_shape=(None, 128, 128, nb_filtermat[2]), is_first_layer=False)(deconv6) #128
            deconv7  = Dropout((dr_rate))(deconv7)

            deconv8 = _residual_block(block_fn, nb_filter=nb_filtermat[1], repetitions=rep[7], output_shape=(None, 256, 256, nb_filtermat[1]), is_first_layer=False)(deconv7) #256
            deconv8  = Dropout((dr_rate))(deconv8)

            deconv9 = _residual_block(block_fn, nb_filter=nb_filtermat[0], repetitions=rep[7], output_shape=(None, 512, 512, nb_filtermat[0]), is_first_layer=False)(deconv8) #256
            deconv9  = Dropout((dr_rate))(deconv9)


        #deconv9 = _deconv_bn_relu(nb_filter=1, nb_row=2, nb_col=2, output_shape=(None, 256, 256,1), subsample=(1, 1))(deconv8) #256
        deconv10 = _residual_blockconv_final(block_fn, nb_filter=nb_filtermat[0], repetitions=rep[7], is_first_layer=False)(deconv9)  #1
        deconv11 = _residual_blockconv_final(block_fn, nb_filter=1, repetitions=rep[7], is_first_layer=False)(deconv10)  #1
        #deconv10 = _bn_relu_conv(nb_filter=16, nb_row=3, nb_col=3, subsample=(1, 1))(deconv9) #1
        #deconv11 = _bn_relu_conv(nb_filter=1, nb_row=3, nb_col=3, subsample=(1, 1))(deconv10) #1
        deconv11 = Activation("relu")(deconv11)
        
        permute1=Permute((3,1,2))(deconv11)
        print('all done')

        if type2 and not imginput:
            model = Model(input=[aux_input,aux_input1], output=permute1)
        elif type2 and imginput:
            model = Model(input=[aux_input,aux_input2], output=permute1)
        else:
            model = Model(input=aux_input, output=permute1)

        #plot(model, to_file='%s.png' % 'Type2_net', show_shapes=True, show_layer_names=True)
            
        return model


    @staticmethod
    def build_plainnet(input_shapeaux, type2, shape_add, unet,imginput):
        return ResnetBuilder.build_plain(input_shapeaux, type2, shape_add, unet,imginput)

    @staticmethod
    def build_resnet(input_shapeaux, type2, shape_add, unet,imginput):
        return ResnetBuilder.build_res(input_shapeaux, basic_block, [2, 2, 2, 2, 2, 2, 2, 2, 2], type2, shape_add, unet,imginput)


if __name__ == '__main__':

    input_shapeaux= (1,512,512)
    ResnetBuilder.build_resnet(input_shapeaux, type2=True, shape_add=4, unet=True)
    #ResnetBuilder.build_plainnet(input_shapeaux, type2=True, shape_add=128, unet=False)


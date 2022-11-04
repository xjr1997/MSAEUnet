from xml.dom.expatbuilder import FilterVisibilityController
from keras_applications import get_submodules_from_kwargs
import tensorflow as tf

from ._common_blocks import Conv2dBn
from ._utils import freeze_model, filter_keras_submodules
from ..backbones.backbones_factory import Backbones
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, Reshape, Dense, multiply, Permute, Concatenate, Conv2D, Add, Activation, Lambda,Conv1D
import keras
import keras.backend as K
backend = None
layers = None
models = None
keras_utils = None


# ---------------------------------------------------------------------
#  Utility functions
# ---------------------------------------------------------------------

def get_submodules():
    return {
        'backend': backend,
        'models': models,
        'layers': layers,
        'utils': keras_utils,
    }


# ---------------------------------------------------------------------
#  Blocks
# ---------------------------------------------------------------------

def Conv3x3BnReLU(filters, use_batchnorm, name=None):
    kwargs = get_submodules()

    def wrapper(input_tensor):
        return Conv2dBn(
            filters,
            kernel_size=3,
            activation='relu',
            kernel_initializer='he_uniform',
            padding='same',
            use_batchnorm=use_batchnorm,
            name=name,
            **kwargs
        )(input_tensor)

    return wrapper

def Conv1x1BnReLU(filters, use_batchnorm, name=None):
    kwargs = get_submodules()

    def wrapper(input_tensor):
        return Conv2dBn(
            filters,
            kernel_size=1,
            activation='relu',
            kernel_initializer='he_uniform',
            padding='same',
            use_batchnorm=use_batchnorm,
            name=name,
            **kwargs
        )(input_tensor)

    return wrapper

def Conv1x1Bnsigm(filters, use_batchnorm, name=None):
    kwargs = get_submodules()

    def wrapper(input_tensor):
        return Conv2dBn(
            filters,
            kernel_size=1,
            activation='sigmoid',
            kernel_initializer='he_uniform',
            padding='same',
            use_batchnorm=use_batchnorm,
            name=name,
            **kwargs
        )(input_tensor)

    return wrapper

def attention(filters, use_batchnorm=False):
    concat_axis = 3 if backend.image_data_format() == 'channels_last' else 1

    def wrapper(input_tensor , skip):
        gating= Conv1x1BnReLU(1, use_batchnorm)(input_tensor)
        skip1= Conv1x1BnReLU(1, use_batchnorm)(skip)
        x =layers.Concatenate(axis=concat_axis)([gating, skip1])
        #x = layers.add([gating , skip])
        x = layers.Activation('relu')(x)
        a = Conv1x1Bnsigm(1, use_batchnorm)(x)
        y = layers.multiply([a, skip])
        y= Conv1x1BnReLU(filters, use_batchnorm)(y)
        return y

    return wrapper

def SE(filters, use_batchnorm=False):
    concat_axis = 3 if backend.image_data_format() == 'channels_last' else 1

    def wrapper(input_tensor ):
        ave=  layers.GlobalAveragePooling2D()(input_tensor)
        #max= layers.GlobalMaxPooling2D()(input_tensor)
        fc1 = layers.Dense(filters//16, kernel_initializer='he_normal', activation='relu',
                                use_bias=True, bias_initializer='zeros')(ave)
        fc2 = layers.Dense(input_tensor.shape[-1], kernel_initializer='he_normal', activation='relu',
                                use_bias=True, bias_initializer='zeros')(fc1)                        
        #out = tf.reduce_sum(out, axis=1)(fc2)      		# shape=(256, 512)
        out = layers.Activation('sigmoid')(fc2)
        out = layers.Reshape((1, 1, out.shape[1]))(out)
        y = layers.multiply([input_tensor, out])
        return y

    return wrapper
def SE1(filters, use_batchnorm=False):
    concat_axis = 3 if backend.image_data_format() == 'channels_last' else 1

    def wrapper(input_tensor ):
        ave=  layers.GlobalAveragePooling2D()(input_tensor)
        max= layers.GlobalMaxPooling2D()(input_tensor)
        x = layers.add([ave , max])
        fc1 = layers.Dense(filters//16, kernel_initializer='he_normal', activation='relu',
                                use_bias=True, bias_initializer='zeros')(x)
        fc2 = layers.Dense(input_tensor.shape[-1], kernel_initializer='he_normal', activation='relu',
                                use_bias=True, bias_initializer='zeros')(fc1)                        
        #out = tf.reduce_sum(out, axis=1)(fc2)      		# shape=(256, 512)
        out = layers.Activation('sigmoid')(fc2)
        out = layers.Reshape((1, 1, out.shape[1]))(out)
        y = layers.multiply([input_tensor, out])
        return y

    return wrapper
def SE2(filters, use_batchnorm=False):
    concat_axis = 3 if backend.image_data_format() == 'channels_last' else 1

    def wrapper(input_tensor ):
        
        ave= Conv1x1BnReLU(filters, use_batchnorm)(input_tensor)
        max= Conv3x3BnReLU(filters, use_batchnorm)(input_tensor)
        x = layers.add([ave , max])
        x=  layers.GlobalAveragePooling2D()(x)
        fc1 = layers.Dense(filters//16, kernel_initializer='he_normal', activation='relu',
                                use_bias=True, bias_initializer='zeros')(x)
        fc2 = layers.Dense(input_tensor.shape[-1], kernel_initializer='he_normal', activation='relu',
                                use_bias=True, bias_initializer='zeros')(fc1)                        
        #out = tf.reduce_sum(out, axis=1)(fc2)      		# shape=(256, 512)
        out = layers.Activation('sigmoid')(fc2)
        out = layers.Reshape((1, 1, out.shape[1]))(out)
        y = layers.multiply([input_tensor, out])
        return y
    
    return wrapper
    
def spatial_attention(input_feature):

    avg_pool = Lambda(lambda x:K.mean(x,axis=3,keepdims=True))(input_feature)
    max_pool = Lambda(lambda x:K.max(x,axis=3,keepdims=True))(input_feature)

    concat = Concatenate(axis=3)([avg_pool,max_pool])
    cbam_feature = Conv2D(1,(7,7),strides=1,padding='same',activation='sigmoid')(concat)

    return multiply([input_feature,cbam_feature])

def DecoderUpsamplingX2Block(filters, decoder_filters,stage, use_batchnorm=False):
    up_name = 'decoder_stage{}_upsampling'.format(stage)
    gat_name = 'decoder_stage{}_gating'.format(stage)
    conv1_name = 'decoder_stage{}a'.format(stage)
    down_name = 'decoder_stage{}'.format(stage)
    conv2_name = 'decoder_stage{}b'.format(stage)
    concat_name = 'decoder_stage{}_concat'.format(stage)

    concat_axis = 3 if backend.image_data_format() == 'channels_last' else 1

    def wrapper(input_tensor, skip=None):
        x = layers.UpSampling2D(size=2, name=up_name)(input_tensor)
        

        if skip is not None:
            gating = layers.UpSampling2D(size=2, name=gat_name)(input_tensor)
            a=attention(64, use_batchnorm=use_batchnorm)(gating,skip[stage])
            x = layers.Concatenate(axis=concat_axis, name=concat_name)([x, a])
            
            for i in range(stage+1,4):   
                b = layers.MaxPooling2D(pool_size=(2**(i-stage)),strides =(2**(i-stage)), name='kuayue_stage{}{}'.format(stage,i))(skip[i])
                s = SE(64, use_batchnorm=use_batchnorm)(b)
                b= Conv3x3BnReLU(64, use_batchnorm)(b)
                x = layers.Concatenate(axis=concat_axis, name='kuayue_stage{}{}_concat1'.format(stage,i))([x, s])
                x = layers.Concatenate(axis=concat_axis, name='kuayue_stage{}{}_concat2'.format(stage,i))([x, b])

        x = Conv3x3BnReLU(filters, use_batchnorm, name=conv1_name)(x)
        x = Conv3x3BnReLU(filters, use_batchnorm, name=conv2_name)(x)

        return x

    return wrapper


def DecoderTransposeX2Block(filters, stage, use_batchnorm=False):
    transp_name = 'decoder_stage{}a_transpose'.format(stage)
    bn_name = 'decoder_stage{}a_bn'.format(stage)
    relu_name = 'decoder_stage{}a_relu'.format(stage)
    conv_block_name = 'decoder_stage{}b'.format(stage)
    concat_name = 'decoder_stage{}_concat'.format(stage)

    concat_axis = bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1

    def layer(input_tensor, skip=None):

        x = layers.Conv2DTranspose(
            filters,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding='same',
            name=transp_name,
            use_bias=not use_batchnorm,
        )(input_tensor)

        if use_batchnorm:
            x = layers.BatchNormalization(axis=bn_axis, name=bn_name)(x)

        x = layers.Activation('relu', name=relu_name)(x)

        if skip is not None:
            x = layers.Concatenate(axis=concat_axis, name=concat_name)([x, skip])

        x = Conv3x3BnReLU(filters, use_batchnorm, name=conv_block_name)(x)

        return x

    return layer
def DoubleConv3x3BnReLU(filters, use_batchnorm, name=None):
    name1, name2 = None, None
    if name is not None:
        name1 = name + 'a'
        name2 = name + 'b'

    def wrapper(input_tensor):
        x = Conv3x3BnReLU(filters, use_batchnorm, name=name1)(input_tensor)
        x = Conv3x3BnReLU(filters, use_batchnorm, name=name2)(x)
        return x

    return wrapper

# ---------------------------------------------------------------------
#  Unet Decoder
# ---------------------------------------------------------------------

def build_unet(
        backbone,
        decoder_block,
        skip_connection_layers,
        decoder_filters=(256, 128, 64, 32, 16),
        n_upsample_blocks=5,
        classes=1,
        activation='sigmoid',
        use_batchnorm=True,
):
    input_ = backbone.input
    x = backbone.output

    # extract skip connections
    skips = ([backbone.get_layer(name=i).output if isinstance(i, str)
              else backbone.get_layer(index=i).output for i in skip_connection_layers])

    # add center block if previous operation was maxpooling (for vgg models)
    if isinstance(backbone.layers[-1], layers.MaxPooling2D):
        concat_axis = 3 if backend.image_data_format() == 'channels_last' else 1
        x = Conv3x3BnReLU(512, use_batchnorm, name='center_block1')(x)
        x2 = SE1(512, use_batchnorm=use_batchnorm)(x)
        x3 = SE2(512, use_batchnorm=use_batchnorm)(x)
        x4 = layers.Concatenate(axis=concat_axis)([x2, x3])
        x = spatial_attention(x4)
        x = Conv3x3BnReLU(512, use_batchnorm, name='center_block2')(x)

    # building decoder blocks
    #for i in range(n_upsample_blocks):

        #if i < len(skips):
            #skip = skips
        #else:
            #skip = None
    concat_axis = 3 if backend.image_data_format() == 'channels_last' else 1
    x5 = decoder_block(decoder_filters[0],decoder_filters, stage=0, use_batchnorm=use_batchnorm)(x, skips)
    s51 = layers.UpSampling2D((2, 2), interpolation='nearest', name='upsampling_stage5')(x)
    x5 = layers.Concatenate(axis=concat_axis, name='UP1_concat')([s51, x5])
    x4 = decoder_block(decoder_filters[1],decoder_filters, stage=1, use_batchnorm=use_batchnorm)(x5, skips)
    s41 = layers.UpSampling2D((4, 4), interpolation='nearest', name='upsampling_stage41')(x)
    s42 = layers.UpSampling2D((2, 2), interpolation='nearest', name='upsampling_stage42')(x5)
    x4 = layers.Concatenate(axis=concat_axis, name='UP2_concat')([s42, s41, x4])
    x3 = decoder_block(decoder_filters[2],decoder_filters, stage=2, use_batchnorm=use_batchnorm)(x4, skips)
    s31 = layers.UpSampling2D((8, 8), interpolation='nearest', name='upsampling_stage31')(x)
    s32 = layers.UpSampling2D((4, 4), interpolation='nearest', name='upsampling_stage32')(x5)
    s33 = layers.UpSampling2D((2, 2), interpolation='nearest', name='upsampling_stage33')(x4)
    x3 = layers.Concatenate(axis=concat_axis, name='UP3_concat')([s31, s32, s33, x3])
    x2 = decoder_block(decoder_filters[3],decoder_filters, stage=3, use_batchnorm=use_batchnorm)(x3, skips)
    

    # add segmentation head to each
    #s5 = DoubleConv3x3BnReLU(128, use_batchnorm, name='segm_stage5')(x5)
    #s4 = DoubleConv3x3BnReLU(128, use_batchnorm, name='segm_stage4')(x4)
    #s3 = DoubleConv3x3BnReLU(128, use_batchnorm, name='segm_stage3')(x3)
    #s2 = DoubleConv3x3BnReLU(128, use_batchnorm, name='segm_stage2')(x2)

    # upsampling to same resolution
    #s5 = layers.UpSampling2D((8, 8), interpolation='nearest', name='upsampling_stage5')(s5)
    #s4 = layers.UpSampling2D((4, 4), interpolation='nearest', name='upsampling_stage4')(s4)
    #s3 = layers.UpSampling2D((2, 2), interpolation='nearest', name='upsampling_stage3')(s3)
    # model head (define number of output classes)
    #concat_axis = 3 if backend.image_data_format() == 'channels_last' else 1
    #x = layers.Concatenate(axis=concat_axis, name='aggregation_concat')([s3, s4, s5])
    #x = Conv3x3BnReLU(decoder_filters[3], use_batchnorm, name='last_stage')(x)
    #x = layers.Concatenate(axis=concat_axis, name='aggregation_concat2')([s2,x])
    x = Conv3x3BnReLU(decoder_filters[4], use_batchnorm, name='final_stage')(x2)
    x = layers.UpSampling2D(size=(2, 2), interpolation='bilinear', name='final_upsampling')(x)
    x = layers.Conv2D(
        filters=classes,
        kernel_size=(3, 3),
        padding='same',
        use_bias=True,
        kernel_initializer='glorot_uniform',
        name='final_conv',
    )(x)
    x = layers.Activation(activation, name=activation)(x)

    # create keras model instance
    model = models.Model(input_, x)

    return model


# ---------------------------------------------------------------------
#  Unet Model
# ---------------------------------------------------------------------

def Unet(
        backbone_name='vgg16',
        input_shape=(None, None, 3),
        classes=1,
        activation='sigmoid',
        weights=None,
        encoder_weights='imagenet',
        encoder_freeze=False,
        encoder_features='default',
        decoder_block_type='upsampling',
        decoder_filters=(256, 128, 64, 32, 16),
        decoder_use_batchnorm=True,
        **kwargs
):
    """ Unet_ is a fully convolution neural network for image semantic segmentation

    Args:
        backbone_name: name of classification model (without last dense layers) used as feature
            extractor to build segmentation model.
        input_shape: shape of input data/image ``(H, W, C)``, in general
            case you do not need to set ``H`` and ``W`` shapes, just pass ``(None, None, C)`` to make your model be
            able to process images af any size, but ``H`` and ``W`` of input images should be divisible by factor ``32``.
        classes: a number of classes for output (output shape - ``(h, w, classes)``).
        activation: name of one of ``keras.activations`` for last model layer
            (e.g. ``sigmoid``, ``softmax``, ``linear``).
        weights: optional, path to model weights.
        encoder_weights: one of ``None`` (random initialization), ``imagenet`` (pre-training on ImageNet).
        encoder_freeze: if ``True`` set all layers of encoder (backbone model) as non-trainable.
        encoder_features: a list of layer numbers or names starting from top of the model.
            Each of these layers will be concatenated with corresponding decoder block. If ``default`` is used
            layer names are taken from ``DEFAULT_SKIP_CONNECTIONS``.
        decoder_block_type: one of blocks with following layers structure:

            - `upsampling`:  ``UpSampling2D`` -> ``Conv2D`` -> ``Conv2D``
            - `transpose`:   ``Transpose2D`` -> ``Conv2D``

        decoder_filters: list of numbers of ``Conv2D`` layer filters in decoder blocks
        decoder_use_batchnorm: if ``True``, ``BatchNormalisation`` layer between ``Conv2D`` and ``Activation`` layers
            is used.

    Returns:
        ``keras.models.Model``: **Unet**

    .. _Unet:
        https://arxiv.org/pdf/1505.04597

    """

    global backend, layers, models, keras_utils
    submodule_args = filter_keras_submodules(kwargs)
    backend, layers, models, keras_utils = get_submodules_from_kwargs(submodule_args)
   
    if decoder_block_type == 'upsampling':
        
        decoder_block = DecoderUpsamplingX2Block
    elif decoder_block_type == 'transpose':
        
        decoder_block = DecoderTransposeX2Block
    else:
        raise ValueError('Decoder block type should be in ("upsampling", "transpose"). '
                         'Got: {}'.format(decoder_block_type))

    backbone = Backbones.get_backbone(
        backbone_name,
        input_shape=input_shape,
        weights=encoder_weights,
        include_top=False,
        **kwargs,
    )

    if encoder_features == 'default':
        encoder_features = Backbones.get_feature_layers(backbone_name, n=4)

    model = build_unet(
        backbone=backbone,
        decoder_block=decoder_block,
        skip_connection_layers=encoder_features,
        decoder_filters=decoder_filters,
        classes=classes,
        activation=activation,
        n_upsample_blocks=len(decoder_filters),
        use_batchnorm=decoder_use_batchnorm,
    )

    # lock encoder weights for fine-tuning
    if encoder_freeze:
        freeze_model(backbone, **kwargs)

    # loading model weights
    if weights is not None:
        model.load_weights(weights)

    return model
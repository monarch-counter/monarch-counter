from numpy.lib.arraypad import pad
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Conv2D, 
    Conv2DTranspose,
    Activation, 
    BatchNormalization, 
    Flatten,
    Input, 
    MaxPool2D, 
    Dropout, 
    concatenate
)
from tensorflow.python.keras.layers.merge import add
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow import math as tf_math
from tensorflow.python.keras.utils.generic_utils import default


#
### Metrics 
#
def mean_percent_count_err(y_true, y_pred):
    """Function that calculates a custom accuracy metric for the UNet

    Args:
        y_true (Tensor): 4D tensor containing the batch's true labels
        y_pred (Tensor): 4D tensor containing the batch's predicted labels

    Returns:
        float: Mean error in the counts of the objects
    """
    # y_true_counts = tf_math.scalar_mul(0.001, tf_math.reduce_sum(y_true, axis=1))
    # y_pred_counts = tf_math.scalar_mul(0.001, tf_math.reduce_sum(y_pred, axis=1))
    y_true_counts = tf_math.reduce_sum(y_true, axis=1)
    y_pred_counts = tf_math.reduce_sum(y_pred, axis=1)
    count_delta = tf_math.abs(tf_math.subtract(y_true_counts, y_pred_counts))
    count_delta_percent = tf_math.scalar_mul(100.0, tf_math.divide_no_nan(count_delta, y_true_counts))
    
    return tf_math.reduce_mean(count_delta_percent)


#
### MultiResUNet blocks
#  
def get_multires_conv_block(inputs=None, n_filters=32, dropout_prob=0.5, max_pooling=True, alpha=1.67):
    """Convolutional Downsampling block based on MultiResUNet paper

    Args:
        inputs ([Tensor], optional): Input tensor to the. Defaults to None.
        n_filters ([int], optional): Number of filters (kernels) in the convolutional operations. Defaults to 32.
        dropout_prob ([float], optional): Probability of the dropout layer. Defaults to 0.5.
        max_pooling ([bool], optional): Whether the end result is to be reduced spatially. Defaults to True.
        alpha ([float], optional): Multiplyer for the number of filters (kernels). Defaults to 1.67.

    Returns:
        x, skip_connection ([Tensor], [Tensor]): A tuple of tensors that are passed to the next layer and corresponding 
            up-sampling layer
    """
    
    # A multiplyer for the original number of filters (kernels)
    W = n_filters * alpha

    # 1v1 for later use
    n_filters_1x1 = int(W * 0.167) + int(W * 0.333) + int(W * 0.5)
    shortcut = Conv2D(n_filters_1x1, 1, padding='same', kernel_initializer='he_normal')(inputs)
    shortcut = BatchNormalization()(shortcut)
    
    # First 3x3 conv
    conv3x3 = Conv2D(int(W * 0.167), 3, padding='same', kernel_initializer='he_normal')(inputs)
    conv3x3 = BatchNormalization()(conv3x3)
    conv3x3 = Activation(activation='relu')(conv3x3)

    # Second 3x3 conv (emulating 5x5 conv)
    conv5x5 = Conv2D(int(W * 0.333), 3, padding='same', kernel_initializer='he_normal')(conv3x3)
    conv5x5 = BatchNormalization()(conv5x5)
    conv5x5 = Activation(activation='relu')(conv5x5)

    # Third 3x3 conv (emulating 7x7 conv)
    conv7x7 = Conv2D(int(W * 0.5), 3, padding='same', kernel_initializer='he_normal')(conv5x5)
    conv7x7 = BatchNormalization()(conv7x7)
    conv7x7 = Activation(activation='relu')(conv7x7)

    # Concatinate the results of all 3 convolutions to get a quasi-inception layer
    concat = concatenate([conv3x3, conv5x5, conv7x7], axis=3)
    concat = BatchNormalization()(concat)

    # Merge the 1x1 and quasi-inception layer, and apply an activation 
    x = add([shortcut, concat])
    x = Activation('relu')(x)
    
    if dropout_prob > 0.0:
        x = Dropout(rate=dropout_prob)(x)

    if max_pooling:
        out = MaxPool2D(pool_size=(2, 2))(x)
    else:
        out = x
    
    skip_connection = x
    
    return out, skip_connection

def get_multires_deconv_block(expansive_inputs, contractive_inputs, n_filters=32, alpha=1.67):
    """Convolutional Upsampling block based on MultiResUNet paper

    Args:
        expansive_inputs ([Tensor]): The input tensor which comes from the previous upsampling block
        contractive_inputs ([Tensor]): The input tensor which comes from the corresponding downsampling block
        n_filters ([int], optional): The number of filters that are to be used in the Conv/DeConv operations. Defaults to 32.
        alpha ([float], optional): Multiplier for the number of filters (kernels). Defaults to 1.67.

    Returns:
        result ([Tensor]): Output tensor for the next upsampling block
    """
    x = Conv2DTranspose(n_filters, (3, 3), strides=(2, 2), padding='same')(expansive_inputs)
    merged = concatenate([x, contractive_inputs], axis=3)

    # A multiplyer for the original number of filters (kernels)
    W = n_filters * alpha

    # 1v1 for later use
    n_filters_1x1 = int(W * 0.167) + int(W * 0.333) + int(W * 0.5)
    shortcut = Conv2D(n_filters_1x1, 1, padding='same', kernel_initializer='he_normal')(merged)
    shortcut = BatchNormalization()(shortcut)
    
    # First 3x3 conv
    conv3x3 = Conv2D(int(W * 0.167), 3, padding='same', kernel_initializer='he_normal')(merged)
    conv3x3 = BatchNormalization()(conv3x3)
    conv3x3 = Activation(activation='relu')(conv3x3)

    # Second 3x3 conv (emulating 5x5 conv)
    conv5x5 = Conv2D(int(W * 0.333), 3, padding='same', kernel_initializer='he_normal')(conv3x3)
    conv5x5 = BatchNormalization()(conv5x5)
    conv5x5 = Activation(activation='relu')(conv5x5)

    # Third 3x3 conv (emulating 7x7 conv)
    conv7x7 = Conv2D(int(W * 0.5), 3, padding='same', kernel_initializer='he_normal')(conv5x5)
    conv7x7 = BatchNormalization()(conv7x7)
    conv7x7 = Activation(activation='relu')(conv7x7)

    # Concatinate the results of all 3 convolutions to get a quasi-inception layer
    concat = concatenate([conv3x3, conv5x5, conv7x7], axis=3)
    concat = BatchNormalization()(concat)

    # Merge the 1x1 and quasi-inception layer, and apply an activation 
    x = add([shortcut, concat])
    x = Activation('relu')(x)
    result = BatchNormalization()(x)

    return result


# 
### Vanilla UNet blocks
#
def get_default_conv_block(inputs=None, n_filters=32, dropout_prob=0.5, max_pooling=True):
    """Convolutional downsampling block (vanilla) in UNet

    Args:
        inputs ([Tensor], optional): Input tensor. Defaults to None.
        n_filters ([int], optional): Number of filters (kernels) in Conv operation. Defaults to 32.
        dropout_prob ([float], optional): Dropout probability for Dropout layers. Defaults to 0.5.
        max_pooling ([bool], optional): Whether the end result is to be reduced in spatial dimension. 
            Defaults to True.
    Returns:
        result [Tensor]: Output tensor.
        skip_connection [Tensor]: Skip connection tensor.
    """
    x = Conv2D(n_filters, 3, padding='same', 
            kernel_initializer='he_normal')(inputs)
    x = BatchNormalization()(x)
    x = Activation(activation='relu')(x)
    
    x = Conv2D(n_filters, 3, padding='same',
            kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation(activation='relu')(x)
    
    if dropout_prob > 0:
        x = Dropout(dropout_prob)(x)
    
    if max_pooling:
        result = MaxPool2D(pool_size=(2, 2))(x)
    else:
        result = x
    
    skip_connection = x
    
    return result, skip_connection

def get_default_deconv_block(expansive_inputs, contractive_inputs, n_filters=32):
    """Convolutional upsampling block (vanilla) in Unet

    Args:
        expansive_inputs ([Tensor]): Input tensor from previous layer
        contractive_input ([Tensor]): Input tensor from corresponding downsampling layer
        n_filters ([int], optional): Number of filters (kernels) in the Conv operations. Defaults to 32.

    Returns:
        result [Tensor]: Output tensor
    """
    x = Conv2DTranspose(n_filters, (3, 3), strides=(2, 2), 
            padding='same')(expansive_inputs)
    
    merged = concatenate([x, contractive_inputs], axis=3)

    x = Conv2D(n_filters, 3, padding='same',
            kernel_initializer='he_normal')(merged)
    x = BatchNormalization()(x)
    x = Activation(activation='relu')(x)

    result = Conv2D(n_filters, 3, padding='same',
            activation='relu',
            kernel_initializer='he_normal')(x)

    return result


#
### Inception like UNet block with dilated kernels
#
def get_dilated_conv_block(inputs=None, n_filters=32, dropout_prob=0.5, max_pooling=True, alpha=1.67):
    # A multiplyer for the original number of filters (kernels)
    W = n_filters * alpha

    # 1v1 for later use
    n_filters_1x1 = int(W * 0.167) + int(W * 0.333) + int(W * 0.5)
    shortcut = Conv2D(n_filters_1x1, 1, padding='same', kernel_initializer='he_normal')(inputs)
    shortcut = BatchNormalization()(shortcut)
    
    # First 3x3 conv
    conv3x3 = Conv2D(int(W * 0.167), 3, padding='same', kernel_initializer='he_normal')(inputs)
    conv3x3 = BatchNormalization()(conv3x3)
    conv3x3 = Activation(activation='relu')(conv3x3)

    # Second 3x3 conv (emulating 5x5 conv)
    conv5x5 = Conv2D(int(W * 0.333), 3, padding='same', dilation_rate=2, kernel_initializer='he_normal')(inputs)
    conv5x5 = BatchNormalization()(conv5x5)
    conv5x5 = Activation(activation='relu')(conv5x5)

    # Third 3x3 conv (emulating 7x7 conv)
    conv7x7 = Conv2D(int(W * 0.5), 3, padding='same', dilation_rate=3, kernel_initializer='he_normal')(inputs)
    conv7x7 = BatchNormalization()(conv7x7)
    conv7x7 = Activation(activation='relu')(conv7x7)

    # Concatinate the results of all 3 convolutions to get a quasi-inception layer
    concat = concatenate([conv3x3, conv5x5, conv7x7], axis=3)
    concat = BatchNormalization()(concat)

    # Merge the 1x1 and quasi-inception layer, and apply an activation 
    x = add([shortcut, concat])
    x = Activation('relu')(x)
    
    if dropout_prob > 0.0:
        x = Dropout(rate=dropout_prob)(x)

    if max_pooling:
        out = MaxPool2D(pool_size=(2, 2))(x)
    else:
        out = x
    
    skip_connection = x
    
    return out, skip_connection

def get_dilated_deconv_block(expansive_inputs, contractive_inputs, n_filters=32, alpha=1.67):
    x = Conv2DTranspose(n_filters, (3, 3), strides=(2, 2), padding='same')(expansive_inputs)
    merged = concatenate([x, contractive_inputs], axis=3)

    # A multiplyer for the original number of filters (kernels)
    W = n_filters * alpha

    # 1v1 for later use
    n_filters_1x1 = int(W * 0.167) + int(W * 0.333) + int(W * 0.5)
    shortcut = Conv2D(n_filters_1x1, 1, padding='same', kernel_initializer='he_normal')(merged)
    shortcut = BatchNormalization()(shortcut)
    
    # First 3x3 conv
    conv3x3 = Conv2D(int(W * 0.167), 3, padding='same', kernel_initializer='he_normal')(merged)
    conv3x3 = BatchNormalization()(conv3x3)
    conv3x3 = Activation(activation='relu')(conv3x3)

    # Second 3x3 conv (emulating 5x5 conv)
    conv5x5 = Conv2D(int(W * 0.333), 3, padding='same', dilation_rate=2, kernel_initializer='he_normal')(merged)
    conv5x5 = BatchNormalization()(conv5x5)
    conv5x5 = Activation(activation='relu')(conv5x5)

    # Third 3x3 conv (emulating 7x7 conv)
    conv7x7 = Conv2D(int(W * 0.5), 3, padding='same', dilation_rate=3, kernel_initializer='he_normal')(merged)
    conv7x7 = BatchNormalization()(conv7x7)
    conv7x7 = Activation(activation='relu')(conv7x7)

    # Concatinate the results of all 3 convolutions to get a quasi-inception layer
    concat = concatenate([conv3x3, conv5x5, conv7x7], axis=3)
    concat = BatchNormalization()(concat)

    # Merge the 1x1 and quasi-inception layer, and apply an activation 
    x = add([shortcut, concat])
    x = Activation('relu')(x)
    result = BatchNormalization()(x)

    return result

#
### Connection paths
def __resnet_block(inputs, n_filters):
    """Helper function to create ResNet blocks for skip connections

    Args:
        inputs (Tensor): Input to the ResNet block
        n_filters (int, optional): No. of filters in the Conv operations. Defaults to 16.

    Returns:
        Tensor: Output tensor of the ResNet block
    """
    shortcut = Conv2D(n_filters, 1, padding='same')(inputs)
    shortcut = BatchNormalization()(shortcut)
    
    out = Conv2D(n_filters, 3, padding='same', kernel_initializer='he_normal')(inputs)
    out = BatchNormalization()(out)
    out = Activation(activation='relu')(out)

    result = add([shortcut, out])
    result = BatchNormalization()(result)
    result = Activation(activation='relu')(result)

    return result

def get_unet_connection(inputs=None, n_filters=32, path_type='default'):
    if path_type == 'default':
        return inputs
    elif path_type == 'resnet':
        x = inputs
        for _ in range(3):
            y = __resnet_block(x, n_filters=n_filters)
            x = y
        return y

#
### Wrapper for UNet blocks
#
def get_unet_conv_block(inputs=None, n_filters=32, dropout_prob=0.5, max_pooling=True, block_type='default'):
    if block_type == 'default':
        return get_default_conv_block(
            inputs=inputs, 
            n_filters=n_filters, 
            dropout_prob=dropout_prob, 
            max_pooling=max_pooling
        )
    elif block_type == 'multires':
        return get_multires_conv_block(
            inputs=inputs, 
            n_filters=n_filters, 
            dropout_prob=dropout_prob, 
            max_pooling=max_pooling
        )
    elif block_type == 'dilated_multires':
        return get_dilated_conv_block(
            inputs=inputs, 
            n_filters=n_filters, 
            dropout_prob=dropout_prob, 
            max_pooling=max_pooling
        )

def get_unet_deconv_block(expansive_inputs, contractive_inputs, n_filters=32, block_type='default'):
    if block_type == 'default':
        return get_default_deconv_block(
            expansive_inputs=expansive_inputs,
            contractive_inputs=contractive_inputs,
            n_filters=n_filters
        )
    elif block_type == 'multires':
        return get_multires_deconv_block(
            expansive_inputs=expansive_inputs,
            contractive_inputs=contractive_inputs,
            n_filters=n_filters
        )
    elif block_type == 'dilated_multires':
        return get_dilated_deconv_block(
            expansive_inputs=expansive_inputs,
            contractive_inputs=contractive_inputs,
            n_filters=n_filters
        )


#
### UNet model 
#
def get_unet(
    img_size=512, n_filters=32, lr=5e-5, dropout_prob=0.0, 
    block_type='default', skip_connection_type='default'):
    """Creates a UNet and returns the model instance

    Args:
        input_size (tuple, optional): 3D tuple containing (l, w, n_channels) of input image. Defaults to (1024, 1024, 3).
        n_filters (int, optional): Number of filters (kernels) in the Conv operations. Defaults to 32.
        lr (float, optional): Learning rate for the optimizer. Defaults to 5e-5.
        dropout_prob (float, optional): Probability for the dropout layers 

    Returns:
        Model: Model instance
    """
    nf = n_filters
    dp = dropout_prob
    valid_block_types = ['default', 'multires', 'dilated_multires']

    print("DEBUG:: unet_blocks = {}".format(block_type))
    if block_type not in valid_block_types:
        raise ValueError("Incompatible argument unet_blocks. \
            Must be either 'default', 'multires' or 'dilated_multires'.")

    input_dims = (img_size, img_size, 3)
    inputs = Input(shape=input_dims)
    
    # Encoder
    eb1 = get_unet_conv_block(inputs, n_filters=nf, dropout_prob=dp, max_pooling=True, block_type=block_type)
    
    nf *= 2
    dp += 0.1

    eb2 = get_unet_conv_block(eb1[0], n_filters=nf, dropout_prob=dp, max_pooling=True, block_type=block_type)
    
    nf *= 2
    dp += 0.1

    eb3 = get_unet_conv_block(eb2[0], n_filters=nf, dropout_prob=dp, max_pooling=True, block_type=block_type)
    
    nf *= 2
    dp += 0.1

    eb4 = get_unet_conv_block(eb3[0], n_filters=nf, dropout_prob=dp, max_pooling=True, block_type=block_type)
    
    nf *= 2
    dp += 0.1
    
    eb5 = get_unet_conv_block(eb4[0], n_filters=nf, dropout_prob=dp, max_pooling=False, block_type=block_type)
    
    nf /= 2

    # Decoder
    skip_tensor_4 = get_unet_connection(inputs=eb4[1], n_filters=nf, path_type=skip_connection_type)
    db6 = get_unet_deconv_block(eb5[0], skip_tensor_4, n_filters=nf, block_type=block_type)
    
    nf /= 2

    skip_tensor_3 = get_unet_connection(inputs=eb3[1], n_filters=nf, path_type=skip_connection_type)
    db7 = get_unet_deconv_block(db6, skip_tensor_3, n_filters=nf, block_type=block_type)
    
    nf /= 2

    skip_tensor_2 = get_unet_connection(inputs=eb2[1], n_filters=nf, path_type=skip_connection_type)
    db8 = get_unet_deconv_block(db7, skip_tensor_2, n_filters=nf, block_type=block_type)
    
    nf /= 2

    skip_tensor_1 = get_unet_connection(inputs=eb1[1], n_filters=nf, path_type=skip_connection_type)
    db9 = get_unet_deconv_block(db8, skip_tensor_1, n_filters=nf, block_type=block_type)
    
    # Output Conv layers and flatten
    conv_out1 = Conv2D(nf, 3, activation='relu', padding='same', kernel_initializer='he_normal')(db9)
    conv_out2 = Conv2D(1, 1, activation='sigmoid')(conv_out1)
    flattened = Flatten()(conv_out2)

    model = Model(inputs=inputs, outputs=flattened)

    learning_rate_schedule = ExponentialDecay(
        initial_learning_rate=lr,
        decay_rate=0.9,
        decay_steps=20000
    )

    model.compile(
        optimizer=Adam(learning_rate=learning_rate_schedule), 
        loss='binary_crossentropy', 
        metrics=['accuracy', 'mean_absolute_error', mean_percent_count_err]
    )
    model.summary()

    # Giving the reference of our custom metric at the loading time 
    model_config = model.get_config()

    return model

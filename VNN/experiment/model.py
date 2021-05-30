"""Models for experimenting"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from network import vlayers
from network import vlayers_conv

def get_scalar_model(dataset_shapes, hidden_layer_units=(2,), activation='relu', output_activation=None, \
                     kernel_initializer='random_normal', bias_initializer='random_normal', \
                     optimizer=keras.optimizers.RMSprop(), loss=keras.losses.MeanSquaredError(), metrics=[keras.metrics.MeanSquaredError()]):
    """Scalar network, standard tensorflow implementation"""
    if output_activation is None:
        output_activation = activation
    
    input_dims = dataset_shapes[0]
    output_dims = dataset_shapes[1]
    
    # Create model
    inputs = keras.Input(shape=input_dims)
    x = inputs
    for h in hidden_layer_units:
        x = layers.Dense(h, activation=activation, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)(x)
    outputs = layers.Dense(output_dims[-1], activation=output_activation, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics
    )
    
    return model

def get_vector_model(dataset_shapes, fractal_depth=1, hidden_layer_units=(2,), inner_hidden_layer_units=(2,), \
                     activation='relu', output_activation=None, \
                     weight_type="unique", weight_initializer='random_normal', \
                     optimizer=keras.optimizers.RMSprop(), loss=keras.losses.MeanSquaredError(), metrics=[keras.metrics.MeanSquaredError()]):
    """Vector network"""
    if output_activation is None:
        output_activation = activation
    
    input_dims = dataset_shapes[0]
    output_dims = dataset_shapes[1]
    
    # Create model
    inputs = keras.Input(shape=input_dims)
    x = vlayers.VInput(hidden_layer_units[0] if len(hidden_layer_units) > 0 else output_dims[-1], weight_initializer='random_normal')(inputs)
    if len(hidden_layer_units) > 0:
        for h in hidden_layer_units[1:] + (output_dims[-1],):
            if fractal_depth < 1:
                x = vlayers.VDense(h, activation=activation, weight_initializer=weight_initializer)(x)
            else:
                x = vlayers.VFractal(h, depth=fractal_depth, hidden_layer_units=inner_hidden_layer_units, activation=activation, weight_initializer=weight_initializer, weight_type=weight_type)(x)
    outputs = vlayers.VOutput(activation=output_activation)(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics
    )
    
    return model

def get_scalar_conv_model1(dataset_shapes, optimizer=keras.optimizers.SGD()):
    input_dims = dataset_shapes[0]
    output_dims = dataset_shapes[1]
    
    # Create model
    first_filter = 7
    inputs = keras.Input(shape=input_dims)
    x = layers.Conv2D(output_dims[0], first_filter, 
        activation='relu', 
        strides=(1,1), 
        padding='valid', 
        kernel_initializer='random_normal', 
        bias_initializer='random_normal'
    )(inputs)
    x = layers.ReLU()(x)
    x = layers.MaxPool2D(pool_size=(input_dims[0]-first_filter+1, input_dims[1]-first_filter+1), strides=(1,1), padding='valid')(x)
    x = layers.Activation('sigmoid')(x)
    outputs = layers.Flatten()(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=optimizer,
        loss=keras.losses.MeanSquaredError(),
        metrics=[keras.metrics.MeanSquaredError()]
    )
    
    return model

def get_vector_conv_model1(dataset_shapes, shared_inner_nets, optimizer=keras.optimizers.SGD()):
    input_dims = dataset_shapes[0]
    output_dims = dataset_shapes[1]
    
    # Create model
    first_filter = 7
    inputs = keras.Input(shape=input_dims)
    x = vlayers_conv.VInputConv((first_filter,first_filter), 
        num_filters=output_dims[0], 
        kernel_type="convolution",
        strides=(1,1), 
        padding_type='valid'
    )(inputs)
    x = vlayers_conv.VConvFractal((input_dims[0]-first_filter+1, input_dims[1]-first_filter+1), 
        kernel_type="pooling",
        strides=(1,1), 
        padding_type='valid',
        layer_type="convolution",
        activation="relu", 
        depth=1, 
        shared_inner_nets=shared_inner_nets, 
        hidden_layer_units=(2,)
    )(x)
    x = vlayers_conv.VOutputConv(layer_type="pooling", pooling="max")(x)
    outputs = layers.Flatten()(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=optimizer,
        loss=keras.losses.MeanSquaredError(),
        metrics=[keras.metrics.MeanSquaredError()]
    )
    
    return model

def get_scalar_conv_model2(dataset_shapes, optimizer=keras.optimizers.SGD()):
    input_dims = dataset_shapes[0]
    output_dims = dataset_shapes[1]
    
    # Create model
    first_filter = 3
    second_filter = 5
    third_filter = input_dims[0] - first_filter - second_filter + 2
    
    first_filter_num = 5
    
    inputs = keras.Input(shape=input_dims)
    x = layers.Conv2D(first_filter_num, first_filter, 
        activation='relu', 
        strides=(1,1), 
        padding='valid', 
        kernel_initializer='random_normal', 
        bias_initializer='random_normal'
    )(inputs)
    x = layers.ReLU()(x)
    x = layers.Conv2D(output_dims[0], second_filter, 
        activation='relu', 
        strides=(1,1), 
        padding='valid', 
        kernel_initializer='random_normal', 
        bias_initializer='random_normal'
    )(x)
    x = layers.ReLU()(x)
    x = layers.MaxPool2D(pool_size=(third_filter,third_filter), strides=(1,1), padding='valid')(x)
    x = layers.Activation('sigmoid')(x)
    outputs = layers.Flatten()(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=optimizer,
        loss=keras.losses.MeanSquaredError(),
        metrics=[keras.metrics.MeanSquaredError()]
    )
    
    return model
    
def get_vector_conv_model2(dataset_shapes, shared_inner_nets, optimizer=keras.optimizers.SGD()):
    input_dims = dataset_shapes[0]
    output_dims = dataset_shapes[1]
    
    # Create model
    first_filter = 3
    second_filter = 5
    third_filter = input_dims[0] - first_filter - second_filter + 2
    
    first_filter_num = 5
    
    inputs = keras.Input(shape=input_dims)
    x = vlayers_conv.VInputConv((first_filter,first_filter), 
        num_filters=first_filter_num, 
        kernel_type="convolution",
        strides=(1,1), 
        padding_type='valid'
    )(inputs)
    x = vlayers_conv.VConvFractal((second_filter,second_filter), 
        kernel_type="convolution",
        num_filters=output_dims[0], 
        strides=(1,1), 
        padding_type='valid',
        layer_type="convolution",
        activation="relu", 
        depth=1, 
        shared_inner_nets=shared_inner_nets, 
        hidden_layer_units=(2,)
    )(x)
    x = vlayers_conv.VConv((third_filter,third_filter), 
        kernel_type="pooling",
        strides=(1,1), 
        padding_type='valid',
        layer_type="convolution",
        activation="relu"
    )(x)
    x = vlayers_conv.VOutputConv(layer_type="pooling", pooling="max")(x)
    outputs = layers.Flatten()(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=optimizer,
        loss=keras.losses.MeanSquaredError(),
        metrics=[keras.metrics.MeanSquaredError()]
    )
    
    return model
    
def get_scalar_conv_model3(dataset_shapes, optimizer=keras.optimizers.SGD()):
    input_dims = dataset_shapes[0]
    output_dims = dataset_shapes[1]
    
    # Settings
    first_filter_num = 5
    first_filter_dim = 3
    second_filter_num = output_dims[0]
    second_filter_dim = ((input_dims[0] // 4) - (first_filter_dim - 1)) // 2
    
    inputs = keras.Input(shape=input_dims)
    # Decrease input twice
    x = layers.AveragePooling2D(pool_size=(4,4), strides=(4,4), padding='valid')(inputs)
    # First convolutional layer
    x = layers.Conv2D(first_filter_num, first_filter_dim, 
        activation='relu', 
        strides=(1,1), 
        padding='valid', 
        kernel_initializer='random_normal', 
        bias_initializer='random_normal'
    )(x)
    x = layers.ReLU()(x)
    # First pooling layer
    x = layers.MaxPool2D(pool_size=(2,2), strides=(2,2), padding='valid')(x)
    # Second convolutional layer
    x = layers.Conv2D(second_filter_num, second_filter_dim, 
        activation='relu', 
        strides=(1,1), 
        padding='valid', 
        kernel_initializer='random_normal', 
        bias_initializer='random_normal'
    )(x)
    x = layers.Activation('softmax')(x)
    outputs = layers.Flatten()(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=optimizer,
        loss=keras.losses.MeanSquaredError(),
        metrics=[keras.metrics.MeanSquaredError()]
    )
    
    return model

def get_vector_conv_model3(dataset_shapes, shared_inner_nets, optimizer=keras.optimizers.SGD()):
    input_dims = dataset_shapes[0]
    output_dims = dataset_shapes[1]
    
    # Settings
    first_filter_num = 5
    first_filter_dim = 3
    second_filter_num = output_dims[0]
    second_filter_dim = ((input_dims[0] // 4) - (first_filter_dim - 1)) // 2
    
    inputs = keras.Input(shape=input_dims)
    # Decrease input
    x = layers.AveragePooling2D(pool_size=(4,4), strides=(4,4), padding='valid')(inputs)
    # First convolutional layer
    x = vlayers_conv.VInputConv((first_filter_dim,first_filter_dim), 
        num_filters=first_filter_num, 
        kernel_type="convolution",
        strides=(1,1), 
        padding_type='valid'
    )(x)
    x = vlayers_conv.VConvFractal((2,2), 
        kernel_type="pooling",
        strides=(2,2), 
        padding_type='valid',
        layer_type="convolution",
        weight_initializer="random_normal",
        activation="relu", 
        depth=1, 
        shared_inner_nets=shared_inner_nets, 
        hidden_layer_units=(2,)
    )(x)
    # First pooling layer
    x = vlayers_conv.VOutputConv(layer_type="pooling", pooling="max")(x)
    # Second convolutional layer
    x = layers.Conv2D(second_filter_num, second_filter_dim, 
        activation='relu', 
        strides=(1,1), 
        padding='valid', 
        kernel_initializer='random_normal', 
        bias_initializer='random_normal'
    )(x)
    x = layers.Activation('softmax')(x)
    outputs = layers.Flatten()(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=optimizer,
        loss=keras.losses.MeanSquaredError(),
        metrics=[keras.metrics.MeanSquaredError()]
    )
    
    return model

def get_scalar_conv_model4(dataset_shapes, optimizer=keras.optimizers.SGD()):
    input_dims = dataset_shapes[0]
    output_dims = dataset_shapes[1]
    
    # Settings
    first_filter_num = 5
    first_filter_dim = 3
    second_filter_num = output_dims[0]
    second_filter_dim = 3
    third_filter_dim = (input_dims[0] // 4) - (first_filter_dim - 1) - (second_filter_dim - 1)
    
    inputs = keras.Input(shape=input_dims)
    # Decrease input twice
    x = layers.AveragePooling2D(pool_size=(4,4), strides=(4,4), padding='valid')(inputs)
    # First convolutional layer
    x = layers.Conv2D(first_filter_num, first_filter_dim, 
        activation='relu', 
        strides=(1,1), 
        padding='valid', 
        kernel_initializer='random_normal', 
        bias_initializer='random_normal'
    )(x)
    x = layers.ReLU()(x)
    # Second convolutional layer
    x = layers.Conv2D(second_filter_num, second_filter_dim, 
        activation='relu', 
        strides=(1,1), 
        padding='valid', 
        kernel_initializer='random_normal', 
        bias_initializer='random_normal'
    )(x)
    x = layers.ReLU()(x)
    # First pooling layer
    x = layers.MaxPool2D(pool_size=(third_filter_dim,third_filter_dim), strides=(1,1), padding='valid')(x)
    x = layers.Activation('softmax')(x)
    outputs = layers.Flatten()(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=optimizer,
        loss=keras.losses.MeanSquaredError(),
        metrics=[keras.metrics.MeanSquaredError()]
    )
    
    return model

def get_vector_conv_model4(dataset_shapes, shared_inner_nets, optimizer=keras.optimizers.SGD()):
    input_dims = dataset_shapes[0]
    output_dims = dataset_shapes[1]
    
    # Settings
    first_filter_num = 5
    first_filter_dim = 3
    second_filter_num = output_dims[0]
    second_filter_dim = 3
    third_filter_dim = (input_dims[0] // 4) - (first_filter_dim - 1) - (second_filter_dim - 1)
    
    inputs = keras.Input(shape=input_dims)
    # Decrease input
    x = layers.AveragePooling2D(pool_size=(4,4), strides=(4,4), padding='valid')(inputs)
    # First convolutional layer
    x = vlayers_conv.VInputConv((first_filter_dim,first_filter_dim), 
        num_filters=first_filter_num, 
        kernel_type="convolution",
        strides=(1,1), 
        padding_type='valid'
    )(x)
    x = vlayers_conv.VConvFractal((second_filter_dim,second_filter_dim), 
        num_filters=second_filter_num, 
        kernel_type="convolution",
        strides=(1,1), 
        padding_type='valid',
        layer_type="convolution",
        weight_initializer="random_normal",
        activation="relu", 
        depth=1, 
        shared_inner_nets=shared_inner_nets, 
        hidden_layer_units=(2,)
    )(x)
    # Second convolutional layer
    x = vlayers_conv.VOutputConv(layer_type="convolution", activation="relu")(x)
    # First pooling layer
    x = layers.MaxPool2D(pool_size=(third_filter_dim,third_filter_dim), strides=(1,1), padding='valid')(x)
    x = layers.Activation('softmax')(x)
    outputs = layers.Flatten()(x)
    x = layers.Activation('softmax')(x)
    outputs = layers.Flatten()(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=optimizer,
        loss=keras.losses.MeanSquaredError(),
        metrics=[keras.metrics.MeanSquaredError()]
    )
    
    return model

def get_le_net_5(dataset_shapes, optimizer=keras.optimizers.SGD()):
    """Modified and simplified LeNet-5
        ReLU layers after each convolution layer are added
        Subsampling layers are replaced with MaxPool
        C3 is simple conv layer
        In fully connected layers sigmoid is replaced with ReLU, 
        Gaussian connection is replaced with softmax
    Unmodified model's source: Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. 
        Gradient-based learning applied to document recognition. Proceedings of the IEEE, 86(11), pp. 2278–2324, 1998."""
    input_dims = dataset_shapes[0] # Initially 32x32x1
    output_dims = dataset_shapes[1] # Initially 10
    
    inputs = keras.Input(shape=input_dims)
    x = inputs
    # Pad input
    if input_dims[0] < 32:
        padding = (32 - input_dims[0]) // 2
        x = layers.ZeroPadding2D(padding=padding)(x)
    # C1
    x = layers.Conv2D(6, 5, 
        activation='relu', 
        strides=(1,1), 
        padding='valid', 
        kernel_initializer='random_normal', 
        bias_initializer='random_normal'
    )(x)
    x = layers.ReLU()(x)
    # S2
    x = layers.MaxPool2D(pool_size=(2,2), strides=(2,2), padding='valid')(x)
    # C3
    x = layers.Conv2D(16, 5, 
        activation='relu', 
        strides=(1,1), 
        padding='valid', 
        kernel_initializer='random_normal', 
        bias_initializer='random_normal'
    )(x)
    # S4
    x = layers.MaxPool2D(pool_size=(2,2), strides=(2,2), padding='valid')(x)
    # C5
    x = layers.Conv2D(120, 5, 
        activation='relu', 
        strides=(1,1), 
        padding='valid', 
        kernel_initializer='random_normal', 
        bias_initializer='random_normal'
    )(x)
    x = layers.Flatten()(x)
    # F6
    x = layers.Dense(84, activation='relu', kernel_initializer='random_normal', bias_initializer='random_normal')(x)
    # Output
    outputs = layers.Dense(output_dims[0], activation='softmax', kernel_initializer='random_normal', bias_initializer='random_normal')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=optimizer,
        loss=keras.losses.MeanSquaredError(),
        metrics=[keras.metrics.MeanSquaredError()]
    )
    
    return model
    
def get_le_net_5_without_fully_connected(dataset_shapes, optimizer=keras.optimizers.SGD()):
    input_dims = dataset_shapes[0] # Initially 32x32x1
    output_dims = dataset_shapes[1] # Initially 10
    
    inputs = keras.Input(shape=input_dims)
    x = inputs
    # Pad input
    if input_dims[0] < 32:
        padding = (32 - input_dims[0]) // 2
        x = layers.ZeroPadding2D(padding=padding)(x)
    # C1
    x = layers.Conv2D(6, 5, 
        activation='relu', 
        strides=(1,1), 
        padding='valid', 
        kernel_initializer='random_normal', 
        bias_initializer='random_normal'
    )(x)
    x = layers.ReLU()(x)
    # S2
    x = layers.MaxPool2D(pool_size=(2,2), strides=(2,2), padding='valid')(x)
    # C3
    x = layers.Conv2D(16, 5, 
        activation='relu', 
        strides=(1,1), 
        padding='valid', 
        kernel_initializer='random_normal', 
        bias_initializer='random_normal'
    )(x)
    # S4
    x = layers.MaxPool2D(pool_size=(2,2), strides=(2,2), padding='valid')(x)
    # C5
    x = layers.Conv2D(output_dims[0], 5, 
        activation='softmax', 
        strides=(1,1), 
        padding='valid', 
        kernel_initializer='random_normal', 
        bias_initializer='random_normal'
    )(x)
    outputs = layers.Flatten()(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=optimizer,
        loss=keras.losses.MeanSquaredError(),
        metrics=[keras.metrics.MeanSquaredError()]
    )
    
    return model

def get_le_net_5_fractal1(dataset_shapes, shared_inner_nets, optimizer=keras.optimizers.SGD()):
    input_dims = dataset_shapes[0] # Initially 32x32x1
    output_dims = dataset_shapes[1] # Initially 10
    
    inputs = keras.Input(shape=input_dims)
    x = inputs
    # Pad input
    if input_dims[0] < 32:
        padding = (32 - input_dims[0]) // 2
        x = layers.ZeroPadding2D(padding=padding)(x)
    # C1
    # x = layers.Conv2D(6, 5, 
    #     activation='relu', 
    #     strides=(1,1), 
    #     padding='valid', 
    #     kernel_initializer='random_normal', 
    #     bias_initializer='random_normal'
    # )(x)
    # x = layers.ReLU()(x)
    x = vlayers_conv.VInputConv((5,5), 
        num_filters=6, 
        kernel_type="convolution",
        strides=(1,1), 
        padding_type='valid'
    )(x)
    x = vlayers_conv.VConvFractal((2,2), 
        kernel_type="pooling",
        strides=(2,2), 
        padding_type='valid',
        layer_type="convolution",
        weight_initializer="random_normal",
        activation="relu", 
        depth=1, 
        shared_inner_nets=shared_inner_nets, 
        hidden_layer_units=(2,)
    )(x)
    # S2
    # x = layers.MaxPool2D(pool_size=(2,2), strides=(2,2), padding='valid')(x)
    x = vlayers_conv.VOutputConv(layer_type="pooling", pooling="max")(x)
    # C3
    x = layers.Conv2D(16, 5, 
        activation='relu', 
        strides=(1,1), 
        padding='valid', 
        kernel_initializer='random_normal', 
        bias_initializer='random_normal'
    )(x)
    # S4
    x = layers.MaxPool2D(pool_size=(2,2), strides=(2,2), padding='valid')(x)
    # C5
    x = layers.Conv2D(output_dims[0], 5, 
        activation='softmax', 
        strides=(1,1), 
        padding='valid', 
        kernel_initializer='random_normal', 
        bias_initializer='random_normal'
    )(x)
    outputs = layers.Flatten()(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=optimizer,
        loss=keras.losses.MeanSquaredError(),
        metrics=[keras.metrics.MeanSquaredError()]
    )
    
    return model
    
def get_le_net_5_fractal2(dataset_shapes, shared_inner_nets, optimizer=keras.optimizers.SGD()):
    input_dims = dataset_shapes[0] # Initially 32x32x1
    output_dims = dataset_shapes[1] # Initially 10
    
    inputs = keras.Input(shape=input_dims)
    x = inputs
    # Pad input
    if input_dims[0] < 32:
        padding = (32 - input_dims[0]) // 2
        x = layers.ZeroPadding2D(padding=padding)(x)
    # C1
    x = layers.Conv2D(6, 5, 
        activation='relu', 
        strides=(1,1), 
        padding='valid', 
        kernel_initializer='random_normal', 
        bias_initializer='random_normal'
    )(x)
    x = layers.ReLU()(x)
    # S2
    x = layers.MaxPool2D(pool_size=(2,2), strides=(2,2), padding='valid')(x)
    # C3
    # x = layers.Conv2D(16, 5, 
    #     activation='relu', 
    #     strides=(1,1), 
    #     padding='valid', 
    #     kernel_initializer='random_normal', 
    #     bias_initializer='random_normal'
    # )(x)
    x = vlayers_conv.VInputConv((5,5), 
        num_filters=16, 
        kernel_type="convolution",
        strides=(1,1), 
        padding_type='valid'
    )(x)
    x = vlayers_conv.VConvFractal((2,2), 
        kernel_type="pooling",
        strides=(2,2), 
        padding_type='valid',
        layer_type="convolution",
        weight_initializer="random_normal",
        activation="relu", 
        depth=1, 
        shared_inner_nets=shared_inner_nets, 
        hidden_layer_units=(2,)
    )(x)
    # S4
    # x = layers.MaxPool2D(pool_size=(2,2), strides=(2,2), padding='valid')(x)
    x = vlayers_conv.VOutputConv(layer_type="pooling", pooling="max")(x)
    # C5
    x = layers.Conv2D(output_dims[0], 5, 
        activation='softmax', 
        strides=(1,1), 
        padding='valid', 
        kernel_initializer='random_normal', 
        bias_initializer='random_normal'
    )(x)
    outputs = layers.Flatten()(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=optimizer,
        loss=keras.losses.MeanSquaredError(),
        metrics=[keras.metrics.MeanSquaredError()]
    )
    
    return model
    
def get_le_net_5_fractal3(dataset_shapes, shared_inner_nets, optimizer=keras.optimizers.SGD()):
    input_dims = dataset_shapes[0] # Initially 32x32x1
    output_dims = dataset_shapes[1] # Initially 10
    
    inputs = keras.Input(shape=input_dims)
    x = inputs
    # Pad input
    if input_dims[0] < 32:
        padding = (32 - input_dims[0]) // 2
        x = layers.ZeroPadding2D(padding=padding)(x)
    x = vlayers_conv.VInputConv((5,5), 
        num_filters=6, 
        kernel_type="convolution",
        strides=(1,1), 
        padding_type='valid'
    )(x)
    x = vlayers_conv.VConvFractal((2,2), 
        kernel_type="pooling",
        strides=(2,2), 
        padding_type='valid',
        layer_type="convolution",
        weight_initializer="random_normal",
        activation="relu", 
        depth=1, 
        shared_inner_nets=shared_inner_nets, 
        hidden_layer_units=(2,)
    )(x)
    # S2
    x = vlayers_conv.VOutputConv(layer_type="pooling", pooling="max")(x)
    # C5
    x = layers.Conv2D(output_dims[0], 14, 
        activation='softmax', 
        strides=(1,1), 
        padding='valid', 
        kernel_initializer='random_normal', 
        bias_initializer='random_normal'
    )(x)
    outputs = layers.Flatten()(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=optimizer,
        loss=keras.losses.MeanSquaredError(),
        metrics=[keras.metrics.MeanSquaredError()]
    )
    
    return model
    
def get_le_net_5_fractal4(dataset_shapes, shared_inner_nets, optimizer=keras.optimizers.SGD()):
    input_dims = dataset_shapes[0] # Initially 32x32x1
    output_dims = dataset_shapes[1] # Initially 10
    
    inputs = keras.Input(shape=input_dims)
    x = inputs
    # Pad input
    if input_dims[0] < 32:
        padding = (32 - input_dims[0]) // 2
        x = layers.ZeroPadding2D(padding=padding)(x)
    # C1
    x = vlayers_conv.VInputConv((5,5), 
        num_filters=1, 
        kernel_type="convolution",
        strides=(1,1), 
        padding_type='valid'
    )(x)
    x = vlayers_conv.VConvFractal((2,2), 
        kernel_type="pooling",
        strides=(2,2), 
        padding_type='valid',
        layer_type="convolution",
        weight_initializer="random_normal",
        activation="relu", 
        depth=1, 
        shared_inner_nets=shared_inner_nets, 
        hidden_layer_units=(2,)
    )(x)
    # S2
    x = vlayers_conv.VOutputConv(layer_type="pooling", pooling="max")(x)
    # C3
    x = layers.Conv2D(16, 5, 
        activation='relu', 
        strides=(1,1), 
        padding='valid', 
        kernel_initializer='random_normal', 
        bias_initializer='random_normal'
    )(x)
    # S4
    x = layers.MaxPool2D(pool_size=(2,2), strides=(2,2), padding='valid')(x)
    # C5
    x = layers.Conv2D(output_dims[0], 5, 
        activation='softmax', 
        strides=(1,1), 
        padding='valid', 
        kernel_initializer='random_normal', 
        bias_initializer='random_normal'
    )(x)
    outputs = layers.Flatten()(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=optimizer,
        loss=keras.losses.MeanSquaredError(),
        metrics=[keras.metrics.MeanSquaredError()]
    )
    
    return model
    
def get_le_net_5_fractal5(dataset_shapes, shared_inner_nets, hidden_layer_units=(2,), activation="relu", optimizer=keras.optimizers.SGD()):
    input_dims = dataset_shapes[0] # Initially 32x32x1
    output_dims = dataset_shapes[1] # Initially 10
    
    inputs = keras.Input(shape=input_dims)
    x = inputs
    # Pad input
    if input_dims[0] < 32:
        padding = (32 - input_dims[0]) // 2
        x = layers.ZeroPadding2D(padding=padding)(x)
    # C1
    x = vlayers_conv.VInputConv((5,5), 
        num_filters=6, 
        kernel_type="convolution",
        strides=(1,1), 
        padding_type='valid'
    )(x)
    x = vlayers_conv.VConvFractal((2,2), 
        kernel_type="pooling",
        strides=(2,2), 
        padding_type='valid',
        layer_type="convolution",
        weight_initializer="random_normal",
        activation=activation, 
        depth=1, 
        shared_inner_nets=shared_inner_nets, 
        hidden_layer_units=hidden_layer_units
    )(x)
    # S2
    x = vlayers_conv.VConv((5,5), 
        layer_type="pooling", pooling="max",
        num_filters=16, 
        kernel_type="convolution",
        strides=(1,1), 
        padding_type='valid', 
        weight_initializer="random_normal"
    )(x)
    # C3
    x = vlayers_conv.VConvFractal((2,2), 
        kernel_type="pooling",
        strides=(2,2), 
        padding_type='valid',
        layer_type="convolution",
        weight_initializer="random_normal",
        activation=activation, 
        depth=1, 
        shared_inner_nets=shared_inner_nets, 
        hidden_layer_units=hidden_layer_units
    )(x)
    # S4
    # x = vlayers_conv.VOutputConv(layer_type="pooling", pooling="max")(x)
    x = vlayers_conv.VConv((5,5), 
        layer_type="pooling", pooling="max",
        num_filters=output_dims[0], 
        kernel_type="convolution",
        strides=(1,1), 
        padding_type='valid', 
        weight_initializer="random_normal"
    )(x)
    # C5
    x = vlayers_conv.VOutputConv(layer_type="convolution", activation='softmax')(x)
    outputs = layers.Flatten()(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=optimizer,
        loss=keras.losses.MeanSquaredError(),
        metrics=[keras.metrics.MeanSquaredError()]
    )
    
    return model
    
def get_le_net_5_fractal6(dataset_shapes, shared_inner_nets, hidden_layer_units=(2,), activation="relu", optimizer=keras.optimizers.SGD()):
    input_dims = dataset_shapes[0] # Initially 32x32x1
    output_dims = dataset_shapes[1] # Initially 10
    
    inputs = keras.Input(shape=input_dims)
    x = inputs
    # Pad input
    if input_dims[0] < 32:
        padding = (32 - input_dims[0]) // 2
        x = layers.ZeroPadding2D(padding=padding)(x)
    # C1
    x = vlayers_conv.VInputConv((5,5), 
        num_filters=6, 
        kernel_type="convolution",
        strides=(1,1), 
        padding_type='valid'
    )(x)
    x = vlayers_conv.VConvFractal((2,2), 
        kernel_type="convolution",
        num_filters = 6,
        strides=(2,2), 
        padding_type='valid',
        layer_type="convolution",
        weight_initializer="random_normal",
        activation=activation, 
        depth=1, 
        shared_inner_nets=shared_inner_nets, 
        hidden_layer_units=hidden_layer_units
    )(x)
    # S2
    x = vlayers_conv.VConv((5,5), 
        layer_type="convolution", activation=activation, 
        num_filters=16, 
        kernel_type="convolution",
        strides=(1,1), 
        padding_type='valid', 
        weight_initializer="random_normal"
    )(x)
    # C3
    x = vlayers_conv.VConvFractal((2,2), 
        kernel_type="convolution",
        num_filters = 16,
        strides=(2,2), 
        padding_type='valid',
        layer_type="convolution",
        weight_initializer="random_normal",
        activation=activation, 
        depth=1, 
        shared_inner_nets=shared_inner_nets, 
        hidden_layer_units=hidden_layer_units
    )(x)
    # S4
    x = vlayers_conv.VConv((5,5), 
        layer_type="convolution", activation=activation, 
        num_filters=output_dims[0], 
        kernel_type="convolution",
        strides=(1,1), 
        padding_type='valid', 
        weight_initializer="random_normal"
    )(x)
    # C5
    x = vlayers_conv.VOutputConv(layer_type="convolution", activation='softmax')(x)
    outputs = layers.Flatten()(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=optimizer,
        loss=keras.losses.MeanSquaredError(),
        metrics=[keras.metrics.MeanSquaredError()]
    )
    
    return model
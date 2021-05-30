import functools
import numpy as np
import operator
import tensorflow as tf
from tensorflow.keras import layers
import tensorflow.sparse as sparse

from .vlayers import concat_biases, VInput, VFractal, VOutput

# Functions
def flatten(inputs, dims_to_flatten, dims_to_save=0):
    """Flatten given dimensions of tensor"""
    input_shape = inputs.shape
    rank = len(input_shape)
    batch_dims = input_shape[:rank - dims_to_flatten - dims_to_save]
    
    if dims_to_save > 0:
        saved_dims = input_shape[-dims_to_save:]
        non_batch_dims = input_shape[-(dims_to_flatten + dims_to_save):-dims_to_save]
    else:
        non_batch_dims = input_shape[-dims_to_flatten:]
    
    if tf.executing_eagerly():
        # Full static shape is guaranteed to be available.
        # Performance: Using `constant_op` is much faster than passing a list.
        if dims_to_save:
            flattened_shape = tf.concat([batch_dims, [-1], saved_dims], 0)
        else:
            flattened_shape = tf.concat([batch_dims, [-1]], 0)
        return tf.reshape(inputs, flattened_shape)
    else:
        last_dim = int(functools.reduce(operator.mul, non_batch_dims))
        if dims_to_save:
            flattened_shape = tf.concat([[-1], batch_dims[1:], [last_dim], saved_dims], 0)
        else:
            flattened_shape = tf.concat([[-1], batch_dims[1:], [last_dim]], 0)
        return tf.reshape(inputs, flattened_shape)


def get_input_shape(input_shape, paddings):
    """Get shape of input feature tensor"""
    input_shape = tf.constant(input_shape[-3:])
    if paddings is not None:
        paddings = tf.reduce_sum(paddings[-3:], axis=-1)
        input_shape += paddings
    return input_shape


def get_full_output_shape(input_shape, kernel_shape, strides, paddings, use_bias):
    """Get shape of output tensor"""
    input_shape = get_input_shape(input_shape, paddings)
    vector_dim = tf.reduce_prod(kernel_shape[:-1])
    if use_bias:
        vector_dim += 1
    
    filters_num = kernel_shape[-1]
    
    input_shape = tf.constant(input_shape[-3:])
    kernel_shape = tf.constant(kernel_shape[-4:-1])
    strides = tf.concat([tf.constant(strides), [1]], axis=0)
    
    # Convolution layer output shape formula
    output_shape = ((input_shape - kernel_shape) // strides) + 1
    # Add filters 
    output_shape *= tf.concat([1, 1, filters_num], axis=0)
    # Add vector dimension
    output_shape = tf.concat([[-1, vector_dim], output_shape], axis=0)
    return output_shape


def get_output_shape_for_single_filter(input_shape, kernel_shape, strides):
    """Get shape of output feature tensor for single filter"""
    input_shape = tf.constant(input_shape[-3:])
    kernel_shape = tf.constant(kernel_shape[-4:-1])
    strides = tf.concat([tf.constant(strides), [1]], axis=0)
    # Convolution layer output shape formula
    output_shape = ((input_shape - kernel_shape) // strides) + 1
    return output_shape


def get_input_gather_indices(input_shape, kernel_shape, strides, paddings, use_bias, is_vector_input):
    """Iterate over indices of elements (to extract from input/activization tensor) for furhter multiplication by weights"""
    # Padded input shape
    input_shape = get_input_shape(input_shape, paddings)
    # Output shape
    output_shape = get_output_shape_for_single_filter(input_shape, kernel_shape, strides)
    output_flat_len = np.prod(output_shape)
    
    # Compute all offsets
    row_offsets = np.arange(output_shape[-3]) * input_shape[-2].numpy() * strides[-2] * input_shape[-1].numpy()
    col_offsets = np.arange(output_shape[-2]) * strides[-1] * input_shape[-1].numpy()
    chan_offsets = np.arange(output_shape[-1])
    offsets = row_offsets[:, np.newaxis, np.newaxis] + col_offsets[:, np.newaxis] + chan_offsets
    offsets = offsets.flatten()
    
    # Find gathering indices without offset
    index_bool_vector = np.zeros(input_shape)
    index_bool_vector[:kernel_shape[-4],:kernel_shape[-3],:kernel_shape[-2]] = 1
    index_bool_vector = index_bool_vector.flatten()
    index_vector = np.nonzero(index_bool_vector)[0]
    index_matrix = index_vector[:, np.newaxis]
    index_matrix = np.tile(index_matrix, (1, output_flat_len))
    
    # Find gathering indices with offsets
    offset_index_matrix = index_matrix + offsets
    
    # Append bias
    if use_bias:
        bias_index = np.prod(input_shape)
        bias = np.full((1, offset_index_matrix.shape[-1]), bias_index, dtype=np.int32)
        offset_index_matrix = np.concatenate((offset_index_matrix, bias), axis=0)
    
    # Set vector indices for all inputs
    if is_vector_input:
        row_num = index_bool_vector.shape[0]
        col_num = output_flat_len
        row_indices = np.arange(row_num)[:,np.newaxis]
        col_indices = np.arange(col_num)
        shifts = np.mod(row_indices - offsets, row_num)
        unshifts = np.mod(row_indices + offsets, row_num)
        # Set indices
        uniqueness_indices = index_bool_vector[:, np.newaxis]
        uniqueness_indices = np.tile(uniqueness_indices, (1, col_num))
        # Shift by offsets
        uniqueness_indices = uniqueness_indices[shifts, col_indices]
        # Get unique indices
        uniqueness_indices = np.cumsum(uniqueness_indices, axis=1)
        # Unshift
        uniqueness_indices = uniqueness_indices[unshifts, col_indices]
        # Slice and convert to zero-based indexing
        uniqueness_indices = uniqueness_indices[index_vector] - 1
        # Append bias
        if use_bias:
            bias = np.full((1, uniqueness_indices.shape[-1]), 0, dtype=np.int32)
            uniqueness_indices = np.concatenate((uniqueness_indices, bias), axis=0)
        
        # Set unique vector indices
        offset_index_matrix = np.stack((offset_index_matrix, uniqueness_indices), axis=-1)
    else:
        offset_index_matrix = offset_index_matrix[...,np.newaxis]
    
    return tf.constant(offset_index_matrix, dtype=tf.int64)


def apply_convolution_kernel(x, flattened_weight, weight_tiles, gather_indices, paddings, use_bias, output_shape, is_vector_input):
    """Apply convolution kernel to input/activation"""
    # Pad
    if paddings is not None:
        x = tf.pad(x, paddings)
    # Flatten
    x_flat = flatten(x, 3, 1 if is_vector_input else 0)
    if use_bias:
        x_flat = concat_biases(x_flat, axis=-2 if is_vector_input else -1)
    # Rearrange
    if is_vector_input:
        x = tf.transpose(x_flat, perm=[1,2,0])
    else:
        x = tf.transpose(x_flat, perm=[1,0])
    x = tf.gather_nd(x, gather_indices)
    x = tf.transpose(x, perm=[2,0,1])
    x = tf.expand_dims(x, -1)
    # Fill weight tensor
    weight = tf.tile(flattened_weight, weight_tiles)
    # Multiply 
    y = x * weight
    # Reshape
    return tf.reshape(y, output_shape)


def apply_pooling_kernel(x, gather_indices, paddings, use_bias, output_shape, is_vector_input):
    """Apply pooling (pseudo)kernel to input/activation"""
    # Pad
    if paddings is not None:
        x = tf.pad(x, paddings)
    # Flatten
    x_flat = flatten(x, 3, 1 if is_vector_input else 0)
    if use_bias:
        x_flat = concat_biases(x_flat, axis=-2 if is_vector_input else -1)
    # Rearrange
    if is_vector_input:
        x = tf.transpose(x_flat, perm=[1,2,0])
    else:
        x = tf.transpose(x_flat, perm=[1,0])
    x = tf.gather_nd(x, gather_indices)
    x = tf.transpose(x, perm=[2,0,1])
    x = tf.expand_dims(x, -1)
    # Reshape
    return tf.reshape(x, output_shape)


def do_convolution(x, activation_fun):
    """Do simple convolution over input"""
    # Sum
    s = tf.reduce_sum(x, -4)
    # Compute activation
    a = activation_fun(s)
    
    return a


def do_convolution_with_inner_net(x, inner_input, inner_hiddens, inner_output):
    """Do convolution over input, using inner fractal network"""
    # Set vector input as last dimension
    x = tf.transpose(x, perm=[0,2,3,4,1])
    # Process data as inner network
    y = inner_input(x)
    for hidden in inner_hiddens:
        y = hidden(y)
    a = inner_output(y)
    
    return a


def do_pooling(x, pooling_fun):
    """Do pooling over input"""
    s = pooling_fun(x, -4)
    
    return s


# Classes
class VInputConv(layers.Layer):
    """Input vector layer for convolutional networks"""
    def __init__(self, filter_shape, num_filters=1, kernel_type="convolution", strides=(1,1), padding_type="valid", padding=None, weight_initializer="random_normal"):
        super().__init__()
        self.filter_shape = filter_shape
        self.num_filters = num_filters
        self.kernel_type = kernel_type
        self.strides = strides
        self.padding_type = padding_type
        self.padding = padding
        self.weight_initializer = weight_initializer
        self.is_vector_input = False
    
    def build_inner(self, input_shape, is_vector_input):
        input_rank = input_shape.rank
        
        # Set kernel shape
        if self.kernel_type == "convolution":
            kernel_shape = tf.concat([self.filter_shape, input_shape[-1:], [self.num_filters]], axis=0)
            use_bias = True
            bias_shape = kernel_shape[-1:]
        else: # elif kernel_type == "pooling"
            kernel_shape = tf.concat([self.filter_shape, [1, 1]], axis=0)
            use_bias = False
        
        # Init padding
        if self.padding is not None:
            paddings = tf.constant(padding)
        elif self.padding_type == "same":
            paddings = tf.constant(kernel_shape[-4:-2]) - 1
            paddings_before = paddings // 2
            paddings_after = paddings - paddings_before
            paddings_before = tf.concat([tf.zeros([input_rank - 3], dtype=tf.int32), paddings_before, [0]], axis=0)
            paddings_after = tf.concat([tf.zeros([input_rank - 3], dtype=tf.int32), paddings_after, [0]], axis=0)
            paddings = tf.stack([paddings_before, paddings_after], axis=1)
        elif self.padding_type == "full":
            paddings = tf.constant(kernel_shape[-4:-2]) - 1
            paddings = tf.concat([tf.zeros([input_rank - 3], dtype=tf.int32), paddings, [0]], axis=0)
            paddings = tf.stack([paddings, paddings], axis=1)
        else: # elif self.padding_type == "valid"
            paddings = None
        
        extended_paddings = paddings
        if is_vector_input and extended_paddings is not None:
            extended_paddings = tf.concat([extended_paddings, [[0], [0]]], axis=-1)
        
        
        # Set output shape
        self.full_output_shape = get_full_output_shape(input_shape, kernel_shape, self.strides, paddings, use_bias)
        
        # Get indices to gather elements input/activation tensor into appropriate shape (rearrange)
        gather_indices = get_input_gather_indices(
            input_shape, 
            kernel_shape, 
            self.strides, 
            paddings, 
            use_bias, 
            is_vector_input
        )
        
        if self.kernel_type == "convolution":
            # Init weights and biases (already flattened)
            flattened_weight_shape = tf.concat(
                [
                    tf.reduce_prod(kernel_shape[:-1]) + 1 if use_bias else 0, 
                    1, 
                    kernel_shape[-1]
                ], 
                axis=0
            )
            self.flattened_weight = self.add_weight(
                shape=flattened_weight_shape,
                initializer=self.weight_initializer
            )
            
            # Dimensions to tile weights
            weight_tiles = tf.concat([[1], gather_indices.shape[-2:-1], [1]], axis=0)
            self.weight_multiply_fun = lambda x: apply_convolution_kernel(
                x, self.flattened_weight, weight_tiles, gather_indices, extended_paddings, use_bias, self.full_output_shape, is_vector_input
            )
        else: # elif kernel_type == "pooling"
            self.weight_multiply_fun = lambda x: apply_pooling_kernel(
                x, gather_indices, extended_paddings, use_bias, self.full_output_shape, is_vector_input
            )
    
    def build(self, input_shape):
        self.build_inner(input_shape, False)
    
    def call(self, inputs):
        return self.weight_multiply_fun(inputs)

        
class VConv(VInputConv):
    """Hidden vector convolution layer"""
    def __init__(self, filter_shape, num_filters=1, kernel_type="convolution", strides=(1,1), padding_type="valid", padding=None, weight_initializer="random_normal", layer_type="convolution", activation="relu", pooling="max"):
        super().__init__(filter_shape, num_filters, kernel_type, strides, padding_type, padding, weight_initializer)
        
        self.layer_type = layer_type
        self.activation = activation
        self.pooling = pooling
        self.activation_fun = tf.keras.activations.deserialize(activation)
        if pooling == "max":
            self.pooling_fun = tf.math.reduce_max
        elif pooling == "mean":
            self.pooling_fun = tf.math.reduce_mean
    
    def build(self, input_shape):
        self.build_inner(input_shape, False)
        
        if self.layer_type == "convolution":
            self.op_fun = lambda x: do_convolution(x, self.activation_fun)
        else: # elif self.layer_type == "pooling"
            self.op_fun = lambda x: do_pooling(x, self.pooling_fun)
    
    def call(self, inputs):
        x = self.op_fun(inputs)
        return self.weight_multiply_fun(x)
    

class VConvFractal(VConv):
    """Hidden vector convolution layer with inner networks"""
    def __init__(self, filter_shape, num_filters=1, kernel_type="convolution", strides=(1,1), padding_type="valid", padding=None, weight_initializer="random_normal", layer_type="convolution", activation="relu", pooling="max", depth=1, shared_inner_nets=False, hidden_layer_units=(2,)):
        super().__init__(
            filter_shape, 
            num_filters, 
            kernel_type, 
            strides, 
            padding_type, 
            padding, 
            weight_initializer, 
            layer_type, 
            activation, 
            pooling
        )
        self.depth = depth
        self.shared_inner_nets = shared_inner_nets
        self.hidden_layer_units = hidden_layer_units
    
    def build(self, input_shape):
        self.build_inner(input_shape, True)
        
        # Initialize inner network
        if self.layer_type == "convolution" and self.depth > 0:
            input_item_rank = 1 if self.shared_inner_nets else 2
            output_units = self.full_output_shape[-4]
            
            self.inner_input = VInput(self.hidden_layer_units[0], 
                                      input_item_rank=input_item_rank,
                                      add_bias=False, 
                                      weight_initializer=self.weight_initializer,
                                      weight_type="unique")
            self.inner_hiddens = [VFractal(u, 
                                           input_item_rank = input_item_rank,
                                           depth=self.depth - 1,
                                           hidden_layer_units=self.hidden_layer_units,
                                           activation=self.activation,
                                           weight_initializer=self.weight_initializer,
                                           weight_type="unique") 
                                  for u 
                                  in self.hidden_layer_units[1:] + (output_units,)]
            self.inner_output = VOutput(activation=self.activation)
            
            self.op_fun = lambda x: do_convolution_with_inner_net(x, self.inner_input, self.inner_hiddens, self.inner_output)
    
    def call(self, inputs):
        return super().call(inputs)
        
    
class VOutputConv(layers.Layer):
    """Output vector layer for convolutional networks"""
    def __init__(self, layer_type="convolution", activation="relu", pooling="max"):
        super().__init__()
        self.layer_type = layer_type
        self.activation = activation
        self.pooling = pooling
        self.activation_fun = tf.keras.activations.deserialize(activation)
        if pooling == "max":
            self.pooling_fun = tf.math.reduce_max
        elif pooling == "mean":
            self.pooling_fun = tf.math.reduce_mean
    
    def build(self, input_shape):
        if self.layer_type == "convolution":
            self.op_fun = lambda x: do_convolution(x, self.activation_fun)
        elif self.layer_type == "pooling":
            self.op_fun = lambda x: do_pooling(x, self.pooling_fun)
        else: # self.layer_type == "none":
            self.op_fun = lambda x: do_convolution(x, tf.identity)
    
    def call(self, inputs):
        return self.op_fun(inputs)
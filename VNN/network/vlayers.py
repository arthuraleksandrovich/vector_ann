"""Vector layers"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

# Functions
def concat_biases(inputs, axis=-1):
    """Add bias to each input vector"""
    inputs_rank = len(inputs.shape)
    # Inputs shape can be partially known, so
    # Get inputs slice with current dimension equals one
    slice_begin = tf.zeros(inputs_rank, dtype=tf.int32)
    slice_size = tf.concat([tf.fill([inputs_rank + axis], -1), tf.constant([1]), tf.fill([-axis - 1], -1)], 0)
    inputs_slice = tf.slice(inputs, slice_begin, slice_size)
    # Create biases shaped like inputs slice
    biases = tf.ones_like(inputs_slice, dtype=inputs.dtype)
    # Concatenate inputs with biases
    x = tf.concat([inputs, biases], axis)
    
    return x


def _iterate_rotation_mask_indices(mask_shape):
    for k in range(mask_shape[-3]):
        for i in range(mask_shape[-2]):
            for j in range(mask_shape[-1]):
                if ((i+j-k))%mask_shape[-3] == 0:
                    yield [k, i, j]


def create_rotation_mask(num_coeffs, weight_shape, dtype=tf.float32):
    """Create rotation mask for given number coefficients and weight matrix shape"""
    num_rows = weight_shape[-2]
    num_cols = weight_shape[-1]
    mask_shape = tf.concat([tf.constant(num_coeffs), num_rows, num_cols], axis=0)
    mask_shape = tf.cast(mask_shape, dtype=tf.int64)
    mask_indices = tf.constant(
        list(_iterate_rotation_mask_indices(mask_shape)), 
        dtype=tf.int64
    )
    mask_values = tf.fill(mask_indices.shape[:1], tf.constant(1, dtype=dtype))
    return tf.sparse.to_dense(tf.sparse.SparseTensor(mask_indices, mask_values, mask_shape))


def coeffs_to_weight_gen(coeffs, weight_shape, rotation_mask=None):
    """Create weigth generator that returns shared weight matrix. 
    At the lower depth level weights are coefficient rotations.
    The same rotation matrix is repeated over all units. 
    """
    # Create rotation mask
    if rotation_mask is None:
        rotation_mask = create_rotation_mask(coeffs.shape[-1], weight_shape)
    # Create weight generator
    def weight_gen():
        nonlocal coeffs
        nonlocal weight_shape
        nonlocal rotation_mask
        # Expand dimensions of coefficient matrix to match desired weight shape
        coeffs_shape = tf.concat([tf.ones([weight_shape.shape[0]-2], tf.int32), coeffs.shape, [1, 1]], 0)
        coeffs_reshaped = tf.reshape(coeffs, coeffs_shape)
        # Fill coefficient matrix with repeating tiles to match desired weight shape 
        coeffs_tiles = tf.concat([weight_shape[:-2], [1, 1, 1]], 0)
        coeffs_tiled = tf.tile(coeffs_reshaped, coeffs_tiles)
        # Create rotation matrix
        return tf.reduce_sum(rotation_mask * coeffs_tiled, axis=tf.rank(coeffs_tiled)-3)
    
    return weight_gen


def outer_weights_to_weight_gen(outer_weights, weight_shape, output_units):
    """Create weigth generator that returns shared weight matrix. 
    Output weights of each unit are used as coefficients for lower-depth-level rotation weight matrix.
    """
    # Create rotation mask
    rotation_mask = create_rotation_mask(output_units, weight_shape)
    # Create weight generator
    def weight_gen():
        nonlocal outer_weights
        nonlocal rotation_mask
        # Get outer weights
        outer_weights_inner = outer_weights() if callable(outer_weights) else outer_weights
        # Spread outer weights 
        outer_weights_spread = tf.reshape(outer_weights_inner, tf.concat([outer_weights_inner.shape, (1, 1)], 0))
        # Create rotation matrix
        return tf.reduce_sum(rotation_mask * outer_weights_spread, axis=tf.rank(outer_weights_spread)-3)
    
    return weight_gen


# Classes
class VLayer(layers.Layer):
    """Base vector layer"""
    def __init__(self):
        super().__init__()
    
    @property
    def w(self):
        """Generate weights"""
        return self._get_w()
    
    # Weight initialization
    def _w_init_unique(self, weight_shape, weigth_initializer):
        """Initialize weight matrix with unique values"""
        self._w = self.add_weight(
            shape=weight_shape,
            initializer=weigth_initializer,
            trainable=True
        )
        # Set weights generator
        self._get_w = lambda: self._w


class VInput(VLayer):
    """Input vector layer"""
    def __init__(self, output_units, input_item_rank=1, add_bias=True, weight_initializer="random_normal", weight_type="unique", coeffs=None):
        super().__init__()
        self.output_units = output_units
        self.input_item_rank = input_item_rank
        self.add_bias = add_bias
        self.weight_initializer = weight_initializer
        self.weight_type = weight_type
        self._coeffs = coeffs
    
    def get_weight_shape(self, input_shape):
        """Compute shape of weight matrix"""
        # Get shape of input item
        if self.add_bias:
            # Shape with bias
            shape_without_last = tf.constant(input_shape[-self.input_item_rank:-1], dtype=tf.int32)
            shape_last = tf.constant(input_shape[-1:], dtype=tf.int32) + 1
            input_item_shape = tf.concat([shape_without_last, shape_last], 0)
        else:
            # Shape without bias
            input_item_shape = tf.constant(input_shape[-self.input_item_rank:])
        # Define shape of weights matrix
        weight_shape = tf.concat((input_item_shape, [self.output_units]), 0)
        
        return weight_shape
    
    # Weight initialization
    def _w_init_shared_from_coeffs(self, weight_shape, coeff_initializer=None):
        """Initialize weight matrix as repeating rotations of coefficients"""
        # Set coefficients
        if self._coeffs is None:
            self._coeffs = self.add_weight(
                shape=tf.constant([self.output_units]),
                initializer=coeff_initializer,
                trainable=True
            )
        # Set weights generator
        self._get_w = coeffs_to_weight_gen(self._coeffs, weight_shape)
    
    def build(self, input_shape):
        weight_shape = self.get_weight_shape(input_shape)
        # Initialize weights of inputs and bias
        if self.weight_type == "unique":
            self._w_init_unique(weight_shape, self.weight_initializer)
        if self.weight_type == "shared":
            self._w_init_shared_from_coeffs(weight_shape, self.weight_initializer)
        else: # if self.weight_type == "none":
            pass
    
    def call(self, inputs):
        if self.add_bias:
            # Add biases to inputs
            x = concat_biases(inputs)
        else:
            x = inputs
        # Transform input scalars to one-dimensional vectors
        x = tf.expand_dims(x, -1)
        # Perform scalar-vector multiplication using broadcasting
        return x * self.w


class VDense(VInput):
    """Hidden dense vector layer"""
    def __init__(self, output_units, input_item_rank=1, activation="relu", weight_initializer="random_normal", weight_type="unique", outer_weights=None):
        super().__init__(output_units, input_item_rank=input_item_rank, 
            weight_initializer=weight_initializer,
            weight_type=weight_type,
            add_bias=True
        )
        self.activation = activation
        self.activation_fun = tf.keras.activations.deserialize(activation)
        self._outer_weights = outer_weights
    
    # Weight initialization
    def _w_init_shared_from_outer_weights(self, weight_shape):
        """Initialize weight matrix as rotations of outer-level weights"""
        self._get_w = outer_weights_to_weight_gen(
            lambda: self._outer_weights()[...,:-1,:], 
            weight_shape, 
            self.output_units
        )
    
    def build(self, input_shape):
        # Exclude axis to be reduced 
        # And exlude batch to avoid error
        reduced_shape = tf.concat((input_shape[-(self.input_item_rank+1):-2], input_shape[-1:]), 0)
        weight_shape = self.get_weight_shape(reduced_shape)
        # Initialize weights of inputs and bias
        if self.weight_type == "unique":
            self._w_init_unique(weight_shape, self.weight_initializer)
        if self.weight_type == "shared":
            self._w_init_shared_from_outer_weights(weight_shape)
        else: # if self.weight_type == "none":
            pass
    
    def call(self, inputs):
        # Sum inputs of each unit
        sums = tf.reduce_sum(inputs, axis=-2)
        # Compute activation
        a = self.activation_fun(sums)
        # Add biases to activations
        a_biased = concat_biases(a)
        # Transform activation scalars to one-dimensional vectors
        a_transformed = tf.expand_dims(a_biased, -1)
        # Perform scalar-vector multiplication using broadcasting
        result = a_transformed * self.w
        return a_transformed * self.w


class VFractal(VDense):
    """Hidden vector layer with inner networks"""
    def __init__(self, output_units, input_item_rank=1, depth=1, hidden_layer_units=(2,), activation="relu", weight_initializer="random_normal", weight_type="unique", outer_weights=None, coeffs=None):
        super().__init__(output_units, 
            input_item_rank=input_item_rank,
            activation=activation,
            weight_initializer=weight_initializer,
            weight_type=weight_type,
            outer_weights=outer_weights
        )
        self.depth = depth
        self.hidden_layer_units = hidden_layer_units
        self._coeffs = coeffs
        # Initialize inner layers
        if self.depth > 0:
            if len(self.hidden_layer_units) > 1 and self.weight_type == "shared":
                raise ValueError('Cannot create VFractal layer with weight_type="shared" and more than 1 inner hidden layer!')
            self.inner_input = VInput(self.hidden_layer_units[0], 
                                      input_item_rank=self.input_item_rank + 1,
                                      add_bias=False, 
                                      weight_initializer=self.weight_initializer,
                                      weight_type=self.weight_type)
            self.inner_hiddens = [VFractal(u, 
                                           input_item_rank = self.input_item_rank + 1,
                                           depth=self.depth - 1,
                                           hidden_layer_units=self.hidden_layer_units,
                                           activation=self.activation,
                                           weight_initializer=self.weight_initializer,
                                           weight_type=weight_type) 
                                  for u 
                                  in self.hidden_layer_units[1:] + (output_units,)]
            self.inner_output = VOutput(activation=self.activation)
    
    def build(self, input_shape):
        if self.depth > 0 and self.input_item_rank == 1:
            weight_shape = self.get_weight_shape(input_shape)
            # Initialize weights of inputs and bias
            # Always unique weights
            if self.weight_type == "unique" or self.weight_type == "shared":
                self._w_init_unique(weight_shape, self.weight_initializer)
            else: # if self.weight_type == "none":
                pass
        else:
            super().build(input_shape)
        
        if self.depth > 0 and self.weight_type == "shared": 
            # Set up inner network
            self.inner_input._coeffs = self._coeffs
            self.inner_hiddens[0]._coeffs = self._coeffs
            self.inner_hiddens[0]._outer_weights = self._get_w
    
    def call(self, inputs):
        if self.depth == 0:
            # Process data as dense layer
            return super().call(inputs)
        else:
            # Transpose inputs
            rank = len(inputs.shape)
            permutation = tf.concat([tf.range(rank-2), [rank-1, rank-2]], 0)
            x = tf.transpose(inputs, perm=permutation)
            # Process data as inner network
            y = self.inner_input(x)
            for hidden in self.inner_hiddens:
                y = hidden(y)
            y = self.inner_output(y)
            # Add biases to output
            y_biased = concat_biases(y, axis=-2)
            # Apply weights
            return y_biased * self.w


class VOutput(layers.Layer):
    """Output vector layer"""
    def __init__(self, activation="relu"):
        super().__init__()
        self.activation_fun = tf.keras.activations.deserialize(activation)
    
    def build(self, input_shape):
        # Has no weights
        pass
    
    def call(self, inputs):
        # Sum inputs of each unit
        sums = tf.reduce_sum(inputs, axis=-2)
        # Compute activation
        return self.activation_fun(sums)
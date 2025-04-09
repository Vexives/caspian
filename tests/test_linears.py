from caspian.layers import Layer, Linear, Dense, Bilinear
from caspian.activations import ReLU
import pytest
import numpy as np

def base_layer_tests():
    class TestLayer(Layer):
        def __init__(self, in_size, out_size):
            super().__init__(in_size, out_size)

        def forward(self, data: np.ndarray, training: bool = False):
            return data, training

    # Shape Inheritance
    layer = TestLayer((3,), (5,))
    assert layer.in_size == (3,)
    assert layer.out_size == (5,)

    # Call Inheritance
    data_in = np.zeros((3,))
    forward_result, train = layer(data_in)
    assert forward_result == data_in
    assert train is False

    _, train = layer(data_in, True)
    assert train is True


def linear_tests():
    # Non-tuple sizes
    layer = Linear(3, 5)
    data_in = np.zeros((3,))
    assert layer(data_in).shape == (5,)
    assert layer.out_size == (5,)

    # Tuple sizes
    layer = Linear((10, 3), 5)
    data_in = np.zeros((10, 3))
    assert layer(data_in).shape == (10, 5)
    assert layer.out_size == (10, 5)

    # Allow for other-valued batch sizes
    data_in = np.zeros((11, 3))
    assert layer(data_in).shape == (11, 5)

    layer = Linear(3, 5)
    assert layer(data_in).shape == (11, 5)

    # Inference mode grad variables
    _ = layer(data_in)
    assert layer.__last_in == None
    assert layer.__last_out == None

    _ = layer(data_in, True)
    assert layer.__last_in != None
    assert layer.__last_out != None
    layer.clear_grad()

    # Backward sizes
    layer = Linear(3, 5)
    data_in = np.zeros((3,))
    data_out = np.zeros((5,))
    _ = layer(data_in, True)
    assert layer.backward(data_out).shape == (3,)

    data_in = np.zeros((10, 3))
    data_out = np.zeros((10, 5))
    _ = layer(data_in, True)
    assert layer.backward(data_out).shape == (10, 3)

    # Type failure checking
    with pytest.raises(TypeError):
        layer = Linear(1.1, 2)
    
    with pytest.raises(TypeError):
        layer = Linear(1, 2.2)

    with pytest.raises(TypeError):
        layer = Linear(1, 2, biases=3)

    # Saving + Loading
    layer = Linear((11, 3), 5, True)
    l_save = layer.save_to_file()
    load_layer = Linear.from_save(l_save)
    assert load_layer.layer_weight == layer.layer_weight
    assert load_layer.bias_weight == layer.bias_weight
    assert load_layer.in_size == layer.in_size
    assert load_layer.out_size == layer.out_size

    # Deepcopy
    layer2 = layer.deepcopy()
    assert layer2 is not layer
    assert layer2.layer_weight == layer.layer_weight
    assert layer2.bias_weight == layer.bias_weight
    assert layer2.in_size == layer.in_size
    assert layer2.out_size == layer.out_size



def dense_tests():
    # Non-tuple sizes
    layer = Dense(ReLU(), 3, 5)
    data_in = np.zeros((3,))
    assert layer(data_in).shape == (5,)
    assert layer.out_size == (5,)

    # Tuple sizes
    layer = Dense(ReLU(), (10, 3), 5)
    data_in = np.zeros((10, 3))
    assert layer(data_in).shape == (10, 5)
    assert layer.out_size == (10, 5)

    # Allow for other-valued batch sizes
    data_in = np.zeros((11, 3))
    assert layer(data_in).shape == (11, 5)

    layer = Dense(ReLU(), 3, 5)
    assert layer(data_in).shape == (11, 5)

    # Inference mode grad variables
    _ = layer(data_in)
    assert layer.__last_in == None
    assert layer.__last_out == None

    _ = layer(data_in, True)
    assert layer.__last_in != None
    assert layer.__last_out != None
    layer.clear_grad()

    # Backward sizes
    layer = Dense(ReLU(), 3, 5)
    data_in = np.zeros((3,))
    data_out = np.zeros((5,))
    _ = layer(data_in, True)
    assert layer.backward(data_out).shape == (3,)

    data_in = np.zeros((10, 3))
    data_out = np.zeros((10, 5))
    _ = layer(data_in, True)
    assert layer.backward(data_out).shape == (10, 3)

    # Type failure checking
    with pytest.raises(TypeError):
        layer = Dense(None, 1.1, 2)
    
    with pytest.raises(TypeError):
        layer = Dense(None, 1, 2.2)

    with pytest.raises(TypeError):
        layer = Dense(None, 3, 2)
        data_in = np.zeros((3,))
        _ = layer(data_in)

    # Saving + Loading
    layer = Dense(ReLU(), (11, 3), 5)
    l_save = layer.save_to_file()
    load_layer = Dense.from_save(l_save)
    assert isinstance(load_layer.funct, ReLU)
    assert load_layer.layer_weight == layer.layer_weight
    assert load_layer.bias_weight == layer.bias_weight
    assert load_layer.in_size == layer.in_size
    assert load_layer.out_size == layer.out_size

    # Deepcopy
    layer2 = layer.deepcopy()
    assert layer2 is not layer
    assert isinstance(layer2.funct, ReLU)
    assert layer2.layer_weight == layer.layer_weight
    assert layer2.bias_weight == layer.bias_weight
    assert layer2.in_size == layer.in_size
    assert layer2.out_size == layer.out_size


def bilinear_tests():
    # Non-tuple sizes
    layer = Bilinear(ReLU(), 4, 3, 5)
    data_1 = np.zeros((4,))
    data_2 = np.zeros((3,))
    assert layer(data_1, data_2).shape == (5,)
    assert layer.out_size == (5,)

    # Tuple sizes
    layer = Dense(ReLU(), (10, 4), (10, 3), 5)
    data_1 = np.zeros((10, 4))
    data_2 = np.zeros((10, 3))
    assert layer(data_1, data_2).shape == (10, 5)
    assert layer.out_size == (10, 5)

    # Allow for other-valued batch sizes
    data_1 = np.zeros((11, 4))
    data_2 = np.zeros((11, 3))
    assert layer(data_1, data_2).shape == (11, 5)

    layer = Dense(ReLU(), 4, 3, 5)
    assert layer(data_1, data_2).shape == (11, 5)

    # Inference mode grad variables
    _ = layer(data_1, data_2)
    assert layer.__last_in == None
    assert layer.__last_out == None

    _ = layer(data_1, data_2, True)
    assert layer.__last_in != None
    assert layer.__last_out != None
    layer.clear_grad()

    # Backward sizes
    layer = Bilinear(ReLU(), 4, 3, 5)
    data_1 = np.zeros((4,))
    data_2 = np.zeros((3,))
    data_out = np.zeros((5,))
    _ = layer(data_1, data_2, True)
    ret_out = layer.backward(data_out)
    assert ret_out[0].shape == (4,)
    assert ret_out[1].shape == (3,)

    data_1 = np.zeros((10, 4))
    data_2 = np.zeros((10, 3))
    data_out = np.zeros((10, 5))
    _ = layer(data_1, data_2, True)
    ret_out = layer.backward(data_out)
    assert ret_out[0].shape == (10, 4)
    assert ret_out[1].shape == (10, 3)

    # Type failure checking
    with pytest.raises(TypeError):
        layer = Dense(None, 1.1, 1, 2)
    
    with pytest.raises(TypeError):
        layer = Dense(None, 1, 1.1, 2)

    with pytest.raises(TypeError):
        layer = Dense(None, 1, 1, 2.2)

    with pytest.raises(TypeError):
        layer = Dense(None, 2, 3, 2)
        data_1 = np.zeros((2,))
        data_2 = np.zeros((3,))
        _ = layer(data_1, data_2)

    with pytest.raises(ValueError):
        layer = Dense(ReLU(), (11, 2), (11, 3), 2)
        data_1 = np.zeros((11, 2))
        data_2 = np.zeros((10, 3))
        _ = layer(data_1, data_2)

    # Saving + Loading
    layer = Bilinear(ReLU(), (11, 4), (11, 3), 5)
    l_save = layer.save_to_file()
    load_layer = Bilinear.from_save(l_save)
    assert isinstance(load_layer.funct, ReLU)
    assert load_layer.layer_weight == layer.layer_weight
    assert load_layer.bias_weight == layer.bias_weight
    assert load_layer.in_size == layer.in_size
    assert load_layer.out_size == layer.out_size

    # Deepcopy
    layer2 = layer.deepcopy()
    assert layer2 is not layer
    assert isinstance(layer2.funct, ReLU)
    assert layer2.layer_weight == layer.layer_weight
    assert layer2.bias_weight == layer.bias_weight
    assert layer2.in_size == layer.in_size
    assert layer2.out_size == layer.out_size
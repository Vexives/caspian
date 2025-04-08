from caspian.layers import Layer, Linear, Dense, Bilinear
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
    data_in = np.random.randn((3,))
    forward_result, train = layer(data_in)
    assert forward_result == data_in
    assert train is False

    _, train = layer(data_in, True)
    assert train is True


def linear_tests():
    # Non-tuple sizes
    layer = Linear(3, 5)
    data_in = np.random.randn((3,))
    assert layer(data_in).shape == (5,)
    assert layer.out_size == (5,)

    # Tuple sizes
    layer = Linear((10, 3), 5)
    data_in = np.random.randn((10, 3))
    assert layer(data_in).shape == (10, 5)
    assert layer.out_size == (10, 5)

    # Allow for other-valued batch sizes
    data_in = np.random.randn((11, 3))
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
    data_in = np.random.randn((3,))
    data_out = np.random.randn((5,))
    _ = layer(data_in, True)
    assert layer.backward(data_out).shape == (3,)

    data_in = np.random.randn((10, 3))
    data_out = np.random.randn((10, 5))
    _ = layer(data_in, True)
    assert layer.backward(data_out).shape == (10, 3)

    # Type failure checking
    with pytest.raises(TypeError):
        layer = Linear(1.1, 2)
    
    with pytest.raises(TypeError):
        layer = Linear(1, 2.2)

    with pytest.raises(TypeError):
        layer = Linear(1, 2, biases=3)


def dense_tests():
    pass
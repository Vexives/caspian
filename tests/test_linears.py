from caspian.layers import Layer, Linear, Dense, Bilinear, Embedding
from caspian.activations import ReLU
from caspian.utilities import InvalidDataException, ShapeIncompatibilityException
import pytest
import numpy as np

def test_base_layer():
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
    assert np.allclose(forward_result, data_in)
    assert train is False

    _, train = layer(data_in, True)
    assert train is True




def test_linear():
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
    data_out = np.zeros((5,))
    with pytest.raises(AttributeError):
        _ = layer.backward(data_out)


    # Private variables not accessable outside of layer
    _ = layer(data_in, True)
    with pytest.raises(AttributeError):
        _ = layer.__last_in == None
    with pytest.raises(AttributeError):
        _ = layer.__last_out == None
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
    with pytest.raises(InvalidDataException):
        layer = Linear(1.1, 2)
    
    with pytest.raises(InvalidDataException):
        layer = Linear(1, 2.2)

    with pytest.raises(InvalidDataException):
        layer = Linear(-1, 2)

    with pytest.raises(InvalidDataException):
        layer = Linear((-1, 2), 2)

    with pytest.raises(AttributeError):
        layer = Linear(1, 2, biases=3)
        _ = layer.bias_weight.shape


    # Incorrect sizing
    layer = Linear(10, 5)
    data_in = np.zeros((10,))
    data_false_in = np.zeros((11,))
    data_false_out = np.zeros((6,))
    with pytest.raises(ValueError):
        _ = layer(data_false_in)

    _ = layer(data_in, True)
    with pytest.raises(ValueError):
        _ = layer.backward(data_false_out)


    # Saving + Loading
    layer = Linear((11, 3), 5, True)
    l_save = layer.save_to_file()
    load_layer = Linear.from_save(l_save)
    assert np.allclose(load_layer.layer_weight, layer.layer_weight)
    assert np.allclose(load_layer.bias_weight, layer.bias_weight)
    assert np.allclose(load_layer.in_size, layer.in_size)
    assert np.allclose(load_layer.out_size, layer.out_size)


    # Deepcopy
    layer2 = layer.deepcopy()
    assert layer2 is not layer
    assert np.allclose(layer2.layer_weight, layer.layer_weight)
    assert np.allclose(layer2.bias_weight, layer.bias_weight)
    assert np.allclose(layer2.in_size, layer.in_size)
    assert np.allclose(layer2.out_size, layer.out_size)




def test_dense():
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
    data_out = np.zeros((5,))
    with pytest.raises(AttributeError):
        _ = layer.backward(data_out)


    # Private variables not accessable outside of layer
    _ = layer(data_in, True)
    with pytest.raises(AttributeError):
        _ = layer.__last_in == None
    with pytest.raises(AttributeError):
        _ = layer.__last_out == None
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
    with pytest.raises(InvalidDataException):
        layer = Dense(None, 1.1, 2)
    
    with pytest.raises(InvalidDataException):
        layer = Dense(None, 1, 2.2)

    with pytest.raises(InvalidDataException):
        layer = Dense(None, -1, 2)

    with pytest.raises(InvalidDataException):
        layer = Dense(None, (-1, 2), 2)

    with pytest.raises(TypeError):
        layer = Dense(None, 3, 2)
        data_in = np.zeros((3,))
        _ = layer(data_in)


    # Incorrect sizing
    layer = Dense(ReLU(), 10, 5)
    data_in = np.zeros((10,))
    data_false_in = np.zeros((11,))
    data_false_out = np.zeros((6,))
    with pytest.raises(ValueError):
        _ = layer(data_false_in)

    _ = layer(data_in, True)
    with pytest.raises(ValueError):
        _ = layer.backward(data_false_out)


    # Saving + Loading
    layer = Dense(ReLU(), (11, 3), 5)
    l_save = layer.save_to_file()
    load_layer = Dense.from_save(l_save)
    assert isinstance(load_layer.funct, ReLU)
    assert np.allclose(load_layer.layer_weight, layer.layer_weight)
    assert np.allclose(load_layer.bias_weight, layer.bias_weight)
    assert np.allclose(load_layer.in_size, layer.in_size)
    assert np.allclose(load_layer.out_size, layer.out_size)


    # Deepcopy
    layer2 = layer.deepcopy()
    assert layer2 is not layer
    assert isinstance(layer2.funct, ReLU)
    assert np.allclose(layer2.layer_weight, layer.layer_weight)
    assert np.allclose(layer2.bias_weight, layer.bias_weight)
    assert np.allclose(layer2.in_size, layer.in_size)
    assert np.allclose(layer2.out_size, layer.out_size)




def test_bilinear():
    # Non-tuple sizes
    layer = Bilinear(ReLU(), 4, 3, 5)
    data_1 = np.zeros((4,))
    data_2 = np.zeros((3,))
    assert layer(data_1, data_2).shape == (5,)
    assert layer.out_size == (5,)


    # Tuple sizes
    layer = Bilinear(ReLU(), (10, 4), (10, 3), 5)
    data_1 = np.zeros((10, 4))
    data_2 = np.zeros((10, 3))
    assert layer(data_1, data_2).shape == (10, 5)
    assert layer.out_size == (10, 5)


    # Allow for other-valued batch sizes
    data_1 = np.zeros((11, 4))
    data_2 = np.zeros((11, 3))
    assert layer(data_1, data_2).shape == (11, 5)

    layer = Bilinear(ReLU(), 4, 3, 5)
    assert layer(data_1, data_2).shape == (11, 5)


    # Inference mode grad variables
    _ = layer(data_1, data_2)
    data_out = np.zeros((5,))
    with pytest.raises(AttributeError):
        _ = layer.backward(data_out)


    # Private variables are not able to be accessed outside of layer
    _ = layer(data_1, data_2, True)
    with pytest.raises(AttributeError):
        _ = layer.__last_in == None
    with pytest.raises(AttributeError):
        _ = layer.__last_out == None
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
    with pytest.raises(InvalidDataException):
        layer = Bilinear(None, 1.1, 1, 2)
    
    with pytest.raises(InvalidDataException):
        layer = Bilinear(None, 1, 1.1, 2)

    with pytest.raises(InvalidDataException):
        layer = Bilinear(None, 1, 1, 2.2)

    with pytest.raises(InvalidDataException):
        layer = Bilinear(None, -1, -1, 2)

    with pytest.raises(InvalidDataException):
        layer = Bilinear(None, (-1, 2), (-1, 2), 2)

    with pytest.raises(TypeError):
        layer = Bilinear(None, 2, 3, 2)
        data_1 = np.zeros((2,))
        data_2 = np.zeros((3,))
        _ = layer(data_1, data_2)

    with pytest.raises(ValueError):
        layer = Bilinear(ReLU(), (11, 2), (11, 3), 2)
        data_1 = np.zeros((11, 2))
        data_2 = np.zeros((10, 3))
        _ = layer(data_1, data_2)


    # Incorrect sizing
    layer = Bilinear(ReLU(), 10, 15, 5)
    data_1 = np.zeros((10,))
    data_2 = np.zeros((15,))

    data_false_1 = np.zeros((11,))
    data_false_2 = np.zeros((16,))

    data_false_out = np.zeros((6,))
    with pytest.raises(ValueError):
        _ = layer(data_false_1, data_false_2)

    _ = layer(data_1, data_2, True)
    with pytest.raises(ValueError):
        _ = layer.backward(data_false_out)


    # Saving + Loading
    layer = Bilinear(ReLU(), (11, 4), (11, 3), 5)
    l_save = layer.save_to_file()
    load_layer = Bilinear.from_save(l_save)
    assert isinstance(load_layer.funct, ReLU)
    assert np.allclose(load_layer.layer_weight, layer.layer_weight)
    assert np.allclose(load_layer.bias_weight, layer.bias_weight)
    assert np.allclose(load_layer.in_size, layer.in_size)
    assert np.allclose(load_layer.out_size, layer.out_size)


    # Deepcopy
    layer2 = layer.deepcopy()
    assert layer2 is not layer
    assert isinstance(layer2.funct, ReLU)
    assert np.allclose(layer2.layer_weight, layer.layer_weight)
    assert np.allclose(layer2.bias_weight, layer.bias_weight)
    assert np.allclose(layer2.in_size, layer.in_size)
    assert np.allclose(layer2.out_size, layer.out_size)




def test_embedding():
    # Integer size enforcement
    with pytest.raises(InvalidDataException):
        _ = Embedding(10, (1, 10))

    with pytest.raises(InvalidDataException):
        _ = Embedding((1, 10), 10)
    

    # Output sizes
    layer = Embedding(10, 5)
    data_in = np.zeros((12, 10))
    assert layer(data_in).shape == (12, 5)


    # Backward pass sizes
    data_out = np.zeros((12, 5))
    _ = layer(data_in, True)
    assert layer.backward(data_out).shape == (12, 10)


    # Incorrect sizing
    layer = Embedding(10, 5)
    data_in = np.zeros((10,))
    data_false_in = np.zeros((11,))
    data_false_out = np.zeros((6,))
    with pytest.raises(ValueError):
        _ = layer(data_false_in)

    _ = layer(data_in, True)
    with pytest.raises(ValueError):
        _ = layer.backward(data_false_out)


    # Saving + Loading
    layer = Embedding(12, 20)
    l_save = layer.save_to_file()
    load_layer = Embedding.from_save(l_save)
    assert np.allclose(load_layer.embed_table, layer.embed_table)
    assert load_layer.v_len == layer.v_len
    assert load_layer.e_len == layer.e_len


    # Deepcopy
    layer2 = layer.deepcopy()
    assert layer2 is not layer
    assert np.allclose(layer2.embed_table, layer.embed_table)
    assert layer2.v_len == layer.v_len
    assert layer2.e_len == layer.e_len
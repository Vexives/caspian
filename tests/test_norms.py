from caspian.layers import BatchNorm, LayerNorm
from caspian.utilities import InvalidDataException
import numpy as np
import pytest

def test_batchnorm():
    # Invalid init tests
    with pytest.raises(InvalidDataException):
        _ = BatchNorm(0, 2)

    with pytest.raises(InvalidDataException):
        _ = BatchNorm(2, 0)

    with pytest.raises(InvalidDataException):
        _ = BatchNorm(2, 2, "test")

    with pytest.raises(InvalidDataException):
        _ = BatchNorm(2, 2, False, "test")

    with pytest.raises(InvalidDataException):
        _ = BatchNorm(2, 2, momentum = 1.0)

    with pytest.raises(InvalidDataException):
        _ = BatchNorm(2, 2, momentum = 0.0)

    with pytest.raises(InvalidDataException):
        _ = BatchNorm(2, 2, var_eps = 0.0)

    with pytest.raises(InvalidDataException):
        _ = BatchNorm(4, 3, axis = 4)

    
    # Running mean and variance presence
    layer = BatchNorm(4, 2)
    assert layer.running_mean is not None
    assert layer.running_var is not None

    layer = BatchNorm(4, 3, momentum = None)
    assert layer.running_mean is None
    assert layer.running_var is None

    layer = BatchNorm(4, 3, False, False)
    data_in = np.random.uniform(0.0, 1.0, (10, 4, 5, 5))
    for i in range(10):
        _ = layer(data_in, True)
    rm, rv = layer.running_mean, layer.running_var

    _ = layer(data_in)
    assert np.allclose(rm, layer.running_mean)
    assert np.allclose(rv, layer.running_var)


    # Affine variable presence
    layer = BatchNorm(4, 3, True, False)
    assert layer.gamma is not None
    assert layer.beta is None

    layer = BatchNorm(4, 3, False, True)
    assert layer.gamma is None
    assert layer.beta is not None

    l1 = BatchNorm(4, 3, True, True)
    l2 = BatchNorm(4, 3, False, False)
    data_in = np.random.uniform(0.0, 1.0, (10, 4, 5, 5))
    data_out = np.random.uniform(0.0, 1.0, data_in.shape)

    x, y = l1(data_in, True), l2(data_in, True)
    assert x.shape == y.shape
    assert np.allclose(x, y)

    _, _ = l1.backward(data_out), l2.backward(data_out)
    x, y = l1(data_in), l2(data_in)
    assert x.shape == y.shape
    assert not np.allclose(x, y)


    # Differing channel axis
    layer = BatchNorm(4, 3, axis = 3)
    data_in = np.zeros((10, 5, 5, 4))
    assert layer(data_in).shape == (10, 5, 5, 4)


    # Private variable access
    _ = layer(data_in, True)
    with pytest.raises(AttributeError):
        _ = layer.__norm_res


    # Deepcopy
    layer = BatchNorm(4, 3, True, False, 0.995, 3, 1e-10)
    layer2 = layer.deepcopy()
    assert layer2 is not layer
    assert np.allclose(layer2.running_mean, layer.running_mean)
    assert np.allclose(layer2.running_var, layer.running_var)
    assert np.allclose(layer2.gamma, layer.gamma)
    assert layer2.beta is None
    assert layer2.axis == layer.axis
    assert layer2.dims == layer.dims
    assert layer2.var_eps == layer.var_eps
    assert layer2.momentum == layer.momentum


    # Saving and loading
    context = layer.save_to_file()
    load_layer = BatchNorm.from_save(context)
    assert np.allclose(load_layer.running_mean, layer.running_mean)
    assert np.allclose(load_layer.running_var, layer.running_var)
    assert np.allclose(load_layer.gamma, layer.gamma)
    assert load_layer.beta is None
    assert load_layer.axis == layer.axis
    assert load_layer.dims == layer.dims
    assert load_layer.var_eps == layer.var_eps
    assert load_layer.momentum == layer.momentum




def test_layernorm():
    # Invalid init tests
    with pytest.raises(InvalidDataException):
        _ = LayerNorm("test")

    with pytest.raises(InvalidDataException):
        _ = LayerNorm((5, 10, 10), "test")

    with pytest.raises(InvalidDataException):
        _ = LayerNorm((5, 10, 10), False, "test")

    with pytest.raises(InvalidDataException):
        _ = LayerNorm((5, 10, 10), var_eps = 0.0)


    # Affine variable presence
    layer = LayerNorm((5, 10, 10), True, False)
    assert layer.layer_weight is not None
    assert layer.bias_weight is None

    layer = LayerNorm((5, 10, 10), False, True)
    assert layer.layer_weight is None
    assert layer.bias_weight is not None

    l1 = LayerNorm((5, 10, 10), True, True)
    l2 = LayerNorm((5, 10, 10), False, False)
    data_in = np.random.uniform(0.0, 1.0, (5, 5, 10, 10))
    x, y = l1(data_in), l2(data_in)
    assert x.shape == y.shape
    assert not np.allclose(x, y)

    l1 = LayerNorm((5, 10, 10), False, False)
    l2 = LayerNorm((5, 10, 10), False, False)
    data_in = np.random.uniform(0.0, 1.0, (5, 5, 10, 10))
    x, y = l1(data_in), l2(data_in)
    assert x.shape == y.shape
    assert np.allclose(x, y)


    # Private variable access
    _ = layer(data_in, True)
    with pytest.raises(AttributeError):
        _ = layer.__norm_res


    # Deepcopy
    layer = LayerNorm((5, 10, 20, 15), True, False, 1e-9)
    layer2 = layer.deepcopy()
    assert layer2 is not layer
    assert layer2.bias_weight is None
    assert np.allclose(layer2.layer_weight, layer.layer_weight)
    assert layer2.in_size == layer.in_size
    assert layer2.out_size == layer.out_size
    assert layer2.var_eps == layer.var_eps


    # Saving and loading
    context = layer.save_to_file()
    load_layer = LayerNorm.from_save(context)
    assert load_layer.bias_weight is None
    assert np.allclose(load_layer.layer_weight, layer.layer_weight)
    assert load_layer.in_size == layer.in_size
    assert load_layer.out_size == layer.out_size
    assert load_layer.var_eps == layer.var_eps
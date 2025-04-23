from caspian.layers import Layer, Container, Dropout, Reshape, Sequence, Upsampling1D, Upsampling2D, Upsampling3D
from caspian.activations import ReLU, Identity
from caspian.utilities import InvalidDataException, ShapeIncompatibilityException
import numpy as np
import pytest

def test_container():
    # Standard usage
    layer = Container(ReLU())
    assert isinstance(layer, Layer)

    t_func = ReLU()
    data_in = np.random.uniform(-1.0, 1.0, (5, 5))
    assert np.allclose(layer(data_in), t_func(data_in))
    assert np.allclose(layer.backward(data_in), t_func.backward(data_in))

    
    # Incorrect function pass
    with pytest.raises(InvalidDataException):
        _ = Container("test")

    with pytest.raises(InvalidDataException):
        _ = Container(np.max)


    # Deepcopy
    layer2 = layer.deepcopy()
    assert layer2 is not layer
    assert isinstance(layer2.funct, ReLU)


    # Saving
    context = layer.save_to_file()
    layer2 = Container.from_save(context)
    assert isinstance(layer2.funct, ReLU)




def test_dropout():
    # Bad float value tests
    with pytest.raises(InvalidDataException):
        _ = Dropout("test")

    with pytest.raises(InvalidDataException):
        _ = Dropout(0.0)

    with pytest.raises(InvalidDataException):
        _ = Dropout(1.0)


    # Backwards mask test
    layer = Dropout()
    data = np.zeros((5, 10))
    with pytest.raises(ShapeIncompatibilityException):
        _ = layer.backward(data)

    _ = layer(data, True)
    assert layer.backward(data).shape == (5, 10)


    # Deepcopy
    layer2 = layer.deepcopy()
    assert layer2 is not layer
    assert layer2.chance == layer.chance


    # Saving
    context = layer.save_to_file()
    layer2 = Dropout.from_save(context)
    assert layer2.chance == layer.chance




def test_reshape():
    # Invalid shape values (below 0 excluding -1)
    with pytest.raises(ShapeIncompatibilityException):
        _ = Reshape((-2, 1, 2, 3), (1, 1, 2, 3))

    with pytest.raises(ShapeIncompatibilityException):
        _ = Reshape((0.1, 5, 5), (-1, 5, 5))

    with pytest.raises(ShapeIncompatibilityException):
        _ = Reshape((5, 5, 5), (0, 5, 5, 5))

    with pytest.raises(ShapeIncompatibilityException):
        _ = Reshape("test", (5, 5, 5))


    # Incompatible reshape sizes
    with pytest.raises(ShapeIncompatibilityException):
        _ = Reshape((5, 5, 5), (5, 5, 6))

    with pytest.raises(ShapeIncompatibilityException):
        _ = Reshape((10, 5), (4, 10))

    with pytest.raises(ShapeIncompatibilityException):
        _ = Reshape((-1, 5, 5), (20, 5, 5))


    # Standard usage
    layer = Reshape((5, 5, -1), (-1, 25))
    data = np.zeros((25, 5, 5))
    assert layer(data).shape == (25, 25)

    layer = Reshape((20, 5, 5), (-1, 5))
    data = np.zeros((20, 5, 5))
    assert layer(data).shape == (100, 5)


    # Deepcopy
    layer2 = layer.deepcopy()
    assert layer2 is not layer
    assert layer2.in_size == layer.in_size
    assert layer2.out_size == layer2.out_size


    # Saving
    context = layer.save_to_file()
    layer2 = Reshape.from_save(context)
    assert layer2.in_size == layer.in_size
    assert layer2.out_size == layer2.out_size




def test_sequence():
    pass




def test_upsample1D():
    pass




def test_upsample2D():
    pass




def test_upsample3D():
    pass
from caspian.layers import Layer, Container, Dropout, Reshape, Sequence, Upsampling1D, Upsampling2D, Upsampling3D, Linear, Bilinear, PositionalEncoding, CosineSimilarity
from caspian.activations import ReLU, Identity
from caspian.utilities import InvalidDataException, ShapeIncompatibilityException, BackwardSequenceException
import numpy as np
import pytest

def test_container():
    # Standard usage
    layer = Container(ReLU())
    assert isinstance(layer, Layer)

    t_func = ReLU()
    data_in = np.random.uniform(-1.0, 1.0, (5, 5))
    out_err = np.random.uniform(1.0, 2.0, (5, 5))
    assert np.allclose(layer(data_in, True), t_func(data_in))
    assert np.allclose(layer.backward(out_err), 
                       t_func(data_in, out_err))


    # No training mode test
    layer = Container(ReLU())
    data_in = np.random.uniform(-1.0, 1.0, (5, 5))
    _ = layer(data_in)
    with pytest.raises(AttributeError):
        _ = layer.backward(data_in)

    
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

    with pytest.raises(InvalidDataException):
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
    # Invalid input tests
    with pytest.raises(InvalidDataException):
        _ = Sequence([])

    with pytest.raises(InvalidDataException):
        _ = Sequence([Reshape((1, 2, 3), (3, 2, 1)), "test"])

    with pytest.raises(InvalidDataException):
        _ = Sequence(["test", Reshape((1, 2, 3), (3, 2, 1))])

    with pytest.raises(InvalidDataException):
        _ = Sequence([Linear(3, 5), Linear(6, 5)])

    layer = Sequence([Reshape((1, 2, 3), (3, 2, 1)), Upsampling2D(2)])
    with pytest.raises(InvalidDataException):
        layer += "test"

    with pytest.raises(InvalidDataException):
        layer += 1.1

    
    # Non-Sequentiable layer test
    with pytest.raises(InvalidDataException):
        _ = Sequence([Bilinear(Identity(), 3, 5, 8)])


    # General usage tests
    layer = Sequence([Linear(5, 3), Linear(3, 6), Linear(6, 8)])
    data_in = np.zeros((5,))
    assert layer(data_in).shape == (8,)
    assert layer.in_size == (5,)
    assert layer.out_size == (8,)
    assert layer.num_layers == 3

    layer = Sequence([Upsampling2D(3), Upsampling2D(2)])
    data_in = np.zeros((3, 3, 3))
    assert layer(data_in).shape == (3, 18, 18)
    assert layer.in_size == None
    assert layer.out_size == None
    assert layer.num_layers == 2


    # Backwards pass testing
    layer = Sequence([Linear(5, 3, True), Linear(3, 6, True), Linear(6, 8, True)])
    data_in = np.zeros((5,))
    data_out = np.zeros((8,))
    with pytest.raises(BackwardSequenceException):
        _ = layer.backward(data_out)

    _ = layer(data_in)
    with pytest.raises(BackwardSequenceException):
        _ = layer.backward(data_out)

    _ = layer(data_in, True)
    result = layer.backward(data_out)
    assert result.shape == (5,)


    # Deepcopy
    layer2 = layer.deepcopy()
    assert layer2 is not layer
    assert all(l2 is not l1 for l2, l1 in zip(layer2.layers, layer.layers))


    # Saving
    context = layer.save_to_file()
    layer2 = Sequence.from_save(context)
    assert layer2.num_layers == 3
    assert layer2.in_size == layer.in_size
    assert layer2.out_size == layer.out_size
    assert all(isinstance(l, Linear) for l in layer2.layers)
    assert all(np.allclose(l2.layer_weight, l1.layer_weight) for l2, l1 in zip(layer2.layers, layer.layers))
    assert all(np.allclose(l2.bias_weight, l1.bias_weight) for l2, l1 in zip(layer2.layers, layer.layers))




def test_upsample1D():
    # Invalid value tests
    with pytest.raises(InvalidDataException):
        _ = Upsampling1D(1.1)
    
    with pytest.raises(InvalidDataException):
        _ = Upsampling1D("test")

    with pytest.raises(InvalidDataException):
        _ = Upsampling1D((1,))


    # Standard usage tests
    layer = Upsampling1D(2)
    data_in = np.zeros((3, 5))
    assert layer(data_in).shape == (3, 10)

    layer = Upsampling1D(1)
    assert layer(data_in).shape == (3, 5)

    layer = Upsampling1D(3)
    assert layer(data_in).shape == (3, 15)


    # Input variation tests
    layer = Upsampling1D(2)
    data_in = np.zeros((3, 3, 5))
    assert layer(data_in).shape == (3, 3, 10)

    data_in = np.zeros((3, 4, 3, 5))
    assert layer(data_in).shape == (3, 4, 3, 10)

    data_in = np.zeros((5,))
    assert layer(data_in).shape == (10,)


    # Backward pass tests
    layer = Upsampling1D(2)
    data_in = np.zeros((3, 5))
    data_out = np.zeros((3, 10))
    _ = layer(data_in, True)
    assert layer.backward(data_out).shape == (3, 5)

    layer = Upsampling1D(1)
    data_out = np.zeros((3, 5))
    _ = layer(data_in, True)
    assert layer.backward(data_out).shape == (3, 5)

    layer = Upsampling1D(3)
    data_out = np.zeros((3, 15))
    _ = layer(data_in, True)
    assert layer.backward(data_out).shape == (3, 5)


    # Deepcopy
    layer2 = layer.deepcopy()
    assert layer2 is not layer
    assert layer2.rate == layer.rate


    # Saving
    context = layer.save_to_file()
    layer2 = Upsampling1D.from_save(context)
    assert layer2.rate == layer.rate




def test_upsample2D():
    # Invalid value tests
    with pytest.raises(InvalidDataException):
        _ = Upsampling2D(1.1)
    
    with pytest.raises(InvalidDataException):
        _ = Upsampling2D("test")

    with pytest.raises(InvalidDataException):
        _ = Upsampling2D((1,))

    with pytest.raises(InvalidDataException):
        _ = Upsampling2D((2, 1, 1, 1))


    # Standard usage tests
    layer = Upsampling2D(2)
    data_in = np.zeros((3, 3, 5))
    assert layer(data_in).shape == (3, 6, 10)

    layer = Upsampling2D(1)
    assert layer(data_in).shape == (3, 3, 5)

    layer = Upsampling2D(3)
    assert layer(data_in).shape == (3, 9, 15)

    layer = Upsampling2D((1, 2))
    assert layer(data_in).shape == (3, 3, 10)

    layer = Upsampling2D((2, 1))
    assert layer(data_in).shape == (3, 6, 5)


    # Input variation tests
    layer = Upsampling2D(2)
    data_in = np.zeros((4, 3, 3, 5))
    assert layer(data_in).shape == (4, 3, 6, 10)

    data_in = np.zeros((9, 3, 4, 3, 5))
    assert layer(data_in).shape == (9, 3, 4, 6, 10)

    data_in = np.zeros((2, 5))
    assert layer(data_in).shape == (4, 10)

    data_in = np.zeros((5,))
    assert layer(data_in).shape == (2, 10)


    # Backward pass tests
    layer = Upsampling2D(2)
    data_in = np.zeros((5, 3, 5))
    data_out = np.zeros((5, 6, 10))
    _ = layer(data_in, True)
    assert layer.backward(data_out).shape == (5, 3, 5)

    layer = Upsampling2D(1)
    data_out = np.zeros((5, 3, 5))
    _ = layer(data_in, True)
    assert layer.backward(data_out).shape == (5, 3, 5)

    layer = Upsampling2D(3)
    data_out = np.zeros((5, 9, 15))
    _ = layer(data_in, True)
    assert layer.backward(data_out).shape == (5, 3, 5)

    layer = Upsampling2D((2, 1))
    data_out = np.zeros((5, 6, 5))
    _ = layer(data_in, True)
    assert layer.backward(data_out).shape == (5, 3, 5)

    layer = Upsampling2D((1, 2))
    data_out = np.zeros((5, 3, 10))
    _ = layer(data_in, True)
    assert layer.backward(data_out).shape == (5, 3, 5)


    # Deepcopy
    layer2 = layer.deepcopy()
    assert layer2 is not layer
    assert layer2.rate == layer.rate


    # Saving
    context = layer.save_to_file()
    layer2 = Upsampling2D.from_save(context)
    assert layer2.rate == layer.rate




def test_upsample3D():
    # Invalid value tests
    with pytest.raises(InvalidDataException):
        _ = Upsampling3D(1.1)
    
    with pytest.raises(InvalidDataException):
        _ = Upsampling3D("test")

    with pytest.raises(InvalidDataException):
        _ = Upsampling3D((1,))

    with pytest.raises(InvalidDataException):
        _ = Upsampling3D((2, 1, 1, 1))


    # Standard usage tests
    layer = Upsampling3D(2)
    data_in = np.zeros((4, 3, 3, 5))
    assert layer(data_in).shape == (4, 6, 6, 10)

    layer = Upsampling3D(1)
    assert layer(data_in).shape == (4, 3, 3, 5)

    layer = Upsampling3D(3)
    assert layer(data_in).shape == (4, 9, 9, 15)

    layer = Upsampling3D((2, 1, 2))
    assert layer(data_in).shape == (4, 6, 3, 10)

    layer = Upsampling3D((2, 1, 1))
    assert layer(data_in).shape == (4, 6, 3, 5)

    layer = Upsampling3D((1, 2, 1))
    assert layer(data_in).shape == (4, 3, 6, 5)


    # Input variation tests
    layer = Upsampling3D(2)
    data_in = np.zeros((5, 4, 3, 3, 5))
    assert layer(data_in).shape == (5, 4, 6, 6, 10)

    data_in = np.zeros((9, 9, 3, 4, 3, 5))
    assert layer(data_in).shape == (9, 9, 3, 8, 6, 10)

    data_in = np.zeros((2, 5))
    assert layer(data_in).shape == (2, 4, 10)

    data_in = np.zeros((5,))
    assert layer(data_in).shape == (2, 2, 10)


    # Backward pass tests
    layer = Upsampling3D(2)
    data_in = np.zeros((4, 5, 3, 5))
    data_out = np.zeros((4, 10, 6, 10))
    _ = layer(data_in, True)
    assert layer.backward(data_out).shape == (4, 5, 3, 5)

    layer = Upsampling3D(1)
    data_out = np.zeros((4, 5, 3, 5))
    _ = layer(data_in, True)
    assert layer.backward(data_out).shape == (4, 5, 3, 5)

    layer = Upsampling3D(3)
    data_out = np.zeros((4, 15, 9, 15))
    _ = layer(data_in, True)
    assert layer.backward(data_out).shape == (4, 5, 3, 5)

    layer = Upsampling3D((1, 2, 1))
    data_out = np.zeros((4, 5, 6, 5))
    _ = layer(data_in, True)
    assert layer.backward(data_out).shape == (4, 5, 3, 5)

    layer = Upsampling3D((1, 2, 3))
    data_out = np.zeros((4, 5, 6, 15))
    _ = layer(data_in, True)
    assert layer.backward(data_out).shape == (4, 5, 3, 5)


    # Deepcopy
    layer2 = layer.deepcopy()
    assert layer2 is not layer
    assert layer2.rate == layer.rate


    # Saving
    context = layer.save_to_file()
    layer2 = Upsampling3D.from_save(context)
    assert layer2.rate == layer.rate




def test_pos_encoding():
    # Invalid value tests
    with pytest.raises(InvalidDataException):
        _ = PositionalEncoding(-1, 1)

    with pytest.raises(InvalidDataException):
        _ = PositionalEncoding(1, -1)

    with pytest.raises(InvalidDataException):
        _ = PositionalEncoding(1.1, 1)

    with pytest.raises(InvalidDataException):
        _ = PositionalEncoding(1, 1.1)

    with pytest.raises(InvalidDataException):
        _ = PositionalEncoding(1, 11)


    # Standard usage tests
    layer = PositionalEncoding(32, 20)
    data_in = np.zeros((16, 20))
    assert layer(data_in).shape == data_in.shape
    assert np.allclose(layer.backward(data_in), data_in)

    data_in = np.zeros((33, 20))
    with pytest.raises(InvalidDataException):
        _ = layer(data_in)
    

    # Deepcopy
    layer2 = layer.deepcopy()
    assert layer2 is not layer
    assert np.allclose(layer2.encoding, layer.encoding)


    # Saving
    context = layer.save_to_file()
    layer2 = PositionalEncoding.from_save(context)
    assert np.allclose(layer2.encoding, layer.encoding)




def test_cosinesimilarity():
    # Invalid value tests
    with pytest.raises(InvalidDataException):
        _ = CosineSimilarity(1.1)
    
    with pytest.raises(InvalidDataException):
        _ = CosineSimilarity(eps = "test")

    with pytest.raises(InvalidDataException):
        _ = CosineSimilarity(eps = 0.0)

    with pytest.raises(InvalidDataException):
        l = CosineSimilarity(2)
        a = np.zeros((2, 5))
        b = np.zeros((2, 5))
        _ = l(a, b)

    
    # Standard usage tests
    layer = CosineSimilarity(0)
    a = np.arange(20).reshape(2, 10)
    b = (np.arange(20) * 2).reshape(2, 10)
    assert np.allclose(layer(a, b), np.ones((10,)))

    layer2 = CosineSimilarity(1)
    layer3 = CosineSimilarity(2)
    a = np.random.randn(2, 4, 5)
    b = np.random.randn(2, 4, 5)
    assert layer(a, b, True).shape == (4, 5)
    assert layer2(a, b, True).shape == (2, 5)
    assert layer3(a, b, True).shape == (2, 4)

    g_1 = np.random.randn(4, 5)
    g_2 = np.random.randn(2, 5)
    g_3 = np.random.randn(2, 4)
    out_1a, out_1b = layer.backward(g_1)
    out_2a, out_2b = layer2.backward(g_2)
    out_3a, out_3b = layer3.backward(g_3)
    assert out_1a.shape == a.shape
    assert out_2a.shape == a.shape
    assert out_3a.shape == a.shape
    assert out_1b.shape == b.shape
    assert out_2b.shape == b.shape
    assert out_3b.shape == b.shape


    # Deepcopy
    layer_copy = layer.deepcopy()
    assert layer_copy is not layer
    assert layer_copy.dim == layer.dim
    assert layer_copy.eps == layer.eps


    # Saving
    context = layer.save_to_file()
    layer_s = CosineSimilarity.from_save(context)
    assert layer_s.dim == layer.dim
    assert layer_s.eps == layer.eps
from caspian.layers import Conv1D, Conv2D, Conv3D, Conv1DTranspose, Conv2DTranspose, Conv3DTranspose
from caspian.activations import ReLU, Identity
from caspian.utilities import InvalidDataException, UnsafeMemoryAccessException
import numpy as np
import pytest

def test_conv1d():
    # Incorrect sizes
    with pytest.raises(InvalidDataException):
        _ = Conv1D(Identity(), 2, 2, 5)

    with pytest.raises(InvalidDataException):
        _ = Conv1D(Identity(), 2, 2, (5,))


    # Inference mode grad variables
    layer = Conv1D(Identity(), 2, 2, (2, 10), 2)
    data_in = np.zeros((2, 10))
    _ = layer(data_in)
    data_out = np.zeros((2, 5))
    with pytest.raises(AttributeError):
        _ = layer.backward(data_out)


    # Private variables not accessable outside of layer
    _ = layer(data_in, True)
    with pytest.raises(AttributeError):
        _ = layer.__last_in
    layer.clear_grad()

    with pytest.raises(AttributeError):
        _ = layer.__window_shape


    # Forward sizes
    layer = Conv1D(Identity(), 2, 2, (2, 20), 2)
    data_in = np.zeros((2, 20))
    assert layer(data_in).shape == (2, 10)

    layer = Conv1D(Identity(), 4, 3, (2, 20), 1)
    data_in = np.zeros((2, 20))
    assert layer(data_in).shape == (4, 18)

    layer = Conv1D(Identity(), 4, 3, (2, 20), 1, 2)
    data_in = np.zeros((2, 20))
    assert layer(data_in).shape == (4, 20)

    layer = Conv1D(Identity(), 1, 2, (2, 20), 2, 3)
    data_in = np.zeros((2, 20))
    assert layer(data_in).shape == (1, 11)

    layer = Conv1D(Identity(), 3, 3, (2, 20), 2)
    data_in = np.zeros((2, 20))
    assert layer(data_in).shape == (3, 9)


    # Backward sizes
    layer = Conv1D(Identity(), 3, 2, (2, 20), 2)
    data_in = np.zeros((2, 20))
    data_out = np.zeros((3, 10))
    _ = layer(data_in, True)
    assert layer.backward(data_out).shape == (2, 20)

    layer = Conv1D(Identity(), 5, 3, (2, 20), 1)
    data_in = np.zeros((2, 20))
    data_out = np.zeros((5, 18))
    _ = layer(data_in, True)
    assert layer.backward(data_out).shape == (2, 20)

    layer = Conv1D(Identity(), 4, 2, (2, 20), 2, 4)
    data_in = np.zeros((2, 20))
    data_out = np.zeros((4, 12))
    _ = layer(data_in, True)
    assert layer.backward(data_out).shape == (2, 20)  

    layer = Conv1D(Identity(), 4, 3, (2, 20), 2, 3)
    data_in = np.zeros((2, 20))
    data_out = np.zeros((4, 11))
    _ = layer(data_in, True)
    assert layer.backward(data_out).shape == (2, 20)  


    # Type failure checking
    with pytest.raises(InvalidDataException):
        layer = Conv1D(Identity(), 1.1, 2, (2, 10))
    
    with pytest.raises(InvalidDataException):
        layer = Conv1D(Identity(), 2, 2, "test")

    with pytest.raises(InvalidDataException):
        layer = Conv1D(Identity(), 2, 3, (2, 10), "c")

    with pytest.raises(InvalidDataException):
        layer = Conv1D(Identity(), 2, 2, (2, 10), 0)

    with pytest.raises(InvalidDataException):
        layer = Conv1D(Identity(), 2, 2, (2, 10), 1, -1)

    with pytest.raises(InvalidDataException):
        layer = Conv1D(None, 2, 2, (2, 10))


    # Incorrect shape tests
    layer = Conv1D(Identity(), 3, 2, (2, 20), 2)
    data_in = np.zeros((2, 20))
    data_false_in = np.zeros((4, 20))
    data_false_out = np.zeros((4, 11))
    with pytest.raises(UnsafeMemoryAccessException):
        _ = layer(data_false_in)

    _ = layer(data_in, True)
    with pytest.raises(UnsafeMemoryAccessException):
        _ = layer.backward(data_false_out)


    # Saving + Loading
    layer = Conv1D(Identity(), 2, 3, (2, 20), 2, 3)
    l_save = layer.save_to_file()
    load_layer = Conv1D.from_save(l_save)
    assert np.allclose(load_layer.kernel_weights, layer.kernel_weights)
    assert np.allclose(load_layer.bias_weights, layer.bias_weights)
    assert load_layer.kernel_size == layer.kernel_size
    assert load_layer.padding_all == layer.padding_all
    assert load_layer.strides == layer.strides
    assert load_layer.in_size == layer.in_size
    assert load_layer.out_size == layer.out_size


    # Deepcopy
    layer2 = layer.deepcopy()
    assert layer2 is not layer
    assert np.allclose(layer2.kernel_weights, layer.kernel_weights)
    assert np.allclose(layer2.bias_weights, layer.bias_weights)
    assert layer2.kernel_size == layer.kernel_size
    assert layer2.padding_all == layer.padding_all
    assert layer2.strides == layer.strides
    assert layer2.in_size == layer.in_size
    assert layer2.out_size == layer.out_size


    # From-Kernel initialization tests
    kernel = np.zeros((5, 2, 2))
    biases = np.zeros((5, 12))
    layer = Conv1D.from_kernel(Identity(), (2, 20), kernel, 2, 4, biases)
    assert layer.out_size == (5, 12)

    with pytest.raises(InvalidDataException):
        _ = Conv1D.from_kernel(Identity(), (2, 20), "test")

    with pytest.raises(InvalidDataException):
        _ = Conv1D.from_kernel(Identity(), (2, 20), kernel, 2, 4, "test")






def test_conv2d():
    pass




def test_conv3d():
    pass




def test_conv1d_transpose():
    pass




def test_conv2d_transpose():
    pass




def test_conv3d_transpose():
    pass
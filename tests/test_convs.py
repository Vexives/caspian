from caspian.layers import Conv1D, Conv2D, Conv3D, ConvND, Conv1DTranspose, Conv2DTranspose, Conv3DTranspose, ConvNDTranspose
from caspian.activations import Identity
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
        layer = Conv1D(Identity(), -1, 2, (2, 10))
    
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

    with pytest.raises(InvalidDataException):
        layer = Conv1D(Identity(), 2, 2, (2, -10))


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
    assert load_layer.use_bias == layer.use_bias


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
    assert layer2.use_bias == layer.use_bias


    # From-Kernel initialization tests
    kernel = np.zeros((5, 2, 2))
    biases = np.zeros((5, 12))
    layer = Conv1D.from_kernel(Identity(), (2, 20), kernel, 2, 4, biases)
    assert layer.out_size == (5, 12)

    with pytest.raises(InvalidDataException):
        _ = Conv1D.from_kernel(Identity(), (2, 20), "test")

    with pytest.raises(InvalidDataException):
        _ = Conv1D.from_kernel(Identity(), (2, 20), kernel, 2, 4, "test")

    with pytest.raises(InvalidDataException):
        _ = Conv1D.from_kernel(Identity(), (2, 20), kernel, 0, 4, biases)

    with pytest.raises(InvalidDataException):
        _ = Conv1D.from_kernel(Identity(), (2, 20), kernel, 2, -1, biases)

    with pytest.raises(InvalidDataException):
        _ = Conv1D.from_kernel(Identity(), (2, -20), kernel, 2, 4, biases)




def test_conv2d():
    # Incorrect sizes
    with pytest.raises(InvalidDataException):
        _ = Conv2D(Identity(), 2, 2, 5)

    with pytest.raises(InvalidDataException):
        _ = Conv2D(Identity(), 2, 2, (5, 5))


    # Inference mode grad variables
    layer = Conv2D(Identity(), 2, 2, (2, 10, 16), 2)
    data_in = np.zeros((2, 10, 16))
    _ = layer(data_in)
    data_out = np.zeros((2, 5, 8))
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
    layer = Conv2D(Identity(), 2, 2, (2, 20, 20), 2)
    data_in = np.zeros((2, 20, 20))
    assert layer(data_in).shape == (2, 10, 10)

    layer = Conv2D(Identity(), 4, 3, (2, 20, 20), 1)
    data_in = np.zeros((2, 20, 20))
    assert layer(data_in).shape == (4, 18, 18)

    layer = Conv2D(Identity(), 4, 3, (2, 20, 20), 1, 2)
    data_in = np.zeros((2, 20, 20))
    assert layer(data_in).shape == (4, 20, 20)

    layer = Conv2D(Identity(), 1, 2, (2, 20, 13), 2, 3)
    data_in = np.zeros((2, 20, 13))
    assert layer(data_in).shape == (1, 11, 8)

    layer = Conv2D(Identity(), 3, 3, (2, 20, 20), 2)
    data_in = np.zeros((2, 20, 20))
    assert layer(data_in).shape == (3, 9, 9)

    layer = Conv2D(Identity(), 3, (4, 2), (2, 20, 20), (4, 2))
    data_in = np.zeros((2, 20, 20))
    assert layer(data_in).shape == (3, 5, 10)


    # Backward sizes
    layer = Conv2D(Identity(), 3, 2, (2, 20, 26), 2)
    data_in = np.zeros((2, 20, 26))
    data_out = np.zeros((3, 10, 13))
    _ = layer(data_in, True)
    assert layer.backward(data_out).shape == (2, 20, 26)

    layer = Conv2D(Identity(), 5, 3, (2, 20, 10), 1)
    data_in = np.zeros((2, 20, 10))
    data_out = np.zeros((5, 18, 8))
    _ = layer(data_in, True)
    assert layer.backward(data_out).shape == (2, 20, 10)

    layer = Conv2D(Identity(), 4, 2, (2, 20, 20), 2, 4)
    data_in = np.zeros((2, 20, 20))
    data_out = np.zeros((4, 12, 12))
    _ = layer(data_in, True)
    assert layer.backward(data_out).shape == (2, 20, 20)  

    layer = Conv2D(Identity(), 4, 3, (2, 20, 15), (2, 3), (3, 0))
    data_in = np.zeros((2, 20, 15))
    data_out = np.zeros((4, 11, 5))
    _ = layer(data_in, True)
    assert layer.backward(data_out).shape == (2, 20, 15)  


    # Type failure checking
    with pytest.raises(InvalidDataException):
        layer = Conv2D(Identity(), 1.1, 2, (2, 10, 10))
    
    with pytest.raises(InvalidDataException):
        layer = Conv2D(Identity(), 2, 2, "test")

    with pytest.raises(InvalidDataException):
        layer = Conv2D(Identity(), 2, 3, (2, 10, 10), "c")

    with pytest.raises(InvalidDataException):
        layer = Conv2D(Identity(), 2, 2, (2, 10, 10), 0)

    with pytest.raises(InvalidDataException):
        layer = Conv2D(Identity(), 2, 2, (2, 10, 10), 1, -1)

    with pytest.raises(InvalidDataException):
        layer = Conv2D(None, 2, 2, (2, 10, 10))

    with pytest.raises(InvalidDataException):
        layer = Conv2D(Identity(), 2, 2, (2, -10, 10))

    with pytest.raises(InvalidDataException):
        layer = Conv2D(Identity(), 2, (2, 0), (2, 10, 10))

    with pytest.raises(InvalidDataException):
        layer = Conv2D(Identity(), -1, (2, 1), (2, 10, 10))

    with pytest.raises(InvalidDataException):
        layer = Conv2D(Identity(), 2, (2,), (2, 10, 10))

    with pytest.raises(InvalidDataException):
        layer = Conv2D(Identity(), 2, 2, (2, 10, 10), (2,))

    with pytest.raises(InvalidDataException):
        layer = Conv2D(Identity(), 2, 2, (2, 10, 10), 2, (0,))

    with pytest.raises(InvalidDataException):
        layer = Conv2D(Identity(), 2, 2, (2, 10, 10), (2, 0))


    # Incorrect shape tests
    layer = Conv2D(Identity(), 3, 2, (2, 20, 20), 2)
    data_in = np.zeros((2, 20, 20))
    data_false_in = np.zeros((4, 20, 19))
    data_false_out = np.zeros((4, 11, 10))
    with pytest.raises(UnsafeMemoryAccessException):
        _ = layer(data_false_in)

    _ = layer(data_in, True)
    with pytest.raises(UnsafeMemoryAccessException):
        _ = layer.backward(data_false_out)


    # Saving + Loading
    layer = Conv2D(Identity(), 2, (2, 3), (2, 20, 18), (2, 3), (4, 6))
    l_save = layer.save_to_file()
    load_layer = Conv2D.from_save(l_save)
    assert np.allclose(load_layer.kernel_weights, layer.kernel_weights)
    assert np.allclose(load_layer.bias_weights, layer.bias_weights)
    assert load_layer.kernel_height == layer.kernel_height
    assert load_layer.kernel_width == layer.kernel_width
    assert load_layer.padding_all == layer.padding_all
    assert load_layer.stride_h == layer.stride_h
    assert load_layer.stride_w == layer.stride_w
    assert load_layer.in_size == layer.in_size
    assert load_layer.out_size == layer.out_size
    assert load_layer.use_bias == layer.use_bias


    # Deepcopy
    layer2 = layer.deepcopy()
    assert layer2 is not layer
    assert np.allclose(layer2.kernel_weights, layer.kernel_weights)
    assert np.allclose(layer2.bias_weights, layer.bias_weights)
    assert layer2.kernel_height == layer.kernel_height
    assert layer2.kernel_width == layer.kernel_width
    assert layer2.padding_all == layer.padding_all
    assert layer2.stride_h == layer.stride_h
    assert layer2.stride_w == layer.stride_w
    assert layer2.in_size == layer.in_size
    assert layer2.out_size == layer.out_size
    assert layer2.use_bias == layer.use_bias


    # From-Kernel initialization tests
    kernel = np.zeros((5, 2, 4, 4))
    biases = np.zeros((5, 6, 6))
    layer = Conv2D.from_kernel(Identity(), (2, 20, 20), kernel, 4, 4, biases)
    assert layer.out_size == (5, 6, 6)

    with pytest.raises(InvalidDataException):
        _ = Conv2D.from_kernel(Identity(), (2, 20, 20), "test")

    with pytest.raises(InvalidDataException):
        _ = Conv2D.from_kernel(Identity(), (2, 20, 20), kernel, 2, 4, "test")

    with pytest.raises(InvalidDataException):
        _ = Conv2D.from_kernel(Identity(), (2, 20, -20), kernel, 2, 4, biases)

    with pytest.raises(InvalidDataException):
        _ = Conv2D.from_kernel(Identity(), (2, 20, 20), kernel, 0, 4, biases)

    with pytest.raises(InvalidDataException):
        _ = Conv2D.from_kernel(Identity(), (2, 20, 20), kernel, 2, -1, biases)

    with pytest.raises(InvalidDataException):
        _ = Conv2D.from_kernel(Identity(), (2, 20, 20), kernel, (2, 0), 4, biases)

    with pytest.raises(InvalidDataException):
        _ = Conv2D.from_kernel(Identity(), (2, 20, 20), kernel, 2, (0, -1), biases)




def test_conv3d():
    # Incorrect sizes
    with pytest.raises(InvalidDataException):
        _ = Conv3D(Identity(), 2, 2, 5)

    with pytest.raises(InvalidDataException):
        _ = Conv3D(Identity(), 2, 2, (5, 5, 5))


    # Inference mode grad variables
    layer = Conv3D(Identity(), 2, 2, (2, 10, 16, 24), 2)
    data_in = np.zeros((2, 10, 16, 24))
    _ = layer(data_in)
    data_out = np.zeros((2, 5, 8, 12))
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
    layer = Conv3D(Identity(), 2, 2, (2, 20, 20, 20), 2)
    data_in = np.zeros((2, 20, 20, 20))
    assert layer(data_in).shape == (2, 10, 10, 10)

    layer = Conv3D(Identity(), 4, 3, (2, 20, 20, 20), 1)
    data_in = np.zeros((2, 20, 20, 20))
    assert layer(data_in).shape == (4, 18, 18, 18)

    layer = Conv3D(Identity(), 4, 3, (2, 20, 20, 20), 1, 2)
    data_in = np.zeros((2, 20, 20, 20))
    assert layer(data_in).shape == (4, 20, 20, 20)

    layer = Conv3D(Identity(), 1, 2, (2, 20, 13, 9), 2, 3)
    data_in = np.zeros((2, 20, 13, 9))
    assert layer(data_in).shape == (1, 11, 8, 6)

    layer = Conv3D(Identity(), 3, 3, (2, 20, 20, 20), 2)
    data_in = np.zeros((2, 20, 20, 20))
    assert layer(data_in).shape == (3, 9, 9, 9)

    layer = Conv3D(Identity(), 3, (4, 2, 3), (2, 20, 20, 21), (4, 2, 1), (0, 4, 1))
    data_in = np.zeros((2, 20, 20, 21))
    assert layer(data_in).shape == (3, 5, 12, 20)


    # Backward sizes
    layer = Conv3D(Identity(), 3, 2, (2, 20, 26, 28), 2)
    data_in = np.zeros((2, 20, 26, 28))
    data_out = np.zeros((3, 10, 13, 14))
    _ = layer(data_in, True)
    assert layer.backward(data_out).shape == (2, 20, 26, 28)

    layer = Conv3D(Identity(), 5, 3, (2, 20, 10, 15), 1)
    data_in = np.zeros((2, 20, 10, 15))
    data_out = np.zeros((5, 18, 8, 13))
    _ = layer(data_in, True)
    assert layer.backward(data_out).shape == (2, 20, 10, 15)

    layer = Conv3D(Identity(), 4, 2, (2, 20, 20, 20), 2, 4)
    data_in = np.zeros((2, 20, 20, 20))
    data_out = np.zeros((4, 12, 12, 12))
    _ = layer(data_in, True)
    assert layer.backward(data_out).shape == (2, 20, 20, 20)  

    layer = Conv3D(Identity(), 4, 3, (2, 20, 15, 18), (2, 3, 3), (3, 0, 3))
    data_in = np.zeros((2, 20, 15, 18))
    data_out = np.zeros((4, 11, 5, 7))
    _ = layer(data_in, True)
    assert layer.backward(data_out).shape == (2, 20, 15, 18)  


    # Type failure checking
    with pytest.raises(InvalidDataException):
        layer = Conv3D(Identity(), 1.1, 2, (2, 10, 10, 10))
    
    with pytest.raises(InvalidDataException):
        layer = Conv3D(Identity(), 2, 2, "test")

    with pytest.raises(InvalidDataException):
        layer = Conv3D(Identity(), 2, 3, (2, 10, 10, 10), "c")

    with pytest.raises(InvalidDataException):
        layer = Conv3D(Identity(), 2, 2, (2, 10, 10, 10), 0)

    with pytest.raises(InvalidDataException):
        layer = Conv3D(Identity(), 2, 2, (2, 10, 10, 10), 1, -1)

    with pytest.raises(InvalidDataException):
        layer = Conv3D(None, 2, 2, (2, 10, 10, 10))

    with pytest.raises(InvalidDataException):
        layer = Conv3D(Identity(), 2, 2, (2, -10, 10, 10))

    with pytest.raises(InvalidDataException):
        layer = Conv3D(Identity(), 2, (2, 0, 2), (2, 10, 10, 10))

    with pytest.raises(InvalidDataException):
        layer = Conv3D(Identity(), -1, (2, 1, 1), (2, 10, 10, 10))

    with pytest.raises(InvalidDataException):
        layer = Conv3D(Identity(), 2, 2, (2, 10, 10, 10), (2, 0, 2))

    with pytest.raises(InvalidDataException):
        layer = Conv3D(Identity(), 2, 2, (2, 10, 10, 10), (2, 2))

    with pytest.raises(InvalidDataException):
        layer = Conv3D(Identity(), 2, 2, (2, 10, 10, 10), 2, (2, 3))

    with pytest.raises(InvalidDataException):
        layer = Conv3D(Identity(), 2, (3, 2), (2, 10, 10, 10), 2)

    with pytest.raises(InvalidDataException):
        layer = Conv3D(Identity(), "a", (3, 2, 2), (2, 10, 10, 10), 2)


    # Incorrect shape tests
    layer = Conv3D(Identity(), 3, 2, (2, 20, 20, 20), 2)
    data_in = np.zeros((2, 20, 20, 20))
    data_false_in = np.zeros((4, 20, 19, 20))
    data_false_out = np.zeros((4, 11, 10, 10))
    with pytest.raises(UnsafeMemoryAccessException):
        _ = layer(data_false_in)

    _ = layer(data_in, True)
    with pytest.raises(UnsafeMemoryAccessException):
        _ = layer.backward(data_false_out)


    # Saving + Loading
    layer = Conv3D(Identity(), 10, (2, 3, 4), (2, 20, 18, 16), (2, 3, 4), (4, 6, 4))
    l_save = layer.save_to_file()
    load_layer = Conv3D.from_save(l_save)
    assert np.allclose(load_layer.kernel_weights, layer.kernel_weights)
    assert np.allclose(load_layer.bias_weights, layer.bias_weights)
    assert load_layer.kernel_height == layer.kernel_height
    assert load_layer.kernel_width == layer.kernel_width
    assert load_layer.kernel_depth == layer.kernel_depth
    assert load_layer.padding_all == layer.padding_all
    assert load_layer.stride_h == layer.stride_h
    assert load_layer.stride_w == layer.stride_w
    assert load_layer.stride_d == layer.stride_d
    assert load_layer.in_size == layer.in_size
    assert load_layer.out_size == layer.out_size
    assert load_layer.use_bias == layer.use_bias


    # Deepcopy
    layer2 = layer.deepcopy()
    assert layer2 is not layer
    assert np.allclose(layer2.kernel_weights, layer.kernel_weights)
    assert np.allclose(layer2.bias_weights, layer.bias_weights)
    assert layer2.kernel_height == layer.kernel_height
    assert layer2.kernel_width == layer.kernel_width
    assert layer2.kernel_depth == layer.kernel_depth
    assert layer2.padding_all == layer.padding_all
    assert layer2.stride_h == layer.stride_h
    assert layer2.stride_w == layer.stride_w
    assert layer2.stride_d == layer.stride_d
    assert layer2.in_size == layer.in_size
    assert layer2.out_size == layer.out_size
    assert layer2.use_bias == layer.use_bias


    # From-Kernel initialization tests
    kernel = np.zeros((5, 2, 4, 4, 3))
    biases = np.zeros((5, 6, 6, 7))
    layer = Conv3D.from_kernel(Identity(), (2, 20, 20, 18), kernel, (4, 4, 3), (4, 4, 3), biases)
    assert layer.out_size == (5, 6, 6, 7)

    with pytest.raises(InvalidDataException):
        _ = Conv3D.from_kernel(Identity(), (2, 20, 20, 20), "test")

    with pytest.raises(InvalidDataException):
        _ = Conv3D.from_kernel(Identity(), (2, 20, 20, 20), kernel, 2, 4, "test")

    with pytest.raises(InvalidDataException):
        _ = Conv3D.from_kernel(Identity(), (2, 20, -20, 20), kernel, 2, 4, biases)

    with pytest.raises(InvalidDataException):
        _ = Conv3D.from_kernel(Identity(), (2, 20, 20, 20), kernel, 0, 4, biases)

    with pytest.raises(InvalidDataException):
        _ = Conv3D.from_kernel(Identity(), (2, 20, 20, 20), kernel, 2, -1, biases)

    with pytest.raises(InvalidDataException):
        _ = Conv3D.from_kernel(Identity(), (2, 20, 20, 20), kernel, (2, 0, 2), 4, biases)

    with pytest.raises(InvalidDataException):
        _ = Conv3D.from_kernel(Identity(), (2, 20, 20, 20), kernel, 2, (0, -1, 0), biases)

    with pytest.raises(InvalidDataException):
        _ = Conv3D.from_kernel(Identity(), (2, 20, 20, 20), kernel, 2, (0, 0), biases)

    with pytest.raises(InvalidDataException):
        _ = Conv3D.from_kernel(Identity(), (2, 20, 20), kernel, 2, (0, 1, 0), biases)

    with pytest.raises(InvalidDataException):
        _ = Conv3D.from_kernel(Identity(), (2, 20, 20, 20), kernel, (2, 2), 0, biases)

    with pytest.raises(InvalidDataException):
        _ = Conv3D.from_kernel(None, (2, 20, 20, 20), kernel, 2, 0, biases)




def test_convNd():
    # Incorrect sizes
    with pytest.raises(InvalidDataException):
        _ = ConvND(Identity(), 2, 2, 5)

    with pytest.raises(InvalidDataException):
        _ = ConvND(Identity(), 2, 2, (5,))


    # Inference mode grad variables
    layer = ConvND(Identity(), 2, 2, (2, 10, 16, 24, 22), 2)
    data_in = np.zeros((2, 10, 16, 24, 22))
    _ = layer(data_in)
    data_out = np.zeros((2, 5, 8, 12, 11))
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
    layer = ConvND(Identity(), 2, 2, (2, 20, 20, 20), 2)
    data_in = np.zeros((2, 20, 20, 20))
    assert layer(data_in).shape == (2, 10, 10, 10)

    layer = ConvND(Identity(), 4, 3, (2, 20, 20, 20, 10), 1)
    data_in = np.zeros((2, 20, 20, 20, 10))
    assert layer(data_in).shape == (4, 18, 18, 18, 8)

    layer = ConvND(Identity(), 4, 3, (2, 20, 20), 1, 2)
    data_in = np.zeros((2, 20, 20))
    assert layer(data_in).shape == (4, 20, 20)

    layer = ConvND(Identity(), 1, 2, (2, 20, 13, 9, 15), 2, 3)
    data_in = np.zeros((2, 20, 13, 9, 15))
    assert layer(data_in).shape == (1, 11, 8, 6, 9)

    layer = ConvND(Identity(), 3, 3, (2, 20), 2)
    data_in = np.zeros((2, 20))
    assert layer(data_in).shape == (3, 9)

    layer = ConvND(Identity(), 3, (4, 2, 3), (2, 20, 20, 21), (4, 2, 1), (0, 4, 1))
    data_in = np.zeros((2, 20, 20, 21))
    assert layer(data_in).shape == (3, 5, 12, 20)


    # Backward sizes
    layer = ConvND(Identity(), 3, 2, (2, 20, 26, 28, 31), 2)
    data_in = np.zeros((2, 20, 26, 28, 31))
    data_out = np.zeros((3, 10, 13, 14, 15))
    _ = layer(data_in, True)
    assert layer.backward(data_out).shape == (2, 20, 26, 28, 31)

    layer = ConvND(Identity(), 5, 3, (2, 20, 10), 1)
    data_in = np.zeros((2, 20, 10))
    data_out = np.zeros((5, 18, 8))
    _ = layer(data_in, True)
    assert layer.backward(data_out).shape == (2, 20, 10)

    layer = ConvND(Identity(), 4, 2, (2, 20, 20, 20, 36), 2, 4)
    data_in = np.zeros((2, 20, 20, 20, 36))
    data_out = np.zeros((4, 12, 12, 12, 20))
    _ = layer(data_in, True)
    assert layer.backward(data_out).shape == (2, 20, 20, 20, 36)  

    layer = ConvND(Identity(), 4, 3, (2, 20, 15, 18), (2, 3, 3), (3, 0, 3))
    data_in = np.zeros((2, 20, 15, 18))
    data_out = np.zeros((4, 11, 5, 7))
    _ = layer(data_in, True)
    assert layer.backward(data_out).shape == (2, 20, 15, 18) 


    # Type failure checking
    with pytest.raises(InvalidDataException):
        layer = ConvND(Identity(), 1.1, 2, (2, 10, 10, 10))
    
    with pytest.raises(InvalidDataException):
        layer = ConvND(Identity(), 2, 2, "test")

    with pytest.raises(InvalidDataException):
        layer = ConvND(Identity(), 2, 3, (2, 10, 10, 10), "c")

    with pytest.raises(InvalidDataException):
        layer = ConvND(Identity(), 2, 2, (2, 10, 10, 10), 0)

    with pytest.raises(InvalidDataException):
        layer = ConvND(Identity(), 2, 2, (2, 10, 10, 10), 1, -1)

    with pytest.raises(InvalidDataException):
        layer = ConvND(None, 2, 2, (2, 10, 10, 10))

    with pytest.raises(InvalidDataException):
        layer = ConvND(Identity(), 2, 2, (2, -10, 10, 10))

    with pytest.raises(InvalidDataException):
        layer = ConvND(Identity(), 2, (2, 0, 2), (2, 10, 10, 10))

    with pytest.raises(InvalidDataException):
        layer = ConvND(Identity(), -1, (2, 1, 1), (2, 10, 10, 10))

    with pytest.raises(InvalidDataException):
        layer = ConvND(Identity(), 2, 2, (2, 10, 10, 10), (2, 0, 2))

    with pytest.raises(InvalidDataException):
        layer = ConvND(Identity(), 2, 2, (2, 10, 10, 10), (2, 2))

    with pytest.raises(InvalidDataException):
        layer = ConvND(Identity(), 2, 2, (2, 10, 10, 10), 2, (2, 3))

    with pytest.raises(InvalidDataException):
        layer = ConvND(Identity(), 2, (3, 2), (2, 10, 10, 10), 2)

    with pytest.raises(InvalidDataException):
        layer = ConvND(Identity(), "a", (3, 2, 2), (2, 10, 10, 10), 2)


    # Incorrect shape tests
    layer = ConvND(Identity(), 3, 2, (2, 20, 20, 20, 20), 2)
    data_in = np.zeros((2, 20, 20, 20, 20))
    data_false_in = np.zeros((4, 20, 19, 20, 20))
    data_false_out = np.zeros((4, 11, 10, 10, 10))
    with pytest.raises(UnsafeMemoryAccessException):
        _ = layer(data_false_in)

    _ = layer(data_in, True)
    with pytest.raises(UnsafeMemoryAccessException):
        _ = layer.backward(data_false_out)


    # Saving + Loading
    layer = ConvND(Identity(), 10, (2, 3, 4, 3), (2, 20, 18, 16, 22), (2, 3, 4, 1), (4, 6, 4, 0))
    l_save = layer.save_to_file()
    load_layer = ConvND.from_save(l_save)
    assert np.allclose(load_layer.kernel_weights, layer.kernel_weights)
    assert np.allclose(load_layer.bias_weights, layer.bias_weights)
    assert load_layer.kernel_size == layer.kernel_size
    assert load_layer.padding_all == layer.padding_all
    assert load_layer.pad_details == layer.pad_details
    assert load_layer.strides_all == layer.strides_all
    assert load_layer.use_bias == layer.use_bias
    assert load_layer.input_length == layer.input_length
    assert load_layer.in_size == layer.in_size
    assert load_layer.out_size == layer.out_size


    # Deepcopy
    layer2 = layer.deepcopy()
    assert layer2 is not layer
    assert np.allclose(layer2.kernel_weights, layer.kernel_weights)
    assert np.allclose(layer2.bias_weights, layer.bias_weights)
    assert layer2.kernel_size == layer.kernel_size
    assert layer2.padding_all == layer.padding_all
    assert layer2.pad_details == layer.pad_details
    assert layer2.strides_all == layer.strides_all
    assert layer2.use_bias == layer.use_bias
    assert layer2.input_length == layer.input_length
    assert layer2.in_size == layer.in_size
    assert layer2.out_size == layer.out_size
    
    
    # From-Kernel initialization tests
    kernel = np.zeros((5, 2, 4, 4, 3, 3))
    biases = np.zeros((5, 6, 6, 7, 6))
    layer = ConvND.from_kernel(Identity(), (2, 20, 20, 18, 18), kernel, (4, 4, 3, 3), (4, 4, 3, 0), biases)
    assert layer.out_size == (5, 6, 6, 7, 6)

    with pytest.raises(InvalidDataException):
        _ = ConvND.from_kernel(Identity(), (2, 20, 20, 20), "test")

    with pytest.raises(InvalidDataException):
        _ = ConvND.from_kernel(Identity(), (2, 20, 20, 20), kernel, 2, 4, "test")

    with pytest.raises(InvalidDataException):
        _ = ConvND.from_kernel(Identity(), (2, 20, -20, 20), kernel, 2, 4)

    with pytest.raises(InvalidDataException):
        _ = ConvND.from_kernel(Identity(), (2, 20, 20, 20), kernel, 0, 4)

    with pytest.raises(InvalidDataException):
        _ = ConvND.from_kernel(Identity(), (2, 20, 20, 20), kernel, 2, -1)

    with pytest.raises(InvalidDataException):
        _ = ConvND.from_kernel(Identity(), (2, 20, 20, 20), kernel, (2, 0, 2), 4)

    with pytest.raises(InvalidDataException):
        _ = ConvND.from_kernel(Identity(), (2, 20, 20, 20), kernel, 2, (0, -1, 0))

    with pytest.raises(InvalidDataException):
        _ = ConvND.from_kernel(Identity(), (2, 20, 20, 20), kernel, 2, (0, 0))

    with pytest.raises(InvalidDataException):
        _ = ConvND.from_kernel(Identity(), (2,), kernel, 2, 0)

    with pytest.raises(InvalidDataException):
        _ = ConvND.from_kernel(Identity(), 2, kernel, 2, 0)

    with pytest.raises(InvalidDataException):
        _ = ConvND.from_kernel(Identity(), (2, 20, 20, 20), kernel, (2, 2), 0)

    with pytest.raises(InvalidDataException):
        _ = ConvND.from_kernel(None, (2, 20, 20, 20), kernel, 2, 0)




def test_conv1d_transpose():
    # Incorrect sizes
    with pytest.raises(InvalidDataException):
        _ = Conv1DTranspose(Identity(), 2, 2, 5)

    with pytest.raises(InvalidDataException):
        _ = Conv1DTranspose(Identity(), 2, 2, (5,))


    # Inference mode grad variables
    layer = Conv1DTranspose(Identity(), 2, 2, (2, 10), 2)
    data_in = np.zeros((2, 10))
    _ = layer(data_in)
    data_out = np.zeros((2, 20))
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
    layer = Conv1DTranspose(Identity(), 2, 2, (2, 20), 2)
    data_in = np.zeros((2, 20))
    assert layer(data_in).shape == (2, 40)

    layer = Conv1DTranspose(Identity(), 4, 3, (2, 20), 1)
    data_in = np.zeros((2, 20))
    assert layer(data_in).shape == (4, 22)

    layer = Conv1DTranspose(Identity(), 4, 3, (2, 20), 1, 2)
    data_in = np.zeros((2, 20))
    assert layer(data_in).shape == (4, 20)

    layer = Conv1DTranspose(Identity(), 1, 2, (2, 20), 2, 3)
    data_in = np.zeros((2, 20))
    assert layer(data_in).shape == (1, 37)

    layer = Conv1DTranspose(Identity(), 3, 3, (2, 20), 2)
    data_in = np.zeros((2, 20))
    assert layer(data_in).shape == (3, 41)

    layer = Conv1DTranspose(Identity(), 1, 3, (2, 20), 2, 0, 4)
    data_in = np.zeros((2, 20))
    assert layer(data_in).shape == (1, 45)


    # Backward sizes
    layer = Conv1DTranspose(Identity(), 3, 2, (2, 20), 2)
    data_in = np.zeros((2, 20))
    data_out = np.zeros((3, 40))
    _ = layer(data_in, True)
    assert layer.backward(data_out).shape == (2, 20)

    layer = Conv1DTranspose(Identity(), 5, 3, (2, 20), 1)
    data_in = np.zeros((2, 20))
    data_out = np.zeros((5, 22))
    _ = layer(data_in, True)
    assert layer.backward(data_out).shape == (2, 20)

    layer = Conv1DTranspose(Identity(), 4, 2, (2, 20), 2, 4)
    data_in = np.zeros((2, 20))
    data_out = np.zeros((4, 36))
    _ = layer(data_in, True)
    assert layer.backward(data_out).shape == (2, 20)  

    layer = Conv1DTranspose(Identity(), 4, 3, (2, 20), 2, 3)
    data_in = np.zeros((2, 20))
    data_out = np.zeros((4, 38))
    _ = layer(data_in, True)
    assert layer.backward(data_out).shape == (2, 20)  

    layer = Conv1DTranspose(Identity(), 1, 4, (2, 20), 2, 0, 5)
    data_in = np.zeros((2, 20))
    data_out = np.zeros((1, 47))
    _ = layer(data_in, True)
    assert layer.backward(data_out).shape == (2, 20)  


    # Type failure checking
    with pytest.raises(InvalidDataException):
        layer = Conv1DTranspose(Identity(), 1.1, 2, (2, 10))

    with pytest.raises(InvalidDataException):
        layer = Conv1DTranspose(Identity(), -1, 2, (2, 10))
    
    with pytest.raises(InvalidDataException):
        layer = Conv1DTranspose(Identity(), 2, 2, "test")

    with pytest.raises(InvalidDataException):
        layer = Conv1DTranspose(Identity(), 2, 3, (2, 10), "c")

    with pytest.raises(InvalidDataException):
        layer = Conv1DTranspose(Identity(), 2, 2, (2, 10), 0)

    with pytest.raises(InvalidDataException):
        layer = Conv1DTranspose(Identity(), 2, 2, (2, 10), 1, -1)

    with pytest.raises(InvalidDataException):
        layer = Conv1DTranspose(Identity(), 2, 2, (2, 10), 1, 0, -1)

    with pytest.raises(InvalidDataException):
        layer = Conv1DTranspose(None, 2, 2, (2, 10))

    with pytest.raises(InvalidDataException):
        layer = Conv1DTranspose(Identity(), 2, 2, (2, -10))


    # Incorrect shape tests
    layer = Conv1DTranspose(Identity(), 3, 2, (2, 20), 2)
    data_in = np.zeros((2, 20))
    data_false_in = np.zeros((4, 20))
    data_false_out = np.zeros((3, 41))
    with pytest.raises(UnsafeMemoryAccessException):
        _ = layer(data_false_in)

    _ = layer(data_in, True)
    with pytest.raises(UnsafeMemoryAccessException):
        _ = layer.backward(data_false_out)


    # Saving + Loading
    layer = Conv1DTranspose(Identity(), 2, 3, (2, 20), 2, 3, 1)
    l_save = layer.save_to_file()
    load_layer = Conv1DTranspose.from_save(l_save)
    assert np.allclose(load_layer.kernel_weights, layer.kernel_weights)
    assert np.allclose(load_layer.bias_weights, layer.bias_weights)
    assert load_layer.kernel_size == layer.kernel_size
    assert load_layer.padding_all == layer.padding_all
    assert load_layer.out_padding_all == layer.out_padding_all
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
    assert layer2.out_padding_all == layer.out_padding_all
    assert layer2.strides == layer.strides
    assert layer2.in_size == layer.in_size
    assert layer2.out_size == layer.out_size


    # From-Kernel initialization tests
    kernel = np.zeros((5, 2, 2))
    biases = np.zeros((5, 36))
    layer = Conv1DTranspose.from_kernel(Identity(), (2, 20), kernel, 2, 4, 0, biases)
    assert layer.out_size == (5, 36)

    with pytest.raises(InvalidDataException):
        _ = Conv1DTranspose.from_kernel(Identity(), (2, 20), "test")

    with pytest.raises(InvalidDataException):
        _ = Conv1DTranspose.from_kernel(Identity(), (2, 20), kernel, 2, 4, 0, "test")

    with pytest.raises(InvalidDataException):
        _ = Conv1DTranspose.from_kernel(Identity(), (2, 20), kernel, 0, 4, 0, biases)

    with pytest.raises(InvalidDataException):
        _ = Conv1DTranspose.from_kernel(Identity(), (2, 20), kernel, 2, -1, 0, biases)

    with pytest.raises(InvalidDataException):
        _ = Conv1DTranspose.from_kernel(Identity(), (2, -20), kernel, 2, 4, 0, biases)

    with pytest.raises(InvalidDataException):
        _ = Conv1DTranspose.from_kernel(Identity(), (2, 20), kernel, 1, 4, -1, biases)




def test_conv2d_transpose():
    # Incorrect sizes
    with pytest.raises(InvalidDataException):
        _ = Conv2DTranspose(Identity(), 2, 2, 5)

    with pytest.raises(InvalidDataException):
        _ = Conv2DTranspose(Identity(), 2, 2, (5, 5))


    # Inference mode grad variables
    layer = Conv2DTranspose(Identity(), 2, 2, (2, 10, 16), 2)
    data_in = np.zeros((2, 10, 16))
    _ = layer(data_in)
    data_out = np.zeros((2, 20, 32))
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
    layer = Conv2DTranspose(Identity(), 2, 2, (2, 20, 20), 2)
    data_in = np.zeros((2, 20, 20))
    assert layer(data_in).shape == (2, 40, 40)

    layer = Conv2DTranspose(Identity(), 4, 3, (2, 20, 20), 1)
    data_in = np.zeros((2, 20, 20))
    assert layer(data_in).shape == (4, 22, 22)

    layer = Conv2DTranspose(Identity(), 4, 3, (2, 20, 20), 1, 2)
    data_in = np.zeros((2, 20, 20))
    assert layer(data_in).shape == (4, 20, 20)

    layer = Conv2DTranspose(Identity(), 1, 2, (2, 20, 13), 2, 3)
    data_in = np.zeros((2, 20, 13))
    assert layer(data_in).shape == (1, 37, 23)

    layer = Conv2DTranspose(Identity(), 3, 3, (2, 20, 20), 2)
    data_in = np.zeros((2, 20, 20))
    assert layer(data_in).shape == (3, 41, 41)

    layer = Conv2DTranspose(Identity(), 3, (4, 2), (2, 20, 20), (4, 2))
    data_in = np.zeros((2, 20, 20))
    assert layer(data_in).shape == (3, 80, 40)

    layer = Conv2DTranspose(Identity(), 3, 3, (2, 20, 20), 2, 0, 5)
    data_in = np.zeros((2, 20, 20))
    assert layer(data_in).shape == (3, 46, 46)

    layer = Conv2DTranspose(Identity(), 3, 3, (2, 20, 20), 2, 0, (0, 3))
    data_in = np.zeros((2, 20, 20))
    assert layer(data_in).shape == (3, 41, 44)


    # Backward sizes
    layer = Conv2DTranspose(Identity(), 3, 2, (2, 20, 26), 2)
    data_in = np.zeros((2, 20, 26))
    data_out = np.zeros((3, 40, 52))
    _ = layer(data_in, True)
    assert layer.backward(data_out).shape == (2, 20, 26)

    layer = Conv2DTranspose(Identity(), 5, 3, (2, 20, 10), 1)
    data_in = np.zeros((2, 20, 10))
    data_out = np.zeros((5, 22, 12))
    _ = layer(data_in, True)
    assert layer.backward(data_out).shape == (2, 20, 10)

    layer = Conv2DTranspose(Identity(), 4, 2, (2, 20, 20), 2, 4)
    data_in = np.zeros((2, 20, 20))
    data_out = np.zeros((4, 36, 36))
    _ = layer(data_in, True)
    assert layer.backward(data_out).shape == (2, 20, 20)  

    layer = Conv2DTranspose(Identity(), 4, 3, (2, 20, 15), (2, 3), (3, 0))
    data_in = np.zeros((2, 20, 15))
    data_out = np.zeros((4, 38, 45))
    _ = layer(data_in, True)
    assert layer.backward(data_out).shape == (2, 20, 15)  

    layer = Conv2DTranspose(Identity(), 4, 3, (2, 20, 25), 2, 0, 3)
    data_in = np.zeros((2, 20, 25))
    data_out = np.zeros((4, 44, 54))
    _ = layer(data_in, True)
    assert layer.backward(data_out).shape == (2, 20, 25)  


    # Type failure checking
    with pytest.raises(InvalidDataException):
        layer = Conv2DTranspose(Identity(), 1.1, 2, (2, 10, 10))
    
    with pytest.raises(InvalidDataException):
        layer = Conv2DTranspose(Identity(), 2, 2, "test")

    with pytest.raises(InvalidDataException):
        layer = Conv2DTranspose(Identity(), 2, 3, (2, 10, 10), "c")

    with pytest.raises(InvalidDataException):
        layer = Conv2DTranspose(Identity(), 2, 2, (2, 10, 10), 0)

    with pytest.raises(InvalidDataException):
        layer = Conv2DTranspose(Identity(), 2, 2, (2, 10, 10), 1, -1)

    with pytest.raises(InvalidDataException):
        layer = Conv2DTranspose(None, 2, 2, (2, 10, 10))

    with pytest.raises(InvalidDataException):
        layer = Conv2DTranspose(Identity(), 2, 2, (2, -10, 10))

    with pytest.raises(InvalidDataException):
        layer = Conv2DTranspose(Identity(), 2, (2, 0), (2, 10, 10))

    with pytest.raises(InvalidDataException):
        layer = Conv2DTranspose(Identity(), -1, (2, 1), (2, 10, 10))

    with pytest.raises(InvalidDataException):
        layer = Conv2DTranspose(Identity(), 2, (2,), (2, 10, 10))

    with pytest.raises(InvalidDataException):
        layer = Conv2DTranspose(Identity(), 2, 2, (2, 10, 10), (2,))

    with pytest.raises(InvalidDataException):
        layer = Conv2DTranspose(Identity(), 2, 2, (2, 10, 10), "test")

    with pytest.raises(InvalidDataException):
        layer = Conv2DTranspose(Identity(), 2, 2, (2, 10, 10), 2, (0,))

    with pytest.raises(InvalidDataException):
        layer = Conv2DTranspose(Identity(), 2, 2, (2, 10, 10), 2, "test")

    with pytest.raises(InvalidDataException):
        layer = Conv2DTranspose(Identity(), 2, 2, (2, 10, 10), (2, 0))

    with pytest.raises(InvalidDataException):
        layer = Conv2DTranspose(Identity(), 2, 2, (2, 10, 10), 1, 0, -1)

    with pytest.raises(InvalidDataException):
        layer = Conv2DTranspose(Identity(), 2, 2, (2, 10, 10), 1, 0, (0, -1))

    with pytest.raises(InvalidDataException):
        layer = Conv2DTranspose(Identity(), 2, 2, (2, 10, 10), 1, 0, (1,))


    # Incorrect shape tests
    layer = Conv2DTranspose(Identity(), 3, 2, (2, 20, 20), 2)
    data_in = np.zeros((2, 20, 20))
    data_false_in = np.zeros((3, 20, 19))
    data_false_out = np.zeros((3, 41, 40))
    with pytest.raises(UnsafeMemoryAccessException):
        _ = layer(data_false_in)

    _ = layer(data_in, True)
    with pytest.raises(UnsafeMemoryAccessException):
        _ = layer.backward(data_false_out)


    # Saving + Loading
    layer = Conv2DTranspose(Identity(), 2, (2, 3), (2, 20, 18), (2, 3), (4, 6), (1, 0))
    l_save = layer.save_to_file()
    load_layer = Conv2DTranspose.from_save(l_save)
    assert np.allclose(load_layer.kernel_weights, layer.kernel_weights)
    assert np.allclose(load_layer.bias_weights, layer.bias_weights)
    assert load_layer.kernel_height == layer.kernel_height
    assert load_layer.kernel_width == layer.kernel_width
    assert load_layer.padding_all == layer.padding_all
    assert load_layer.out_padding_all == layer.out_padding_all
    assert load_layer.stride_h == layer.stride_h
    assert load_layer.stride_w == layer.stride_w
    assert load_layer.in_size == layer.in_size
    assert load_layer.out_size == layer.out_size


    # Deepcopy
    layer2 = layer.deepcopy()
    assert layer2 is not layer
    assert np.allclose(layer2.kernel_weights, layer.kernel_weights)
    assert np.allclose(layer2.bias_weights, layer.bias_weights)
    assert layer2.kernel_height == layer.kernel_height
    assert layer2.kernel_width == layer.kernel_width
    assert layer2.padding_all == layer.padding_all
    assert layer2.out_padding_all == layer.out_padding_all
    assert layer2.stride_h == layer.stride_h
    assert layer2.stride_w == layer.stride_w
    assert layer2.in_size == layer.in_size
    assert layer2.out_size == layer.out_size


    # From-Kernel initialization tests
    kernel = np.zeros((5, 2, 2, 2))
    biases = np.zeros((5, 36, 36))
    layer = Conv2DTranspose.from_kernel(Identity(), (2, 20, 20), kernel, 2, 4, 0, biases)
    assert layer.out_size == (5, 36, 36)

    with pytest.raises(InvalidDataException):
        _ = Conv2DTranspose.from_kernel(Identity(), (2, 20, 20), "test")

    with pytest.raises(InvalidDataException):
        _ = Conv2DTranspose.from_kernel(Identity(), (2, 20, 20), kernel, 2, 4, 0, "test")

    with pytest.raises(InvalidDataException):
        _ = Conv2DTranspose.from_kernel(Identity(), (2, 20, -20), kernel, 2, 4, 0, biases)

    with pytest.raises(InvalidDataException):
        _ = Conv2DTranspose.from_kernel(Identity(), (2, 20, 20), kernel, 0, 4, 0, biases)

    with pytest.raises(InvalidDataException):
        _ = Conv2DTranspose.from_kernel(Identity(), (2, 20, 20), kernel, 2, -1, 0, biases)

    with pytest.raises(InvalidDataException):
        _ = Conv2DTranspose.from_kernel(Identity(), (2, 20, 20), kernel, (2, 0), 4, 0, biases)

    with pytest.raises(InvalidDataException):
        _ = Conv2DTranspose.from_kernel(Identity(), (2, 20, 20), kernel, 2, (0, -1), 0, biases)

    with pytest.raises(InvalidDataException):
        _ = Conv2DTranspose.from_kernel(Identity(), (2, 20, 20), kernel, 2, 4, -1, biases)

    with pytest.raises(InvalidDataException):
        _ = Conv2DTranspose.from_kernel(Identity(), (2, 20, 20), kernel, 2, 4, (0, -1), biases)




def test_conv3d_transpose():
    # Incorrect sizes
    with pytest.raises(InvalidDataException):
        _ = Conv3DTranspose(Identity(), 2, 2, 5)

    with pytest.raises(InvalidDataException):
        _ = Conv3DTranspose(Identity(), 2, 2, (5, 5, 5))


    # Inference mode grad variables
    layer = Conv3DTranspose(Identity(), 2, 2, (2, 10, 16, 24), 2)
    data_in = np.zeros((2, 10, 16, 24))
    _ = layer(data_in)
    data_out = np.zeros((2, 20, 32, 48))
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
    layer = Conv3DTranspose(Identity(), 2, 2, (2, 20, 20, 20), 2)
    data_in = np.zeros((2, 20, 20, 20))
    assert layer(data_in).shape == (2, 40, 40, 40)

    layer = Conv3DTranspose(Identity(), 4, 3, (2, 20, 20, 20), 1)
    data_in = np.zeros((2, 20, 20, 20))
    assert layer(data_in).shape == (4, 22, 22, 22)

    layer = Conv3DTranspose(Identity(), 4, 3, (2, 20, 20, 20), 1, 2)
    data_in = np.zeros((2, 20, 20, 20))
    assert layer(data_in).shape == (4, 20, 20, 20)

    layer = Conv3DTranspose(Identity(), 1, 2, (2, 20, 13, 9), 2, 3)
    data_in = np.zeros((2, 20, 13, 9))
    assert layer(data_in).shape == (1, 37, 23, 15)

    layer = Conv3DTranspose(Identity(), 3, 3, (2, 20, 20, 20), 2)
    data_in = np.zeros((2, 20, 20, 20))
    assert layer(data_in).shape == (3, 41, 41, 41)

    layer = Conv3DTranspose(Identity(), 3, (4, 2, 3), (2, 20, 20, 21), (4, 2, 1), (0, 4, 1))
    data_in = np.zeros((2, 20, 20, 21))
    assert layer(data_in).shape == (3, 80, 36, 22)

    layer = Conv3DTranspose(Identity(), 3, 3, (2, 20, 20, 20), 2, 1, 3)
    data_in = np.zeros((2, 20, 20, 20))
    assert layer(data_in).shape == (3, 43, 43, 43)

    layer = Conv3DTranspose(Identity(), 2, 2, (2, 20, 20, 20), 2, 0, (3, 4, 1))
    data_in = np.zeros((2, 20, 20, 20))
    assert layer(data_in).shape == (2, 43, 44, 41)


    # Backward sizes
    layer = Conv3DTranspose(Identity(), 3, 2, (2, 20, 26, 28), 2)
    data_in = np.zeros((2, 20, 26, 28))
    data_out = np.zeros((3, 40, 52, 56))
    _ = layer(data_in, True)
    assert layer.backward(data_out).shape == (2, 20, 26, 28)

    layer = Conv3DTranspose(Identity(), 5, 3, (2, 20, 10, 15), 1)
    data_in = np.zeros((2, 20, 10, 15))
    data_out = np.zeros((5, 22, 12, 17))
    _ = layer(data_in, True)
    assert layer.backward(data_out).shape == (2, 20, 10, 15)

    layer = Conv3DTranspose(Identity(), 4, 2, (2, 20, 20, 20), 2, 4)
    data_in = np.zeros((2, 20, 20, 20))
    data_out = np.zeros((4, 36, 36, 36))
    _ = layer(data_in, True)
    assert layer.backward(data_out).shape == (2, 20, 20, 20)  

    layer = Conv3DTranspose(Identity(), 4, 3, (2, 20, 15, 18), (2, 3, 3), (3, 0, 3))
    data_in = np.zeros((2, 20, 15, 18))
    data_out = np.zeros((4, 38, 45, 51))
    _ = layer(data_in, True)
    assert layer.backward(data_out).shape == (2, 20, 15, 18)  


    # Type failure checking
    with pytest.raises(InvalidDataException):
        layer = Conv3DTranspose(Identity(), 1.1, 2, (2, 10, 10, 10))
    
    with pytest.raises(InvalidDataException):
        layer = Conv3DTranspose(Identity(), 2, 2, "test")

    with pytest.raises(InvalidDataException):
        layer = Conv3DTranspose(Identity(), 2, 3, (2, 10, 10, 10), "c")

    with pytest.raises(InvalidDataException):
        layer = Conv3DTranspose(Identity(), 2, 2, (2, 10, 10, 10), 0)

    with pytest.raises(InvalidDataException):
        layer = Conv3DTranspose(Identity(), 2, 2, (2, 10, 10, 10), 1, -1)

    with pytest.raises(InvalidDataException):
        layer = Conv3DTranspose(None, 2, 2, (2, 10, 10, 10))

    with pytest.raises(InvalidDataException):
        layer = Conv3DTranspose(Identity(), 2, 2, (2, -10, 10, 10))

    with pytest.raises(InvalidDataException):
        layer = Conv3DTranspose(Identity(), 2, (2, 0, 2), (2, 10, 10, 10))

    with pytest.raises(InvalidDataException):
        layer = Conv3DTranspose(Identity(), -1, (2, 1, 1), (2, 10, 10, 10))

    with pytest.raises(InvalidDataException):
        layer = Conv3DTranspose(Identity(), 2, 2, (2, 10, 10, 10), (2, 0, 2))

    with pytest.raises(InvalidDataException):
        layer = Conv3DTranspose(Identity(), 2, 2, (2, 10, 10, 10), (2, 2))

    with pytest.raises(InvalidDataException):
        layer = Conv3DTranspose(Identity(), 2, 2, (2, 10, 10, 10), 2, (2, 3))

    with pytest.raises(InvalidDataException):
        layer = Conv3DTranspose(Identity(), 2, 2, (2, 10, 10, 10), 2, (2, 3, 3), -1)

    with pytest.raises(InvalidDataException):
        layer = Conv3DTranspose(Identity(), 2, 2, (2, 10, 10, 10), 2, (2, 3, 3), (1, 1))

    with pytest.raises(InvalidDataException):
        layer = Conv3DTranspose(Identity(), 2, (3, 2), (2, 10, 10, 10), 2)

    with pytest.raises(InvalidDataException):
        layer = Conv3DTranspose(Identity(), "a", (3, 2, 2), (2, 10, 10, 10), 2)

    with pytest.raises(InvalidDataException):
        layer = Conv3DTranspose(Identity(), 2, "test", (2, 10, 10, 10), 2)

    with pytest.raises(InvalidDataException):
        layer = Conv3DTranspose(Identity(), 2, (3, 2, 2), (2, 10, 10, 10), 2, "test")

    with pytest.raises(InvalidDataException):
        layer = Conv3DTranspose(Identity(), 2, (3, 2, 2), (2, 10, 10, 10), 2, 0, "test")


    # Incorrect shape tests
    layer = Conv3DTranspose(Identity(), 3, 2, (2, 20, 20, 20), 2)
    data_in = np.zeros((2, 20, 20, 20))
    data_false_in = np.zeros((3, 20, 19, 20))
    data_false_out = np.zeros((3, 41, 40, 40))
    with pytest.raises(UnsafeMemoryAccessException):
        _ = layer(data_false_in)

    _ = layer(data_in, True)
    with pytest.raises(UnsafeMemoryAccessException):
        _ = layer.backward(data_false_out)


    # Saving + Loading
    layer = Conv3DTranspose(Identity(), 10, (2, 3, 4), (2, 20, 18, 16), (2, 3, 4), (4, 6, 4), (0, 2, 4))
    l_save = layer.save_to_file()
    load_layer = Conv3DTranspose.from_save(l_save)
    assert np.allclose(load_layer.kernel_weights, layer.kernel_weights)
    assert np.allclose(load_layer.bias_weights, layer.bias_weights)
    assert load_layer.kernel_height == layer.kernel_height
    assert load_layer.kernel_width == layer.kernel_width
    assert load_layer.kernel_depth == layer.kernel_depth
    assert load_layer.padding_all == layer.padding_all
    assert load_layer.out_padding_all == layer.out_padding_all
    assert load_layer.stride_h == layer.stride_h
    assert load_layer.stride_w == layer.stride_w
    assert load_layer.stride_d == layer.stride_d
    assert load_layer.in_size == layer.in_size
    assert load_layer.out_size == layer.out_size


    # Deepcopy
    layer2 = layer.deepcopy()
    assert layer2 is not layer
    assert np.allclose(layer2.kernel_weights, layer.kernel_weights)
    assert np.allclose(layer2.bias_weights, layer.bias_weights)
    assert layer2.kernel_height == layer.kernel_height
    assert layer2.kernel_width == layer.kernel_width
    assert layer2.kernel_depth == layer.kernel_depth
    assert layer2.padding_all == layer.padding_all
    assert layer2.out_padding_all == layer.out_padding_all
    assert layer2.stride_h == layer.stride_h
    assert layer2.stride_w == layer.stride_w
    assert layer2.stride_d == layer.stride_d
    assert layer2.in_size == layer.in_size
    assert layer2.out_size == layer.out_size


    # From-Kernel initialization tests
    kernel = np.zeros((5, 2, 4, 4, 3))
    biases = np.zeros((5, 37, 37, 52))
    layer = Conv3DTranspose.from_kernel(Identity(), (2, 10, 10, 18), kernel, (4, 4, 3), (4, 4, 3), 1, biases)
    assert layer.out_size == (5, 37, 37, 52)

    with pytest.raises(InvalidDataException):
        _ = Conv3DTranspose.from_kernel(Identity(), (2, 20, 20, 20), "test")

    with pytest.raises(InvalidDataException):
        _ = Conv3DTranspose.from_kernel(Identity(), (2, 20, 20, 20), kernel, 2, 4, 0, "test")

    with pytest.raises(InvalidDataException):
        _ = Conv3DTranspose.from_kernel(Identity(), "test", kernel, (2, 2, 3), 0, 0, biases)

    with pytest.raises(InvalidDataException):
        _ = Conv3DTranspose.from_kernel(Identity(), (2, 20, 20, 20), kernel, (2, 2, 2), "test", 0, biases)

    with pytest.raises(InvalidDataException):
        _ = Conv3DTranspose.from_kernel(Identity(), (2, 20, 20, 20), kernel, (2, 2, 2), 0, "test", biases)

    with pytest.raises(InvalidDataException):
        _ = Conv3DTranspose.from_kernel(Identity(), (2, 20, -20, 20), kernel, 2, 4, 0, biases)

    with pytest.raises(InvalidDataException):
        _ = Conv3DTranspose.from_kernel(Identity(), (2, 20, 20, 20), kernel, 0, 4, 0, biases)

    with pytest.raises(InvalidDataException):
        _ = Conv3DTranspose.from_kernel(Identity(), (2, 20, 20, 20), kernel, 2, -1, 0, biases)

    with pytest.raises(InvalidDataException):
        _ = Conv3DTranspose.from_kernel(Identity(), (2, 20, 20, 20), kernel, (2, 0, 2), 4, 0, biases)

    with pytest.raises(InvalidDataException):
        _ = Conv3DTranspose.from_kernel(Identity(), (2, 20, 20, 20), kernel, 2, (0, -1, 0), 0, biases)

    with pytest.raises(InvalidDataException):
        _ = Conv3DTranspose.from_kernel(Identity(), (2, 20, 20, 20), kernel, 2, (0, 0), 0, biases)

    with pytest.raises(InvalidDataException):
        _ = Conv3DTranspose.from_kernel(Identity(), (2, 20, 20), kernel, 2, (0, 1, 0), 0, biases)

    with pytest.raises(InvalidDataException):
        _ = Conv3DTranspose.from_kernel(Identity(), (2, 20, 20, 20), kernel, (2, 2), 0, 0, biases)

    with pytest.raises(InvalidDataException):
        _ = Conv3DTranspose.from_kernel(None, (2, 20, 20, 20), kernel, 2, 0, 0, biases)

    with pytest.raises(InvalidDataException):
        _ = Conv3DTranspose.from_kernel(Identity(), (2, 20, 20, 20), kernel, (2, 2, 1), 0, -1, biases)

    with pytest.raises(InvalidDataException):
        _ = Conv3DTranspose.from_kernel(Identity(), (2, 20, 20, 20), kernel, (2, 2, 1), 0, (1, 1), biases)




def test_convNd_transpose():
    # Incorrect sizes
    with pytest.raises(InvalidDataException):
        _ = ConvNDTranspose(Identity(), 2, 2, 5)

    with pytest.raises(InvalidDataException):
        _ = ConvNDTranspose(Identity(), 2, 2, (5,))


    # Inference mode grad variables
    layer = ConvNDTranspose(Identity(), 2, 2, (2, 10, 16, 24, 20), 2)
    data_in = np.zeros((2, 10, 16, 24, 20))
    _ = layer(data_in)
    data_out = np.zeros((2, 20, 32, 48, 40))
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
    layer = ConvNDTranspose(Identity(), 2, 2, (2, 10, 10, 10, 5), 2)
    data_in = np.zeros((2, 10, 10, 10, 5))
    assert layer(data_in).shape == (2, 20, 20, 20, 10)

    layer = ConvNDTranspose(Identity(), 4, 3, (2, 20, 20, 20), 1)
    data_in = np.zeros((2, 20, 20, 20))
    assert layer(data_in).shape == (4, 22, 22, 22)

    layer = ConvNDTranspose(Identity(), 4, 3, (2, 20, 20), 1, 2)
    data_in = np.zeros((2, 20, 20))
    assert layer(data_in).shape == (4, 20, 20)

    layer = ConvNDTranspose(Identity(), 1, 2, (2, 20, 13, 9), 2, 3)
    data_in = np.zeros((2, 20, 13, 9))
    assert layer(data_in).shape == (1, 37, 23, 15)

    layer = ConvNDTranspose(Identity(), 3, 3, (2, 20, 20, 10, 10), 2)
    data_in = np.zeros((2, 20, 20, 10, 10))
    assert layer(data_in).shape == (3, 41, 41, 21, 21)

    layer = ConvNDTranspose(Identity(), 3, (4, 2, 3), (2, 20, 20, 21), (4, 2, 1), (0, 4, 1))
    data_in = np.zeros((2, 20, 20, 21))
    assert layer(data_in).shape == (3, 80, 36, 22)

    layer = ConvNDTranspose(Identity(), 3, 3, (2, 20), 2, 1, 3)
    data_in = np.zeros((2, 20))
    assert layer(data_in).shape == (3, 43)

    layer = ConvNDTranspose(Identity(), 2, 2, (2, 20, 20, 20), 2, 0, (3, 4, 1))
    data_in = np.zeros((2, 20, 20, 20))
    assert layer(data_in).shape == (2, 43, 44, 41)


    # Backward sizes
    layer = ConvNDTranspose(Identity(), 3, 2, (2, 10, 13, 14, 15), 2)
    data_in = np.zeros((2, 10, 13, 14, 15))
    data_out = np.zeros((3, 20, 26, 28, 30))
    _ = layer(data_in, True)
    assert layer.backward(data_out).shape == (2, 10, 13, 14, 15)

    layer = ConvNDTranspose(Identity(), 5, 3, (2, 20, 10, 15), 1)
    data_in = np.zeros((2, 20, 10, 15))
    data_out = np.zeros((5, 22, 12, 17))
    _ = layer(data_in, True)
    assert layer.backward(data_out).shape == (2, 20, 10, 15)

    layer = ConvNDTranspose(Identity(), 4, 2, (2, 20, 20), 2, 4)
    data_in = np.zeros((2, 20, 20))
    data_out = np.zeros((4, 36, 36))
    _ = layer(data_in, True)
    assert layer.backward(data_out).shape == (2, 20, 20)

    layer = ConvNDTranspose(Identity(), 4, 2, (2, 20), 2, 4, 5)
    data_in = np.zeros((2, 20))
    data_out = np.zeros((4, 41))
    _ = layer(data_in, True)
    assert layer.backward(data_out).shape == (2, 20)   

    layer = ConvNDTranspose(Identity(), 4, 3, (2, 20, 15, 18), (2, 3, 3), (3, 0, 3))
    data_in = np.zeros((2, 20, 15, 18))
    data_out = np.zeros((4, 38, 45, 51))
    _ = layer(data_in, True)
    assert layer.backward(data_out).shape == (2, 20, 15, 18) 


    # Type failure checking
    with pytest.raises(InvalidDataException):
        layer = ConvNDTranspose(Identity(), 1.1, 2, (2, 10, 10, 10))
    
    with pytest.raises(InvalidDataException):
        layer = ConvNDTranspose(Identity(), 2, 2, "test")

    with pytest.raises(InvalidDataException):
        layer = ConvNDTranspose(Identity(), 2, 3, (2, 10, 10, 10), "c")

    with pytest.raises(InvalidDataException):
        layer = ConvNDTranspose(Identity(), 2, 2, (2, 10, 10, 10), 0)

    with pytest.raises(InvalidDataException):
        layer = ConvNDTranspose(Identity(), 2, 2, (2, 10, 10, 10), 1, -1)

    with pytest.raises(InvalidDataException):
        layer = ConvNDTranspose(None, 2, 2, (2, 10, 10, 10))

    with pytest.raises(InvalidDataException):
        layer = ConvNDTranspose(Identity(), 2, 2, (2, -10, 10, 10))

    with pytest.raises(InvalidDataException):
        layer = ConvNDTranspose(Identity(), 2, (2, 0, 2), (2, 10, 10, 10))

    with pytest.raises(InvalidDataException):
        layer = ConvNDTranspose(Identity(), -1, (2, 1, 1), (2, 10, 10, 10))

    with pytest.raises(InvalidDataException):
        layer = ConvNDTranspose(Identity(), 2, 2, (2, 10, 10, 10), (2, 0, 2))

    with pytest.raises(InvalidDataException):
        layer = ConvNDTranspose(Identity(), 2, 2, (2, 10, 10, 10), (2, 2))

    with pytest.raises(InvalidDataException):
        layer = ConvNDTranspose(Identity(), 2, 2, (2, 10, 10, 10), 2, (2, 3))

    with pytest.raises(InvalidDataException):
        layer = ConvNDTranspose(Identity(), 2, 2, (2, 10, 10, 10), 2, (2, 3, 3), -1)

    with pytest.raises(InvalidDataException):
        layer = ConvNDTranspose(Identity(), 2, 2, (2, 10, 10, 10), 2, (2, 3, 3), (1, 1))

    with pytest.raises(InvalidDataException):
        layer = ConvNDTranspose(Identity(), 2, (3, 2), (2, 10, 10, 10), 2)

    with pytest.raises(InvalidDataException):
        layer = ConvNDTranspose(Identity(), "a", (3, 2, 2), (2, 10, 10, 10), 2)

    with pytest.raises(InvalidDataException):
        layer = ConvNDTranspose(Identity(), 2, "test", (2, 10, 10, 10), 2)

    with pytest.raises(InvalidDataException):
        layer = ConvNDTranspose(Identity(), 2, (3, 2, 2), (2, 10, 10, 10), 2, "test")

    with pytest.raises(InvalidDataException):
        layer = ConvNDTranspose(Identity(), 2, (3, 2, 2), (2, 10, 10, 10), 2, 0, "test")


    # Incorrect shape tests
    layer = ConvNDTranspose(Identity(), 3, 2, (2, 10, 10, 10, 5), 2)
    data_in = np.zeros((2, 10, 10, 10, 5))
    data_false_in = np.zeros((3, 10, 9, 10, 5))
    data_false_out = np.zeros((3, 21, 20, 20, 10))
    with pytest.raises(UnsafeMemoryAccessException):
        _ = layer(data_false_in)

    _ = layer(data_in, True)
    with pytest.raises(UnsafeMemoryAccessException):
        _ = layer.backward(data_false_out)


    # Saving + Loading
    layer = ConvNDTranspose(Identity(), 10, (2, 3, 4, 2), (2, 20, 18, 16, 24), (2, 3, 4, 1), (4, 6, 4, 0), (0, 2, 4, 2))
    l_save = layer.save_to_file()
    load_layer = ConvNDTranspose.from_save(l_save)
    assert np.allclose(load_layer.kernel_weights, layer.kernel_weights)
    assert np.allclose(load_layer.bias_weights, layer.bias_weights)
    assert load_layer.kernel_size == layer.kernel_size
    assert load_layer.padding_all == layer.padding_all
    assert load_layer.pad_details == layer.pad_details
    assert load_layer.out_padding_all == layer.out_padding_all
    assert load_layer.out_pad_details == layer.out_pad_details
    assert load_layer.strides_all == layer.strides_all
    assert load_layer.use_bias == layer.use_bias
    assert load_layer.input_length == layer.input_length
    assert load_layer.in_size == layer.in_size
    assert load_layer.out_size == layer.out_size


    # Deepcopy
    layer2 = layer.deepcopy()
    assert layer2 is not layer
    assert np.allclose(layer2.kernel_weights, layer.kernel_weights)
    assert np.allclose(layer2.bias_weights, layer.bias_weights)
    assert layer2.kernel_size == layer.kernel_size
    assert layer2.padding_all == layer.padding_all
    assert layer2.pad_details == layer.pad_details
    assert layer2.out_padding_all == layer.out_padding_all
    assert layer2.out_pad_details == layer.out_pad_details
    assert layer2.strides_all == layer.strides_all
    assert layer2.use_bias == layer.use_bias
    assert layer2.input_length == layer.input_length
    assert layer2.in_size == layer.in_size
    assert layer2.out_size == layer.out_size


    # From-Kernel initialization tests
    kernel = np.zeros((5, 2, 4, 4, 3, 3))
    biases = np.zeros((5, 37, 37, 54, 29))
    layer = ConvNDTranspose.from_kernel(Identity(), (2, 10, 10, 18, 26), kernel, (4, 4, 3, 1), (4, 4, 1, 0), 1, biases)
    assert layer.out_size == (5, 37, 37, 54, 29)

    with pytest.raises(InvalidDataException):
        _ = ConvNDTranspose.from_kernel(Identity(), (2, 20, 20, 20, 20), "test")

    with pytest.raises(InvalidDataException):
        _ = ConvNDTranspose.from_kernel(Identity(), (2, 20, 20, 20), kernel, 2, 4, 0, "test")

    with pytest.raises(InvalidDataException):
        _ = ConvNDTranspose.from_kernel(Identity(), "test", kernel, (2, 2, 3), 0, 0, biases)

    with pytest.raises(InvalidDataException):
        _ = ConvNDTranspose.from_kernel(Identity(), (2, 20, 20, 20), kernel, (2, 2, 2), "test", 0, biases)

    with pytest.raises(InvalidDataException):
        _ = ConvNDTranspose.from_kernel(Identity(), (2, 20, 20, 20), kernel, (2, 2, 2), 0, "test", biases)

    with pytest.raises(InvalidDataException):
        _ = ConvNDTranspose.from_kernel(Identity(), (2, 20, -20, 20), kernel, 2, 4, 0, biases)

    with pytest.raises(InvalidDataException):
        _ = ConvNDTranspose.from_kernel(Identity(), (2, 20, 20, 20), kernel, 0, 4, 0, biases)

    with pytest.raises(InvalidDataException):
        _ = ConvNDTranspose.from_kernel(Identity(), (2, 20, 20, 20), kernel, 2, -1, 0, biases)

    with pytest.raises(InvalidDataException):
        _ = ConvNDTranspose.from_kernel(Identity(), (2, 20, 20, 20), kernel, (2, 0, 2), 4, 0, biases)

    with pytest.raises(InvalidDataException):
        _ = ConvNDTranspose.from_kernel(Identity(), (2, 20, 20, 20), kernel, 2, (0, -1, 0), 0, biases)

    with pytest.raises(InvalidDataException):
        _ = ConvNDTranspose.from_kernel(Identity(), (2, 20, 20, 20, 20), kernel, 2, (0, 0), 0, biases)

    with pytest.raises(InvalidDataException):
        _ = ConvNDTranspose.from_kernel(Identity(), (2, 20), kernel, 2, (0, 1, 0), 0, biases)

    with pytest.raises(InvalidDataException):
        _ = ConvNDTranspose.from_kernel(Identity(), (2, 20, 20, 20), kernel, (2, 2), 0, 0, biases)

    with pytest.raises(InvalidDataException):
        _ = ConvNDTranspose.from_kernel(None, (2, 20, 20, 20), kernel, 2, 0, 0, biases)

    with pytest.raises(InvalidDataException):
        _ = ConvNDTranspose.from_kernel(Identity(), (2, 20, 20, 20), kernel, (2, 2, 1), 0, -1, biases)

    with pytest.raises(InvalidDataException):
        _ = ConvNDTranspose.from_kernel(Identity(), (2, 20, 20, 20), kernel, (2, 2, 1), 0, (1, 1), biases)
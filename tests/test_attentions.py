from caspian.layers import Attention, MultiHeadAttention
from caspian.utilities import InvalidDataException
import numpy as np
import pytest

def test_attention():
    # Invalid input tests
    with pytest.raises(InvalidDataException):
        _ = Attention(-1)

    with pytest.raises(InvalidDataException):
        _ = Attention(1.1)

    with pytest.raises(InvalidDataException):
        _ = Attention(10, "test")

    with pytest.raises(InvalidDataException):
        _ = Attention(10, False, "test")


    # Standard usage tests
    layer = Attention(10, True, False)
    data_in = np.random.randn(20, 10)
    data_out = np.random.randn(20, 10)
    assert layer(data_in, data_in, data_in, True).shape == data_in.shape # All same size

    out1, out2, out3 = layer.backward(data_out)
    assert out1.shape == data_out.shape
    assert out2.shape == data_out.shape
    assert out3.shape == data_out.shape

    data_query_in = np.random.randn(25, 10)
    data_query_out = np.random.randn(25, 10)    # Differing query size
    assert layer(data_query_in, data_in, data_in, True).shape == data_query_in.shape

    out1, out2, out3 = layer.backward(data_query_out)
    assert out1.shape == data_query_out.shape
    assert out2.shape == data_out.shape
    assert out3.shape == data_out.shape


    # Incorrect shape tests
    data_false_in = np.random.randn(20, 11)
    with pytest.raises(ValueError):
        _ = layer(data_false_in, data_in, data_in)

    with pytest.raises(ValueError):
        _ = layer(data_in, data_false_in, data_in)

    with pytest.raises(ValueError):
        _ = layer(data_in, data_in, data_false_in)

    data_false_in = np.random.randn(22, 10)
    with pytest.raises(ValueError):
        _ = layer(data_in, data_false_in, data_in)

    with pytest.raises(ValueError):
        _ = layer(data_in, data_in, data_false_in)

    
    # Private layer variable access
    with pytest.raises(AttributeError):
        _ = layer.__q_layer(data_in)

    with pytest.raises(AttributeError):
        _ = layer.__k_layer(data_in)

    with pytest.raises(AttributeError):
        _ = layer.__v_layer(data_in)

    with pytest.raises(AttributeError):
        _ = layer.__softmax(data_in)

    
    # Deepcopy
    layer2 = layer.deepcopy()
    data_in = np.random.randn(25, 10)
    assert layer2 is not layer
    assert layer2.in_size == layer.in_size
    assert layer2.out_size == layer2.out_size
    assert layer2.d_embed == layer.d_embed
    assert layer2.use_bias == layer.use_bias
    assert layer2.use_mask == layer.use_mask
    assert np.allclose(layer2(data_in, data_in, data_in), layer(data_in, data_in, data_in))


    # Saving
    context = layer.save_to_file()
    layer2 = Attention.from_save(context)
    assert layer2.in_size == layer.in_size
    assert layer2.out_size == layer2.out_size
    assert layer2.d_embed == layer.d_embed
    assert layer2.use_bias == layer.use_bias
    assert layer2.use_mask == layer.use_mask
    assert np.allclose(layer2(data_in, data_in, data_in), layer(data_in, data_in, data_in))




def test_multihead_attention():
    # Invalid input tests
    with pytest.raises(InvalidDataException):
        _ = MultiHeadAttention(-1, 5)

    with pytest.raises(InvalidDataException):
        _ = MultiHeadAttention(1.1, 5)

    with pytest.raises(InvalidDataException):
        _ = MultiHeadAttention(10, -1)

    with pytest.raises(InvalidDataException):
        _ = MultiHeadAttention(10, 1.1)

    with pytest.raises(InvalidDataException):
        _ = MultiHeadAttention(10, 5, -0.1)

    with pytest.raises(InvalidDataException):
        _ = MultiHeadAttention(10, 5, 1.1)

    with pytest.raises(InvalidDataException):
        _ = MultiHeadAttention(11, 5)

    with pytest.raises(InvalidDataException):
        _ = MultiHeadAttention(10, 5, 0.3, "test")

    with pytest.raises(InvalidDataException):
        _ = MultiHeadAttention(10, 5, 0.3, False, "test")


    # Standard usage tests
    layer = MultiHeadAttention(20, 5, 0.3, True, False)
    data_in = np.random.randn(40, 20)
    data_out = np.random.randn(40, 20)
    assert layer(data_in, data_in, data_in, True).shape == data_in.shape # All same size

    out1, out2, out3 = layer.backward(data_out)
    assert out1.shape == data_out.shape
    assert out2.shape == data_out.shape
    assert out3.shape == data_out.shape

    data_query_in = np.random.randn(45, 20)
    data_query_out = np.random.randn(45, 20)    # Differing query size
    assert layer(data_query_in, data_in, data_in, True).shape == data_query_in.shape

    out1, out2, out3 = layer.backward(data_query_out)
    assert out1.shape == data_query_out.shape
    assert out2.shape == data_out.shape
    assert out3.shape == data_out.shape


    # Incorrect shape tests
    data_false_in = np.random.randn(40, 21)
    with pytest.raises(ValueError):
        _ = layer(data_false_in, data_in, data_in)

    with pytest.raises(ValueError):
        _ = layer(data_in, data_false_in, data_in)

    with pytest.raises(ValueError):
        _ = layer(data_in, data_in, data_false_in)

    data_false_in = np.random.randn(44, 20)
    with pytest.raises(ValueError):
        _ = layer(data_in, data_false_in, data_in)

    with pytest.raises(ValueError):
        _ = layer(data_in, data_in, data_false_in)

    
    # Private layer variable access
    with pytest.raises(AttributeError):
        _ = layer.__q_layer(data_in)

    with pytest.raises(AttributeError):
        _ = layer.__k_layer(data_in)

    with pytest.raises(AttributeError):
        _ = layer.__v_layer(data_in)

    with pytest.raises(AttributeError):
        _ = layer.__o_layer(data_in)

    with pytest.raises(AttributeError):
        _ = layer.__dropout(data_in)

    with pytest.raises(AttributeError):
        _ = layer.__softmax(data_in)


    # Deepcopy
    layer2 = layer.deepcopy()
    data_in = np.random.randn(44, 20)
    assert layer2 is not layer
    assert layer2.in_size == layer.in_size
    assert layer2.out_size == layer.out_size
    assert layer2.d_embed == layer.d_embed
    assert layer2.num_heads == layer.num_heads
    assert layer2.head_size == layer.head_size
    assert layer2.use_bias == layer.use_bias
    assert layer2.use_mask == layer.use_mask
    assert np.allclose(layer2(data_in, data_in, data_in), layer(data_in, data_in, data_in))


    # Saving
    context = layer.save_to_file()
    layer2 = MultiHeadAttention.from_save(context)
    assert layer2 is not layer
    assert layer2.in_size == layer.in_size
    assert layer2.out_size == layer.out_size
    assert layer2.d_embed == layer.d_embed
    assert layer2.num_heads == layer.num_heads
    assert layer2.head_size == layer.head_size
    assert layer2.use_bias == layer.use_bias
    assert layer2.use_mask == layer.use_mask
    assert np.allclose(layer2(data_in, data_in, data_in), layer(data_in, data_in, data_in))
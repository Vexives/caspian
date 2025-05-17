from caspian.layers import Mult, Add, Concat, MatMul
from caspian.utilities import InvalidDataException
import pytest
import numpy as np

@pytest.mark.filterwarnings("ignore::numpy.VisibleDeprecationWarning")
def test_mult():
    # Argument + KWarg tests
    with pytest.raises(TypeError):
        _ = Mult(1, 2, 3)
    with pytest.raises(TypeError):
        _ = Mult(new_var = "test")

    layer = Mult()


    # Single element pass tests
    data_in = np.zeros((10, 5))
    with pytest.raises(InvalidDataException):
        _ = layer(data_in)
    with pytest.raises(InvalidDataException):
        _ = layer((data_in))

    
    # Variable shape tests
    data_in = (np.zeros((10, 5)), np.zeros((12, 5)))
    with pytest.raises(ValueError):
        _ = layer(data_in)


    # Private variables inaccessible
    data_in = (np.zeros((10, 5)), np.zeros((10, 5)))
    with pytest.raises(AttributeError):
        _ = layer(data_in, True)
        _ = layer.__last_ins


    # Standard usage tests
    data_in = (np.zeros((10, 5)), np.zeros((10, 5)), np.zeros((10, 5)))
    assert layer(data_in, True).shape == (10, 5)

    data_out = np.zeros((10, 5))
    result = layer.backward(data_out)
    assert len(result) == 3
    assert result[0].shape == (10, 5)




def test_add():
    # Argument + KWarg tests
    with pytest.raises(TypeError):
        _ = Add(1, 2, 3)
    with pytest.raises(TypeError):
        _ = Add(new_var = "test")

    layer = Add()


    # Single element pass tests
    data_in = np.zeros((10, 5))
    with pytest.raises(InvalidDataException):
        _ = layer(data_in)
    with pytest.raises(InvalidDataException):
        _ = layer((data_in))

    
    # Variable shape tests
    data_in = (np.zeros((10, 5)), np.zeros((12, 5)))
    with pytest.raises(ValueError):
        _ = layer(data_in)


    # Standard usage tests
    data_in = (np.zeros((10, 5)), np.zeros((10, 5)), np.zeros((10, 5)))
    assert layer(data_in, True).shape == (10, 5)

    data_out = np.zeros((10, 5))
    result = layer.backward(data_out)
    assert np.allclose(result, data_out)
    assert result.shape == (10, 5)    




@pytest.mark.filterwarnings("ignore::numpy.VisibleDeprecationWarning")
def test_concat():
    # Argument + KWarg tests
    with pytest.raises(TypeError):
        _ = Concat(1, 2, 3)
    with pytest.raises(TypeError):
        _ = Concat(new_var = "test")

    layer = Concat(0)


    # Single element pass tests
    data_in = np.zeros((10, 5))
    with pytest.raises(InvalidDataException):
        _ = layer(data_in)
    with pytest.raises(InvalidDataException):
        _ = layer((data_in))

    
    # Variable shape tests
    data_in = (np.zeros((3, 10, 5)), np.zeros((1, 12, 5)))
    with pytest.raises(ValueError):
        _ = layer(data_in)


    # Private variables inaccessible
    data_in = (np.zeros((10, 5)), np.zeros((10, 5)))
    with pytest.raises(AttributeError):
        _ = layer(data_in, True)
        _ = layer.__last_ins


    # Standard usage tests
    data_in = (np.zeros((1, 10, 5)), np.zeros((3, 10, 5)), np.zeros((2, 10, 5)))
    assert layer(data_in, True).shape == (6, 10, 5)

    data_out = np.zeros((6, 10, 5))
    result = layer.backward(data_out)
    assert result[0].shape == (1, 10, 5) 
    assert result[1].shape == (3, 10, 5)
    assert result[2].shape == (2, 10, 5)  




def test_matmul():
    # Argument + KWarg tests
    with pytest.raises(TypeError):
        _ = MatMul(1, 2, 3)
    with pytest.raises(TypeError):
        _ = MatMul(new_var = "test")

    layer = MatMul()


    # Single element pass tests
    data_in = np.zeros((10, 5))
    with pytest.raises(InvalidDataException):
        _ = layer(data_in)
    with pytest.raises(InvalidDataException):
        _ = layer((data_in))


    # More than two elements pass tests
    data_in = (np.zeros((10, 5)), np.zeros((10, 5)), np.zeros((10, 5)))
    with pytest.raises(InvalidDataException):
        _ = layer(data_in)


    # Private variables inaccessible
    data_in = (np.zeros((5, 10)), np.zeros((10, 5)))
    with pytest.raises(AttributeError):
        _ = layer(data_in, True)
        _ = layer.__last_ins        


    # Standard usage tests
    data_in = (np.zeros((5, 5, 10)), np.zeros((5, 10, 5)))
    assert layer(data_in, True).shape == (5, 5, 5)

    data_out = np.zeros((5, 5, 5))
    result = layer.backward(data_out)
    assert result[0].shape == (5, 5, 10) 
    assert result[1].shape == (5, 10, 5)       
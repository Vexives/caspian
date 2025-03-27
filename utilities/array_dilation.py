from cudalib import np

def dilate_array(array: np.ndarray, new_shape: tuple[int, ...], 
                 strides: tuple[int, ...]) -> np.ndarray:
    slices = (Ellipsis,) + tuple(map(lambda x: slice(None, None, x), strides))
    d_a = np.zeros(new_shape)
    d_a[slices] = array
    return d_a
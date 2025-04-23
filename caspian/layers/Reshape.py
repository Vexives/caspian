from ..cudalib import np
from . import Layer
from ..utilities import ShapeIncompatibilityException

class Reshape(Layer):
    """
    A reshape layer that takes an array of any shape and morphs it into a new shape with the same size.

    Supports any given shape and dimensionality as an input, as long as that shape is given in the initial parameters.
       
        
    Attributes
    ---------
    in_size : tuple[int, ...]
        A tuple containing the expected input shape `(..., X)` where `...` is any 
        intermediate dimension, and `X` is the expected length of the input.
    out_size : tuple[int, ...]
        A tuple containing the same shape as `in_size`.


    Examples
    --------
    >>> layer1 = Reshape((10, 5, 10), (50, 10))
    >>> in_arr = np.ones((10, 5, 10))
    >>> out_arr = layer1(in_arr)
    >>> print(out_arr.shape)
    (50, 10)
    """
    def __init__(self, input_size: tuple[int, ...], output_size: tuple[int, ...]) -> None:
        """
        Initializes a `Reshape` layer using given parameters.

        Parameters
        ----------
        input_size : int | tuple[int, ...]
            An integer or tuple of integers matching the shape of the expected input arrays.
        output_size : int | tuple[int, ...]
            An integer or tuple of integers representing the expected output shape of the input arrays.
            Must be the same size as `input_size` or an AttributeError is raised.

        Raises
        ------
        AttributeError
            If the input and output sizes given are not compatible and the input cannot be reshaped into
            the output shape. 
        """
        try:
            in_test_shape = tuple(filter(lambda x: x != -1, input_size))
            example_arr = np.zeros(in_test_shape)
            example_arr = example_arr.reshape(output_size)
        except:
            raise ShapeIncompatibilityException("Input and output shapes not compatible." + \
                                                f"- {input_size} - {output_size}")

        super().__init__(input_size, output_size)
    

    def forward(self, data: np.ndarray, *_) -> np.ndarray:
        """
        Performs a forward propagation pass through this layer, reshaping the data and returning.
        
        Parameters
        ----------
        data : ndarray
            The data that the forward pass will be performed on. Must match the input size of this layer.
        training : bool
            Specify whether the layer is currently training or not to save the necessary information
            required for the backward pass.
        
        Returns
        -------
        ndarray
            The forward propagated array with reshaped values.
        """ 
        return data.reshape(self.out_size)
    

    def backward(self, cost_err: np.ndarray) -> np.ndarray:
        """
        Performs a backward propagation pass through this layer, reshaping the gradient back into the
        expected input shape.

        Parameters
        ----------
        cost_err : ndarray
            The learning gradient that will be processed and returned.

        Returns
        -------
        ndarray
            The new learning gradient for any layers that provided data to this instance. Will have the
            same shape as this layer's input shape.
        """
        return cost_err.reshape(self.in_size)
    

    def step(self) -> None:
        """Not applicable for this layer."""
        pass
    

    def clear_grad(self) -> None:
        """Not applicable for this layer."""
        pass


    def set_optimizer(self, *_) -> None:
        """Not applicable for this layer."""
        pass
    

    def deepcopy(self) -> 'Reshape':
        """Creates a new deepcopy of this layer with the exact same shapes."""
        return Reshape(self.in_size, self.out_size)
    

    def save_to_file(self, filename: str = None) -> str | None:
        """
        Encodes the current layer information into a string, and saves it to a file if the
        path is specified.

        Parameters
        ----------
        filename : str, default: None
            The file for the layer's information to be stored to. If this is not provided and
            is instead of type `None`, the encoded string will just be returned.

        Returns
        -------
        str | None
            If no file is specified, a string containing all information about this model is returned.
        """
        write_ret_str = f"Reshape\u00A0\n" + " ".join(list(map(str, self.in_size))) + \
                        "\n" + " ".join(list(map(str, self.out_size))) + "\n\u00A0"
        if not filename:
            return write_ret_str
        if filename.find(".cspn") == -1:
            filename += ".cspn"
        with open(filename, "w+") as file:
            file.write(write_ret_str)
            file.close()
    

    @staticmethod
    def from_save(context: str, file_load: bool = False) -> 'Reshape':
        """
        A static method which creates an instance of this layer class based on the information provided.
        The string provided can either be a file name/path, or the encoded string containing the layer's
        information.

        Parameters
        ----------
        context : str
            The string containing either the name/path of the file to be loaded, or the `save_to_file()`
            encoded string. If `context` is the path to a file, then the boolean parameter `file_load`
            MUST be set to True.
        file_load : bool, default: False
            A boolean which determines whether a file will be opened and the context extracted,
            or the `context` string provided will be parsed instead. If set to True, the `context` string
            will be treated as a file path. Otherwise, `context` will be parsed itself.

        Returns
        -------
        Reshape
            A new `Reshape` layer containing all of the information encoded in the string or file provided.
        """
        def parse_and_return(handled_str: str):
            data_arr = handled_str.splitlines()
            in_size = tuple(map(int, data_arr[1].split()))
            out_size = tuple(map(int, data_arr[2].split()))
            return Reshape(in_size, out_size)

        if file_load:
            full_parse_str = None
            with open(context, "r") as file:
                full_parse_str = file.read()
                file.close()
            return parse_and_return(full_parse_str)
        return parse_and_return(context)
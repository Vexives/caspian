from ..cudalib import np
from . import Layer

class Upsampling2D(Layer):
    """
    An upsampling layer which scales the last two dimensions of the input by a given factor.

    Supports any given shape and dimensionality as an input, but only performs the operation
    on the last two layers.
        
    Attributes
    ---------
    rate : tuple[int, int]
        The multiplicative size scaling rate of this layer for both dimensions.


    Examples
    --------
    >>> layer1 = Upsampling2D((3, 1))
    >>> in_arr = np.ones((2, 10, 10))
    >>> out_arr = layer1(in_arr)
    >>> print(out_arr.shape)
    (2, 30, 10)
    """
    def __init__(self, rate: tuple[int, int] | int):
        """
        Initializes an `Upsampling2D` layer using given parameters.

        Parameters
        ----------
        rate : tuple[int, int] | int
            An int or tuple of ints which represent the multiplicative size scaling rate of this 
            layer for both dimensions.
        """
        super().__init__(None, None)
        assert int(np.prod(rate)) > 1, "Upsampling rate must be above 1 for at least one dimension."
        self.rate = rate if isinstance(rate, tuple) else (rate, rate)
    

    def forward(self, data: np.ndarray, training: bool = False) -> np.ndarray:
        """
        Performs a forward propagation pass through this layer given the specified upsample rate.
        
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
            The forward propagated array with a new upsampled size.
        """
        if training:
            self.__last_in = data
        return np.kron(data, np.ones(self.rate))
    

    def backward(self, cost_err: np.ndarray) -> np.ndarray:
        """
        Performs a backward propagation pass through this layer and returns the newly shaped
        gradient with respect to the input.

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
        ret_grad = cost_err.reshape((-1, *self.rate, *self.__last_in.shape[-2:])).sum(axis=(-3, -4))
        return ret_grad
    

    def step(self) -> None:
        """Not applicable for this layer."""
        pass
    

    def clear_grad(self) -> None:
        """Clears any data required by the backward pass."""
        self.__last_in = None


    def set_optimizer(self, *_) -> None:
        """Not applicable for this layer."""
        pass


    def deepcopy(self) -> 'Upsampling2D':
        """Creates a new deepcopy of this layer with the exact same parameters."""
        new_neuron = Upsampling2D(self.rate)
        return new_neuron
    

    def save_to_file(self, filename: str = None) -> None | str:
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
        write_ret_str = f"Upsampling2D\u00A0{self.rate}\n\u00A0" 
        if not filename:
            return write_ret_str
        if filename.find(".cspn") == -1:
            filename += ".cspn"
        with open(filename, "w+") as file:
            file.write(write_ret_str)
            file.close()


    @staticmethod
    def from_save(context: str, file_load: bool = False) -> 'Upsampling2D':
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
        Upsampling2D
            A new `Upsampling2D` layer containing all of the information encoded in the string or file provided.
        """
        def parse_and_return(handled_str: str):
            data_arr = handled_str.splitlines()
            prop_info = data_arr[0].split("\u00A0")
            
            new_neuron = Upsampling2D(tuple(map(int, prop_info[1].split())))
            return new_neuron

        if file_load:
            full_parse_str = None
            with open(context, "r") as file:
                full_parse_str = file.read()
                file.close()
            return parse_and_return(full_parse_str)
        return parse_and_return(context)
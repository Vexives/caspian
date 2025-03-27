from cudalib import np
from . import Reshape

class Dropout(Reshape):
    """
    A dropout layer which cancels out random points of the given data across all dimensions.
    Performs no actions unless training mode is set to be `True` during the forward pass.

    Supports any given shape and dimensionality as an input, as long as that shape is given in the initial parameters.
       
        
    Attributes
    ---------
    in_size : tuple[int, ...]
        A tuple containing the expected input shape `(..., X)` where `...` is any 
        intermediate dimension, and `X` is the expected length of the input.
    out_size : tuple[int, ...]
        A tuple containing the same shape as `in_size`.
    chance : float
        A float representing the dropout chance for each point in the data array.
    drop_mask : ndarray, default: None
        A mask which contains all of the locations that were dropped during the forward pass.
        Only initialized when a forward pass was performed with training mode set to `True`. 


    Examples
    --------
    >>> layer1 = Dropout((1, 5))
    >>> in_arr = np.arange(5)
    >>> out_arr = layer1(in_arr, True)
    >>> print(out_arr)
    [0 1 2 0 4]
    """
    def __init__(self, input_size: tuple[int, ...], drop_chance: float = 0.7) -> None:
        """
        Initializes a `Dropout` layer using given parameters.

        Parameters
        ----------
        input_size : int | tuple[int, ...]
            An integer or tuple of integers matching the shape of the expected input arrays.
        drop_chance : float
            A float representing the dropout chance for each point in the expected input arrays.
        """
        super().__init__(input_size, input_size)
        self.chance = drop_chance
        self.drop_mask = None
    

    def forward(self, data: np.ndarray, training: bool = False) -> np.ndarray:
        """
        Performs a forward propagation pass through this layer, dropping some data points based on
        the provided drop chance.
        
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
            The forward propagated array with dropped values.
        """ 
        assert data.shape == self.in_size
        if training:
            self.drop_mask = np.random.uniform(0.0, 1.0, self.in_size) > self.chance
            mult_comp = (1.0 / 1.0 - self.chance) if self.chance != 1.0 else 1
            return data * self.drop_mask * mult_comp
        return data
    

    def backward(self, cost_err: np.ndarray) -> np.ndarray:
        """
        Performs a backward propagation pass through this layer, applying the mask to the gradient
        and returning.

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
        assert self.drop_mask is not None
        return cost_err * self.drop_mask
    

    def step(self) -> None:
        """Not applicable for this layer."""
        pass


    def clear_grad(self) -> None:
        """Clears the drop mask from this layer."""
        self.drop_mask = None


    def set_optimizer(self, *_) -> None:
        """Not applicable for this layer."""
        pass
    

    def deepcopy(self) -> 'Dropout':
        """Creates a new deepcopy of this layer with the exact same shape and drop chance."""
        return Dropout(self.in_size, self.chance)
    

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
        write_ret_str = f"Dropout\u00A0" + " ".join(list(map(str, self.in_size))) + "\n" + \
                        f"CHANCE {self.chance}" + "\n\u00A0"
        if not filename:
            return write_ret_str
        if filename.find(".cspn") == -1:
            filename += ".cspn"
        with open(filename, "w+") as file:
            file.write(write_ret_str)
            file.close()


    @staticmethod
    def from_save(context: str, file_load: bool = False) -> 'Dropout':
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
        Dropout
            A new `Dropout` layer containing all of the information encoded in the string or file provided.
        """
        def parse_and_return(handled_str: str):
            data_arr = handled_str.splitlines()
            in_size = tuple(map(int, data_arr[0].split("\u00A0")[1]))
            chance = float(data_arr[1].split()[1])
            return Dropout(in_size, chance)

        if file_load:
            full_parse_str = None
            with open(context, "r") as file:
                full_parse_str = file.read()
                file.close()
            return parse_and_return(full_parse_str)
        return parse_and_return(context)
from ..cudalib import np
from . import Layer
from ..utilities import check_types, InvalidDataException

class PositionalEncoding(Layer):
    """
    A positional encoding layer which applies a sinusoidal mask onto the input data.

    Creates an encoding table of shape `(L, E)` where `L` is the maximum token length and `E` is the
    embedding length of the tokens.

        
    Attributes
    ---------
    max_length : int
        The maximum number of tokens in a single array.
    dim_size : int
        The embedding length of this layer. Represents how many values correspond to a single word
        token.
    encoding : ndarray
        The `(V, E)` shape array of weights that are applied to the input to provide sinusoidal positional
        encodings.


    Examples
    --------
    >>> layer1 = PositionalEncoding(20, 64)
    >>> in_arr = np.random.randint(0, 10, (10, 64))
    >>> out_arr = layer1(in_arr)
    >>> print(out_arr.shape)
    (10, 64)
    """
    @check_types(("dim_size", lambda x: x > 0, "Argument \"dim_size\" must be greater than 0."),
                 ("dim_size", lambda x: not bool(x % 2), "Argument \"dim_size\" must be even."),
                 ("max_length", lambda x: x > 0, "Argument \"max_length\" must be greater than 0."))
    def __init__(self, max_length: int, dim_size: int):
        """
        Initializes a `PositionalEncoding` layer using given parameters.

        Parameters
        ----------
        max_length : int
            The maximum length of this layer. Designates how many word or token positions
            are registered as pre-made encodings in the table.
        dim_size : int
            The number of embedded values for each token given as input.
        """
        super().__init__(None, None)
        self.dim_size = dim_size
        self.max_length = max_length

        pos_init = np.expand_dims(np.arange(max_length), 1)
        exp_arr = np.exp(np.arange(0, dim_size, 2) * (-np.log(10000)/dim_size))

        self.encoding = np.zeros((max_length, dim_size))
        self.encoding[:, 0::2] = np.sin(pos_init * exp_arr)
        self.encoding[:, 1::2] = np.cos(pos_init * exp_arr)


    def forward(self, data: np.ndarray, *_) -> np.ndarray:
        """
        Performs a forward propagation pass through this layer given the encoding parameters.
        
        Parameters
        ----------
        data : ndarray
            The data that the forward pass will be performed on. Must match the input size of this layer.
        
        Returns
        -------
        ndarray
            The forward propagated array with the shape equal to the original array's shape.
        """
        if data.shape[0] > self.max_length:
            raise InvalidDataException(f"Data shape at dimension 0 must be less than or equal to the max length. - {self.max_length}")
        return data + self.encoding[:data.shape[0], :]


    def deepcopy(self) -> 'PositionalEncoding':
        """Creates a new deepcopy of this layer with the exact same parameters."""
        return PositionalEncoding(self.dim_size, self.max_length)
    

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
        write_ret_str = f"PositionalEncoding\u00A0{self.dim_size}\u00A0{self.max_length}\n\u00A0" 
        if not filename:
            return write_ret_str
        if filename.find(".cspn") == -1:
            filename += ".cspn"
        with open(filename, "w+") as file:
            file.write(write_ret_str)
            file.close()


    @staticmethod
    def from_save(context: str, file_load: bool = False) -> 'PositionalEncoding':
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
        PositionalEncoding
            A new `PositionalEncoding` layer containing all of the information encoded in the string or file provided.
        """
        def parse_and_return(handled_str: str):
            data_arr = handled_str.splitlines()
            prop_info = data_arr[0].split("\u00A0")
            return PositionalEncoding(int(prop_info[1]), int(prop_info[2]))

        if file_load:
            full_parse_str = None
            with open(context, "r") as file:
                full_parse_str = file.read()
                file.close()
            return parse_and_return(full_parse_str)
        return parse_and_return(context)
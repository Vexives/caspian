from ..cudalib import np
from . import Layer
from ..utilities import check_types, InvalidDataException

class CosineSimilarity(Layer):
    """
    A layer that takes the Cosine Similarity between two given arrays regardless of shape.
    
    The two given arrays must be of the same shape (broadcastable), and the resulting array will
    have `N-1` dimensions, where inputs `A` and `B` have `N` dimensions.

        
    Attributes
    ---------
    dim : int
        The dimension at which this layer will take the cosine similarity of the given inputs.
    eps : float
        An epsilon value that will replace any zeros in the normalized denominator product during
        the forward pass.


    Examples
    --------
    >>> a = np.array([1, 2, 3, 4, 5])
    >>> b = np.array([6, 7, 8, 9, 10])
    >>> layer1 = CosineSimilarity(dim = 0)
    >>> res = layer1(a, b)
    >>> print(res)
    0.9649505047327671
    """
    @check_types(("eps", lambda x: x > 0, "Argument \"eps\" must be greater than 0."))
    def __init__(self, dim: int = 1, eps: float = 1e-8):
        """
        Initializes a `CosineSimilarity` layer using given parameters.

        Parameters
        ----------
        dim : int, default = 1
            Specifies the dimension at which the cosine similarity will be taken.
        eps : float, default = 1e-8
            An epsilon value that will replace any zeros in the normalized denominator product.

        Raises
        ------
        InvalidDataException
            If any parameter is not of the expected data type.
        """
        self.dim = dim
        self.eps = eps
        super().__init__(None, None)


    def __call__(self, data_a: np.ndarray, data_b: np.ndarray, training: bool = False) -> np.ndarray:
        """Calls the class forward function and provides the given parameters."""
        return self.forward(data_a, data_b, training)
    

    def forward(self, data_a: np.ndarray, data_b: np.ndarray, training = False) -> np.ndarray:
        """
        Performs a forward propagation pass through this layer and returns the cosine similarity
        (will be one dimension smaller than given data).
        
        Parameters
        ----------
        data_a : ndarray
            The first set of data points that will be compared to the second set.
        data_b : ndarray
            The second set of data points that will be compared to the first set.
        training : bool
            Specify whether the layer is currently training or not to save the necessary information
            required for the backward pass.
        
        Returns
        -------
        ndarray
            The forward propagated array with one less dimension than the two given input arrays.

        Raises
        ------
        InvalidDataException
            If the shapes of the two inputs are not fully broadcastable and do not match.
        """
        if data_a.shape != data_b.shape:
            raise InvalidDataException(f"Data shapes could not be broadcasted. - {data_a.shape}, {data_b.shape}")
        if len(data_a.shape) <= abs(self.dim + int(self.dim < 0)):
            raise InvalidDataException(f"Layer dimension out of range for data shapes. - {data_a.shape}, {data_b.shape}")

        a_mag = np.linalg.norm(data_a, axis=self.dim, keepdims=True)
        b_mag = np.linalg.norm(data_b, axis=self.dim, keepdims=True)
        mult_mag = np.maximum((a_mag * b_mag), self.eps)
        cos_sim = np.sum((data_a * data_b) / mult_mag, axis=self.dim)
        
        if training:
            self.__last_am = a_mag
            self.__last_bm = b_mag
            self.__last_fm = mult_mag
            self.__last_ins = (data_a, data_b)
            self.__last_res = np.expand_dims(cos_sim, self.dim)
        return cos_sim
    

    def backward(self, cost_err: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Performs a backward propagation pass through this layer and returns two gradients corresponding
        to the first and second data points from the last forward pass.

        Parameters
        ----------
        cost_err : ndarray
            The learning gradient that will be processed and returned. Any information gained by the weights
            will be taken from this gradient and will fit to this data.

        Returns
        -------
        tuple[ndarray, ndarray]
            The new learning gradients for any layers that provided data to this instance. Will have the
            same shapes as the last two inputs.
        """
        last_a, last_b = self.__last_ins
        a_grad = (last_b / self.__last_fm) - (self.__last_res * (last_a / np.square(self.__last_bm)))
        b_grad = (last_a / self.__last_fm) - (self.__last_res * (last_b / np.square(self.__last_am)))

        cost_err = np.expand_dims(cost_err, self.dim)
        a_grad *= cost_err
        b_grad *= cost_err
        return a_grad, b_grad
    

    def clear_grad(self) -> None:
        """Clears any data required by the backward pass."""
        self.__last_am = None
        self.__last_bm = None
        self.__last_fm = None
        self.__last_ins = None
        self.__last_res = None


    def deepcopy(self) -> 'CosineSimilarity':
        """Creates a new deepcopy of this layer with the exact same parameters."""
        return CosineSimilarity(self.dim, self.eps)
    

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
        write_ret_str = f"CosineSimilarity\u00A0{self.dim}\u00A0{self.eps}\n\u00A0" 
        if not filename:
            return write_ret_str
        if filename.find(".cspn") == -1:
            filename += ".cspn"
        with open(filename, "w+") as file:
            file.write(write_ret_str)
            file.close()


    @staticmethod
    def from_save(context: str, file_load: bool = False) -> 'CosineSimilarity':
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
        CosineSimilarity
            A new `CosineSimilarity` layer containing all of the information encoded in the string or file provided.
        """
        def parse_and_return(handled_str: str):
            data_arr = handled_str.splitlines()
            prop_info = data_arr[0].split("\u00A0")[1:]

            new_neuron = CosineSimilarity(int(prop_info[0]), float(prop_info[1]))
            return new_neuron

        if file_load:
            full_parse_str = None
            with open(context, "r") as file:
                full_parse_str = file.read()
                file.close()
            return parse_and_return(full_parse_str)
        return parse_and_return(context)
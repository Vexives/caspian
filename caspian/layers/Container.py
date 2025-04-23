from ..cudalib import np
from . import Layer
from ..activations import Activation, parse_act_info
from ..utilities import InvalidDataException

class Container(Layer):
    '''
    A basic container class which takes in an input size and a function to be used.
    Does not perform any further operations other than applying the function, and allows
    activations to be used alongside other layers in `Sequence`s.
    

    Attributes
    ---------
    in_size : tuple[int, ...]
        A tuple containing the expected input size `(N, ..., X)`, where `N` is the number of batches, 
        `...` is any intermediate dimension, and `X` is the expected length of the input.
    out_size : tuple[int, ...]
        A tuple containing the expected output size matching the `in_size`.
    funct : Activation
        The given activation class which performs the desired non-linear transform to the data.

    
    Examples
    --------
    >>> layer1 = Container(ReLU(), (5,))
    >>> in_arr = np.random.uniform(-0.5, 0.5, (5,))
    >>> out_arr = layer1(in_arr)
    >>> print(out_arr)
    [ 0.42778123  0.          0.          0.11093225  0.        ]
    '''
    def __init__(self, funct: Activation):
        """
        Initializes a `Container` layer using given parameters.

        Parameters
        ----------
        funct : Activation
            An activation function class which supports both forward and backward non-linear 
            transformations.
        inputs : int | tuple[int, ...]
            An integer or tuple of integers matching the shape of the expected input arrays.
        """
        if not isinstance(funct, Activation):
            raise InvalidDataException("Function for container must be a Caspian \'Activation\' class.")
        super().__init__(None, None)
        self.funct = funct
    

    def forward(self, data: np.ndarray, *_) -> np.ndarray:
        """
        Performs a forward propagation pass through this layer given the current weights and biases.
        
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
            The forward propagated array with the shape equal to this layer's output shape.
        """
        return self.funct(data)


    def backward(self, cost_err: np.ndarray) -> np.ndarray:
        """
        Performs a backward propagation pass through this layer and updates the weights according to
        the provided learning gradient.

        Parameters
        ----------
        cost_err : ndarray
            The learning gradient that will be processed and returned. Any information gained by the weights
            will be taken from this gradient and will fit to this data.

        Returns
        -------
        ndarray
            The new learning gradient for any layers that provided data to this instance. Will have the
            same shape as this layer's input shape.
        """
        return self.funct.backward(cost_err)


    def deepcopy(self) -> 'Container':
        """Creates a new deepcopy of this layer with the exact same parameters."""
        return Container(self.funct)


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
        write_ret_str = f"Container\u00A0{repr(self.funct)}\u00A0" + \
                        "\n\u00A0"
        if not filename:
            return write_ret_str
        if filename.find(".cspn") == -1:
            filename += ".cspn"
        with open(filename, "w+") as file:
            file.write(write_ret_str)
            file.close()


    def from_save(context: str, file_load: bool = False) -> 'Container':
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
        Container
            A new `Container` layer containing all of the information encoded in the string or file provided.
        """
        def parse_and_return(handled_str: str):
            data_arr = handled_str.splitlines()
            prop_info = data_arr[0].split("\u00A0")

            new_act_funct = parse_act_info(prop_info[1])
            new_neuron = Container(new_act_funct)
            return new_neuron
        
        if file_load:
            full_parse_str = None
            with open(context, "r") as file:
                full_parse_str = file.read()
                file.close()
            return parse_and_return(full_parse_str)
        return parse_and_return(context)

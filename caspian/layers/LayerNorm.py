from ..cudalib import np
from . import Layer
from ..optimizers import Optimizer, StandardGD, parse_opt_info
from ..utilities import check_types, all_positive

class LayerNorm(Layer):
    """
    A layer-based normalization layer which normalizes the given data across all provided dimensions.

    Supports any given shape and dimensionality as an input, as long as the shape is provided at
    initialization (batch dimension NOT included).
       
        
    Attributes
    ---------
    in_size : tuple[int, ...]
        A tuple containing the expected input shape `(..., X)` where `...` is any 
        intermediate dimension, and `X` is the expected length of the input.
    out_size : tuple[int, ...]
        A tuple containing the same shape as `in_size`.
    axis : int
        The axis in which the expected channels of the input arrays are. Default value is 1, with the
        0th dimension being the batches.
    var_eps : float
        A generally very small float which corresponds to the epsilon value added to the 
        square root variance during normalization.
    opt : Optimizer, optional
        The provided optimizer which modifies the learning gradient before updating weights.
    layer_weight : ndarray
        A single dimensional array of values that correspond to the learnable weight values multiplied
        after normalization.
    bias_weight : ndarray
        A single dimensional array of values that correspond to the learnable bias values added
        after normalization.


    Examples
    --------
    >>> layer1 = LayerNorm((2, 5))
    >>> in_arr = np.arange(10).reshape((2, 5))
    >>> out_arr = layer1(in_arr)
    >>> print(out_arr)
    [[-1.5666989  -1.21854359 -0.87038828 -0.52223297 -0.17407766]
     [ 0.17407766  0.52223297  0.87038828  1.21854359  1.5666989 ]]
    """
    @check_types(("input_size", all_positive, "Argument \"input_size\" must contain all values above 0."),
                 ("var_eps", lambda x: x > 0.0, "Argument \"var_eps\" must be above 0.0."))
    def __init__(self, input_size: tuple[int, ...], weights: bool = True,
                 biases: bool = False, var_eps: float = 1e-8,
                 optimizer: Optimizer = StandardGD()):
        """
        Initializes a `LayerNorm` layer using given parameters.

        Parameters
        ----------
        input_size : int | tuple[int, ...]
            An integer or tuple of integers matching the shape of the expected input arrays.
        weights : bool, default: True
            A boolean which determines whether the learnable weight parameter array is initialized.
        biases : bool, default: True
            A boolean which determines whether the learnable bias parameter array is initialized.
        var_eps : float, default: 1e-8
            A float which corresponds to the epsilon value added to the square root variance during normalization.
        optimizer : Optimizer, default: StandardGD()
            An optimizer class which processes given loss gradients and adjusts them to match a desired 
            gradient descent path.        
        """
        super().__init__(input_size, input_size)
        self.norm_size = int(np.prod(input_size))
        self.layer_weight = np.ones((self.norm_size,)) if weights is True else None
        self.bias_weight = np.zeros((self.norm_size,)) if biases is True else None

        self.var_eps = var_eps
        self.opt = optimizer
    

    def forward(self, data: np.ndarray, training: bool = False) -> np.ndarray:
        """
        Performs a forward propagation pass through this layer given the current weight and bias values
        (if applicable).
        
        Parameters
        ----------
        data : ndarray
            The data that the forward pass will be performed on. Must match the input size of this layer
            except for batches.
        training : bool
            Specify whether the layer is currently training or not to save the necessary information
            required for the backward pass.
        
        Returns
        -------
        ndarray
            The forward propagated array with all values in each batch normalized.
        """        
        batch_data = np.expand_dims(data, axis=0) if len(data.shape) == len(self.in_size) else data
        shaped_data = batch_data.reshape((-1, self.norm_size))

        # Mean and variance of batches
        layer_mean = np.mean(shaped_data, axis=-1, keepdims=True)
        layer_var = np.var(shaped_data, axis=-1, keepdims=True)

        stdv = np.sqrt(layer_var + self.var_eps)
        new_data = ((shaped_data - layer_mean) / stdv)

        if training:
            self.__norm_res = new_data
            self.__last_stdv = stdv

        # Full weight and bias application, if applicable
        new_data = self.layer_weight * new_data if self.layer_weight is not None else new_data
        new_data = new_data + self.bias_weight if self.bias_weight is not None else new_data
        return new_data.reshape(data.shape)
    

    def backward(self, cost_err: np.ndarray) -> np.ndarray:
        """
        Performs a backward propagation pass through this layer and updates the weights and biases (if applicable) 
        according to the provided learning gradient.

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
        batch_err = np.expand_dims(cost_err, axis=0) if len(cost_err.shape) == len(self.in_size) else cost_err
        shaped_err = batch_err.reshape((-1, self.norm_size))
        if self.layer_weight is not None:
            shaped_err *= self.layer_weight

        opt_err = self.opt.process_grad(shaped_err)   # Weight/Bias update gradient with optimizer

        ret_grad = (shaped_err - shaped_err.mean(axis=-1, keepdims=True) 
                    - self.__norm_res * (shaped_err * self.__norm_res).mean(axis=-1, keepdims=True)) \
                    / self.__last_stdv
        
        # Update weights and biases (if applicable)
        if self.layer_weight is not None:
            self.layer_weight += (opt_err * self.__norm_res).sum(axis=0)
    
        if self.bias_weight is not None:
            self.bias_weight += opt_err.sum(axis=0)
        return ret_grad.reshape(batch_err.shape)
    

    def step(self) -> None:
        """Adds one step to this layer's optimizer and scheduler."""
        self.opt.step()
    

    def clear_grad(self) -> None:
        """Clears the optimizer gradient history and deletes any data required by the backward pass."""
        self.__last_stdv = None
        self.__norm_res = None
        self.opt.reset_grad()


    def set_optimizer(self, opt: Optimizer = StandardGD()) -> None:
        """
        Sets the optimizer of this layer to a new one. Will revert to a standard gradient descent
        optimizer if none is provided.
           
        Parameters
        ----------
        opt : Optimizer, default: StandardGD()
            The new optimizer for this layer to keep.
        """
        self.opt = opt


    def deepcopy(self) -> 'LayerNorm':
        """Creates a new deepcopy of this layer with the exact same weights (if applicable) and parameters."""
        new_neuron = LayerNorm(self.in_size,
                               self.layer_weight is not None, self.bias_weight is not None,
                               self.var_eps, self.opt.deepcopy())
        new_neuron.layer_weight = self.layer_weight.copy() if self.layer_weight is not None else None
        new_neuron.bias_weight = self.bias_weight.copy() if self.bias_weight is not None else None
        return new_neuron
    

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
        write_ret_str = f"LayerNorm\u00A0" + " ".join(list(map(str, self.in_size))) + f"\u00A0{self.var_eps}\u00A0{repr(self.opt)}"
        write_ret_str += f"\nWEIGHTS\u00A0" + " ".join(list(map(str, self.layer_weight.flatten().tolist()))) \
                         if self.layer_weight is not None else "\nWEIGHTS\u00A0None"
        write_ret_str += f"\nBIASES\u00A0" + " ".join(list(map(str, self.bias_weight.flatten().tolist()))) \
                         if self.bias_weight is not None else "\nBIASES\u00A0None"
        write_ret_str += "\n\u00A0"

        if not filename:
            return write_ret_str
        if filename.find(".cspn") == -1:
            filename += ".cspn"
        with open(filename, "w+") as file:
            file.write(write_ret_str)
            file.close()


    @staticmethod
    def from_save(context: str, file_load: bool = False) -> 'LayerNorm':
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
        LayerNorm
            A new `LayerNorm` layer containing all of the information encoded in the string or file provided.
        """
        def parse_and_return(handled_str: str):
            data_arr = handled_str.splitlines()
            params = data_arr[0].split("\u00A0")
            in_size = tuple(map(int, params[1].split()))
            eps = float(params[2])
            opt = parse_opt_info(params[-1])

            weight_data, bias_data = data_arr[1].split("\u00A0"), data_arr[2].split("\u00A0")
            weights = None if weight_data[1] == "None" else np.array(list(map(float, weight_data[1].split())))
            biases = None if bias_data[1] == "None" else np.array(list(map(float, bias_data[1].split())))

            new_neuron = LayerNorm(in_size,
                                   False if weights is None else True,
                                   False if biases is None else True,
                                   eps, opt)
            new_neuron.layer_weight = weights
            new_neuron.bias_weight = biases
            return new_neuron

        if file_load:
            full_parse_str = None
            with open(context, "r") as file:
                full_parse_str = file.read()
                file.close()
            return parse_and_return(full_parse_str)
        return parse_and_return(context)
from ..cudalib import np
from . import Layer
from ..optimizers import Optimizer, StandardGD, parse_opt_info
from ..utilities import check_types, InvalidDataException

class GroupNorm(Layer):
    """
    A grouped channel-based normalization layer which normalizes data across a specified number of channels.

    Supports any given shape and dimensionality as an input, as long as the data has at least 2 dimensions for
    batches and channels (where channels must match the expected amount).
       
        
    Attributes
    ---------
    groups : int
        The number of groups that the channel dimension will be split into before processing.
    channels : int
        The number of expected channels that the input data will have.
    channel_divs : int
        The number of expected channels divided by the number of groups, or the number of expected 
        channels in each group.
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
    >>> layer1 = GroupNorm(3, 6)
    >>> in_arr = np.random.randn(10, 6, 5, 5)
    >>> out_arr = layer1(in_arr)
    >>> print(out_arr.shape)
    (10, 6, 5, 5)
    """
    @check_types(("groups", lambda x: x > 0, "Argument \"groups\" must be above 0."),
                 ("channels", lambda x: x > 0, "Argument \"channels\" must be above 0."),
                 ("var_eps", lambda x: x > 0.0, "Argument \"var_eps\" must be above 0.0."))
    def __init__(self, groups: int, channels: int, weights: bool = True,
                 biases: bool = False, var_eps: float = 1e-8,
                 optimizer: Optimizer = StandardGD()):
        """
        Initializes a `GroupNorm` layer using given parameters.

        Parameters
        ----------
        groups : int
            The number of groups that the channel dimension will be split into before processing.
        channels : int
            The number of expected channels that the input data will have.
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
        if channels % groups != 0:
            raise InvalidDataException(f"Channels must be divisible by groups. - {channels} -> {groups}")
        super().__init__(None, None)
        self.channels = channels
        self.groups = groups
        self.channel_divs = channels // groups
        self.layer_weight = np.ones((self.groups, self.channel_divs, 1)) if weights is True else None
        self.bias_weight = np.zeros((self.groups, self.channel_divs, 1)) if biases is True else None

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
        if len(data.shape) < 2:
            raise InvalidDataException(f"Data must have at least 2 dimensions. - {data.shape}")
        if data.shape[1] != self.channels:
            raise InvalidDataException(f"Data channels must be equal to the expected amount. - {data.shape}, {self.channels}")
        shaped_data = data.reshape((data.shape[0], self.groups, self.channel_divs, -1))

        # Mean and variance of batches
        layer_mean = np.mean(shaped_data, axis=(-1, -2), keepdims=True)
        layer_var = np.var(shaped_data, axis=(-1, -2), keepdims=True)

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
        shaped_err = cost_err.reshape((cost_err.shape[0], self.groups, self.channel_divs, -1))
        if self.layer_weight is not None:
            shaped_err *= self.layer_weight

        opt_err = self.opt.process_grad(shaped_err)   # Weight/Bias update gradient with optimizer

        ret_grad = (shaped_err - shaped_err.mean(axis=-1, keepdims=True) 
                    - self.__norm_res * (shaped_err * self.__norm_res).mean(axis=-1, keepdims=True)) \
                    / self.__last_stdv
        
        # Update weights and biases (if applicable)
        if self.layer_weight is not None:
            self.layer_weight += np.squeeze((opt_err * self.__norm_res).sum(axis=(0, -1), keepdims=True), 0)
    
        if self.bias_weight is not None:
            self.bias_weight += np.squeeze(opt_err.sum(axis=(0, -1), keepdims=True), 0)
        return ret_grad.reshape(cost_err.shape)
    

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


    def deepcopy(self) -> 'GroupNorm':
        """Creates a new deepcopy of this layer with the exact same weights (if applicable) and parameters."""
        new_neuron = GroupNorm(self.groups, self.channels,
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
        write_ret_str = f"GroupNorm\u00A0{self.groups}\u00A0{self.channels}\u00A0{self.var_eps}\u00A0{repr(self.opt)}"
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
    def from_save(context: str, file_load: bool = False) -> 'GroupNorm':
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
        GroupNorm
            A new `GroupNorm` layer containing all of the information encoded in the string or file provided.
        """
        def parse_and_return(handled_str: str):
            data_arr = handled_str.splitlines()
            params = data_arr[0].split("\u00A0")[1:]
            groups, channels, eps = int(params[0]), int(params[1]), float(params[2])
            opt = parse_opt_info(params[-1])

            weight_data, bias_data = data_arr[1].split("\u00A0"), data_arr[2].split("\u00A0")
            weights = None if weight_data[1] == "None" else np.array(list(map(float, weight_data[1].split())))
            biases = None if bias_data[1] == "None" else np.array(list(map(float, bias_data[1].split())))

            new_neuron = GroupNorm(groups, channels,
                                   weights is not None,
                                   biases is not None,
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
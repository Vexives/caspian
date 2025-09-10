from ..cudalib import np
from . import Layer
from ..optimizers import Optimizer, StandardGD, parse_opt_info
from ..utilities import check_types

class InstanceNorm(Layer):
    """
    An instance normalization layer which normalizes the given data across all dimensions
    except for the channels/features and the batches (if applicable).

    Supports any given shape and dimensionality as an input, as long as that number of dimensions is
    provided (batch dimension NOT included).
       
        
    Attributes
    ---------
    channels : int
        The number of channels or features that the expected inputs will have.
    dims : int
        The number of total dimensions (excluding the batch dimension) that the expected inputs will
        have.
    momentum : float
        The momentum value multiplied by the running mean and variance at each learning pass. If set
        to `None`, the running variables will not be initialized.
    var_eps : float
        A generally very small float which corresponds to the epsilon value added to the 
        square root variance during normalization.
    opt : Optimizer, optional
        The provided optimizer which modifies the learning gradient before updating weights.
    gamma : ndarray
        A single dimensional array of values that correspond to the learnable gamma values multiplied
        after normalization.
    beta : ndarray
        A single dimensional array of values that correspond to the learnable beta values added
        after normalization.
    running_mean : ndarray
        A single dimensional array of values that correspond to the running mean values calculated
        during the learning phase. Will be set to `None` if `momentum` is `None`.
    running_var : ndarray
        A single dimensional array of values that correspond to the running variance values calculated
        during the learning phase. Will be set to `None` if `momentum` is `None`.


    Examples
    --------
    >>> layer1 = InstanceNorm(5, 3)
    >>> in_arr = np.random.randn(4, 5, 10, 10)
    >>> out_arr = layer1(in_arr)
    >>> print(out_arr.shape)
    (4, 5, 10, 10)
    """
    @check_types(("channels", lambda x: x > 0, "Argument \"channels\" must be greater than 0."),
                 ("dimensions", lambda x: x > 0, "Argument \"dimensions\" must be greater than 0."),
                 ("momentum", lambda x: x is None or 0.0 < x < 1.0, "Argument \"momentum\" must be between 0.0 and 1.0."),
                 ("var_eps", lambda x: x > 0.0, "Argument \"var_eps\" must be greater than 0.0."))
    def __init__(self, channels: int, dimensions: int, 
                 scale: bool = True, shift: bool = True, 
                 momentum: float | None = 0.9, var_eps: float = 1e-8, 
                 optimizer: Optimizer = StandardGD()) -> None:
        """
        Initializes an `InstanceNorm` layer using given parameters.

        Parameters
        ----------
        channels : int
            The number of channels expected by the input arrays.
        dimensions : int
            The number of dimensions that this layer should expect from the inputs and gradients given.
            The batch dimension is NOT counted in this number.
        scale : bool, default: True
            A boolean which determines whether the learnable gamma parameter array is initialized.
        shift : bool, default: True
            A boolean which determines whether the learnable beta parameter array is initialized.
        momentum : float | None, default: 0.9
            The momentum of the running mean and variance of this layer. Set to `None` for the running
            variables to not be used.
        var_eps : float, default: 1e-8
            A float which corresponds to the epsilon value added to the square root variance during normalization.
        optimizer : Optimizer, default: StandardGD()
            An optimizer class which processes given loss gradients and adjusts them to match a desired 
            gradient descent path.        
        """
        super().__init__(None, None)
        self.channels = channels
        self.dims = dimensions
        self.var_eps = var_eps
        self.opt = optimizer

        self.momentum = momentum
        self.running_mean = np.zeros((channels, 1)) if self.momentum is not None else None
        self.running_var = np.ones((channels, 1)) if self.momentum is not None else None

        self.gamma = np.ones((channels, 1)) if scale is True else None
        self.beta = np.zeros((channels, 1)) if shift is True else None


    def forward(self, data: np.ndarray, training: bool = False) -> np.ndarray:
        """
        Performs a forward propagation pass through this layer given the current gamma and beta values
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
            The forward propagated array with all values normalized.
        """        
        batch_data = np.expand_dims(data, axis=0) if len(data.shape) == self.dims else data
        shaped_data = batch_data.reshape((*batch_data.shape[:2], -1))

        #Mean and variance of batches
        batch_mean = np.mean(shaped_data, axis=-1, keepdims=True)
        batch_var = np.var(shaped_data, axis=-1, keepdims=True)

        #Training of running mean and running variance (if applicable)
        if training:
            self.running_mean = self.momentum * self.running_mean + ((1 - self.momentum) * batch_mean) \
                                if self.momentum is not None else self.running_mean
            self.running_var = self.momentum * self.running_var + ((1 - self.momentum) * batch_var) \
                                if self.momentum is not None else self.running_var
            
            new_data = ((shaped_data - batch_mean) / (np.sqrt(batch_var + self.var_eps)))
            
            self.__last_in = shaped_data
            self.__norm_res = new_data
            self.__last_v = batch_var
            self.__last_m = batch_mean
        else:
            if self.momentum:
                batch_mean = self.running_mean
                batch_var = self.running_var
            new_data = ((shaped_data - batch_mean) / (np.sqrt(batch_var + self.var_eps)))    

        new_data = self.gamma * new_data if self.gamma is not None else new_data
        new_data = new_data + self.beta if self.beta is not None else new_data

        return new_data.reshape(data.shape)
    

    def backward(self, cost_err: np.ndarray) -> np.ndarray:
        """
        Performs a backward propagation pass through this layer and updates the gamma and beta (if applicable) 
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
        #Update gamma and beta with optimzied gradient (if applicable)
        batch_err = np.expand_dims(cost_err, axis=0) if len(cost_err.shape) == self.dims else cost_err
        shaped_err = batch_err.reshape((*batch_err.shape[:2], -1))
        shaped_err *= self.gamma if self.gamma is not None else 1

        opt_err = self.opt.process_grad(shaped_err)                   #Gamma/Beta update gradient with optimizer

        elem_num = 1.0 / shaped_err.shape[0]
        d_m = (shaped_err * (-self.__last_v)).mean(axis=-1, keepdims=True)
        d_v = (shaped_err * (self.__last_in - self.__last_m)).sum(axis=-1, keepdims=True) * \
              (-0.5 * (self.__last_v + self.var_eps)**-1.5)

        ret_grad = shaped_err * self.__last_v + d_v * 2 * (self.__last_in - self.__last_m) * elem_num \
                   + d_m * elem_num
        
        if self.gamma is not None:
            self.gamma += np.squeeze((opt_err * self.__norm_res).sum(axis=(0, -1), keepdims=True), 0)
    
        if self.beta is not None:
            self.beta += np.squeeze(opt_err.sum(axis=(0, -1), keepdims=True), 0)
        return ret_grad.reshape(cost_err.shape)
    

    def step(self) -> None:
        """Adds one step to this layer's optimizer and scheduler."""
        self.opt.step()
    

    def clear_grad(self) -> None:
        """Clears the optimizer gradient history and deletes any data required by the backward pass."""
        self.__last_in = None
        self.__last_m = None
        self.__last_v = None
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


    def deepcopy(self) -> 'InstanceNorm':
        """Creates a new deepcopy of this layer with the exact same weights (if applicable) and parameters."""
        new_neuron = InstanceNorm(self.channels, self.dims,
                                  self.gamma is not None, self.beta is not None,
                                  self.momentum, self.var_eps, self.opt.deepcopy())
        new_neuron.gamma = self.gamma.copy() if self.gamma is not None else None
        new_neuron.beta = self.beta.copy() if self.beta is not None else None
        new_neuron.running_mean = self.running_mean.copy() if self.running_mean is not None else None
        new_neuron.running_var = self.running_var.copy() if self.running_var is not None else None
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
        write_ret_str = f"InstanceNorm\u00A0{self.channels}\u00A0{self.dims}\u00A0{repr(self.opt)}" + \
                        f"\nPARAMS\u00A0{self.momentum}\u00A0{self.var_eps}"
        write_ret_str += f"\nGAMMA\u00A0" + " ".join(list(map(str, self.gamma.flatten().tolist()))) if self.gamma is not None else "\nGAMMA\u00A0None"
        write_ret_str += f"\nBETA\u00A0" + " ".join(list(map(str, self.beta.flatten().tolist()))) if self.beta is not None else "\nBETA\u00A0None"
        write_ret_str += f"\nRMEAN\u00A0" + " ".join(list(map(str, self.running_mean.flatten().tolist()))) if self.running_mean is not None else "\nRMEAN\u00A0None"
        write_ret_str += f"\nRVAR\u00A0" + " ".join(list(map(str, self.running_var.flatten().tolist()))) if self.running_var is not None else "\nRVAR\u00A0None"
        write_ret_str += "\n\u00A0"

        if not filename:
            return write_ret_str
        if filename.find(".cspn") == -1:
            filename += ".cspn"
        with open(filename, "w+") as file:
            file.write(write_ret_str)
            file.close()


    @staticmethod
    def from_save(context: str, file_load: bool = False) -> 'InstanceNorm':
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
        InstanceNorm
            A new `InstanceNorm` layer containing all of the information encoded in the string or file provided.
        """
        def parse_and_return(handled_str: str):
            data_arr = handled_str.splitlines()
            gen_info = data_arr[0].split("\u00A0")[1:]
            channels, dims = tuple(map(int, gen_info[:-1]))
            opt = parse_opt_info(gen_info[-1])
            momentum, eps = tuple(map(float, data_arr[1].split("\u00A0")[1:]))

            gamma_data, beta_data = data_arr[2].split("\u00A0"), data_arr[3].split("\u00A0")
            gamma = None if gamma_data[1] == "None" else np.array(list(map(float, gamma_data[1].split()))).reshape((channels,))
            beta = None if beta_data[1] == "None" else np.array(list(map(float, beta_data[1].split()))).reshape((channels,))
            
            rm_data, rv_data = data_arr[4].split("\u00A0"), data_arr[5].split("\u00A0")
            r_mean = None if rm_data[1] == "None" else np.array(list(map(float, rm_data[1].split()))).reshape((channels,))
            r_var = None if rv_data[1] == "None" else np.array(list(map(float, rv_data[1].split()))).reshape((channels,))

            new_neuron = InstanceNorm(channels, dims, 
                                      gamma is not None,
                                      beta is not None, 
                                      momentum, eps, opt)
            new_neuron.gamma = gamma
            new_neuron.beta = beta
            new_neuron.running_mean = r_mean
            new_neuron.running_var = r_var
            return new_neuron

        if file_load:
            full_parse_str = None
            with open(context, "r") as file:
                full_parse_str = file.read()
                file.close()
            return parse_and_return(full_parse_str)
        return parse_and_return(context)
from ..cudalib import np
from . import Layer
from ..optimizers import Optimizer, StandardGD, parse_opt_info

class Linear(Layer):
    """
    A singular linear layer which performs a simple linear transformation using weights and/or biases.
    A more basic version of `Dense`, does not take any activation function.

    Supports any given shape and dimensionality as an input, as long as that shape is given in the 
    initial parameters.

    Result = ``W @ x + b``

        
    Attributes
    ---------
    layer_weight : ndarray
        The `(O, X)` shape array of weights that are matrix multiplied by the input
        to grant the desired output shape. Trainable parameters that are modified
        after each backward pass.
    bias_weight : ndarray
        The `(O, 1)` shape array of bias weights that are applied after the initial
        matrix multiplication. Trainable parameters that are modified after each
        backward pass. Will be set to `None` if not enabled during initialization.
    in_size : tuple[int, ...]
        A tuple containing the expected input size `(N, ..., X)`, where `N` is the number of batches, 
        `...` is any intermediate dimension, and `X` is the expected length of the input.
    out_size : tuple[int, ...]
        A tuple containing the expected output size `(N, ..., O)`, where `N` and `...` are the same 
        as the input, with `O` representing the length of the output.
    opt : Optimizer
        The provided optimizer which modifies the learning gradient before updating weights.


    Examples
    --------
    >>> layer1 = Linear(10, 5)
    >>> in_arr = np.ones((10,))
    >>> out_arr = layer1(in_arr)
    >>> print(out_arr.shape)
    (5,)
    """
    def __init__(self, inputs: tuple[int, ...] | int, outputs: int, biases: bool = False,
                 optimizer: Optimizer = StandardGD()) -> None:
        """
        Initializes a `Linear` layer using given parameters.

        Parameters
        ----------
        funct : Activation
            An activation function class which supports both forward and backward non-linear 
            transformations.
        inputs : int | tuple[int, ...]
            An integer or tuple of integers matching the shape of the expected input arrays.
        outputs : int
            An integer representing the length of the final dimension of the output.
        optimizer : Optimizer, default: StandardGD()
            An optimizer class which processes given loss gradients and adjusts them to match a desired 
            gradient descent path.
        """
        in_size = inputs if isinstance(inputs, tuple) else (inputs,)
        out_size = (*in_size[:-1], outputs,)
        super().__init__(in_size, out_size)

        self.layer_weight = np.random.uniform(-0.5, 0.5, (outputs, in_size[-1]))
        self.bias_weight = np.zeros((outputs, 1)) if biases else None

        self.opt = optimizer


    def forward(self, data: np.ndarray, training: bool = False) -> np.ndarray:
        """
        Performs a forward propagation pass through this layer given the current weights and biases
        (if applicable).
        
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
        data = np.expand_dims(data, axis=-1)
        new_val = self.layer_weight @ data

        if self.bias_weight is not None:
            new_val += self.bias_weight
        if training:
            self.__last_in = data
            self.__last_out = new_val
        return new_val.squeeze(axis=-1)
    

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
        cost_err = np.expand_dims(cost_err, axis=-1)
        new_err = cost_err * self.__last_out
        
        new_grad = self.opt(new_err)
        ret_grad = (np.transpose(self.layer_weight) @ new_err).squeeze(axis=-1)

        layer_grad = new_grad @ np.moveaxis(self.__last_in, -1, -2)
        self.layer_weight += layer_grad if len(layer_grad.shape) == 2 else \
                             layer_grad.reshape(-1, *layer_grad.shape[-2:]).sum(axis=0)
        
        if self.bias_weight is not None:
            self.bias_weight += new_grad if len(new_grad.shape) == 2 else \
                                new_grad.reshape(-1, *new_grad.shape[-2:]).sum(axis=0)
        return ret_grad
    

    def step(self) -> None:
        """Adds one step to this layer's optimizer and scheduler."""
        self.opt.step()
    

    def clear_grad(self) -> None:
        """Clears the optimizer gradient history and deletes any data required by the backward pass."""
        self.__last_in = None
        self.__last_out = None
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
    

    def deepcopy(self) -> 'Linear':
        """Creates a new deepcopy of this layer with the exact same weights (if applicable) and parameters."""
        new_neuron = Linear(self.in_size, self.out_size[-1], optimizer=self.opt.deepcopy())
        new_neuron.layer_weight = self.layer_weight.copy()
        new_neuron.bias_weight = self.bias_weight.copy() if self.bias_weight is not None else None
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
        write_ret_str = f"Linear\u00A0{repr(self.opt)}" + \
                        "\nWEIGHTS\u00A0" + " ".join(list(map(str, self.layer_weight.flatten().tolist()))) + \
                        "\nBIAS\u00A0" + (" ".join(list(map(str, self.bias_weight.flatten().tolist()))) \
                                          if self.bias_weight is not None else "None") + \
                        "\nSIZES\u00A0" + " ".join(list(map(str, self.in_size))) + f"\u00A0{self.out_size[-1]}\n\u00A0" 
        if not filename:
            return write_ret_str
        if filename.find(".cspn") == -1:
            filename += ".cspn"
        with open(filename, "w+") as file:
            file.write(write_ret_str)
            file.close()


    @staticmethod
    def from_save(context: str, file_load: bool = False) -> 'Linear':
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
        Linear
            A new `Linear` layer containing all of the information encoded in the string or file provided.
        """
        def parse_and_return(handled_str: str):
            data_arr = handled_str.splitlines()
            prop_info = data_arr[0].split("\u00A0")
            size_info = data_arr[-2].split("\u00A0")

            in_size = tuple(map(int, size_info[1].split()))
            out_size = int(size_info[2])
            opt = parse_opt_info(prop_info[-1])

            weight_info, bias_info = data_arr[1].split("\u00A0")[1], data_arr[2].split("\u00A0")[1]
            weights = np.array(list(map(float, weight_info.split()))).reshape((out_size, in_size[-1]))
            biases = np.array(list(map(float, bias_info.split()))).reshape((out_size, 1)) \
                             if bias_info != "None" else None

            new_neuron = Linear(in_size, out_size, opt)
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
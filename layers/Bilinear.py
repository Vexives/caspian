from caspian.cudalib import np
from . import Dense
from caspian.optimizers import Optimizer, StandardGD
from caspian.activations import Activation, parse_act_info

class Bilinear(Dense):
    """
    A bilinear dense layer which performs a dual-linear transformation of the input data provided.

    Supports any given shape and dimensionality as an input, as long as both input shapes match
    in every dimension except for the last.

    Result = ``funct(y @ (W @ x) + b)``


    Notes
    -----
    This model can NOT be put into a standard Sequence layer, as it requires more than one input
    to function properly. A custom model which incorporates this layer must be created for it to
    function in a Sequence.

        
    Attributes
    ---------
    layer_weight : ndarray
        The `(O, X, Y)` shape array of weights that are matrix multiplied by the input
        to grant the desired output shape. Trainable parameters that are modified
        after each backward pass. `O` represents the number of outputs, `X` represents the
        length of the first set of data, and `Y` represents the length of the second set of data.
    bias_weight : ndarray
        The `(O, 1)` shape array of bias weights that are applied after the initial
        matrix multiplication. Trainable parameters that are modified after each
        backward pass.
    in_size : tuple[tuple[int, ...], tuple[int, ...]]
        A tuple containing the expected input sizes `((N, ..., X), (N, ..., Y))`, where `N` 
        is the number of batches, `...` is any intermediate dimension, and `X` / 'Y' are the 
        expected lengths of the inputs respectively.
    out_size : tuple[int, ...]
        A tuple containing the expected output size `(N, ..., O)`, where `N` and `...` are the same 
        as the input, with `O` representing the length of the output.
    funct : Activation
        The given activation class which performs the desired non-linear transform to the data.
    opt : Optimizer
        The provided optimizer which modifies the learning gradient before updating weights.


    Examples
    --------
    >>> layer1 = Bilinear(ReLU(), 10, 20, 5)
    >>> in_arr_1 = np.ones((10,))
    >>> in_arr_2 = np.ones((20,))
    >>> out_arr = layer1(in_arr_1, in_arr_2)
    >>> print(out_arr.shape)
    (5,)
    """

    def __init__(self, funct: Activation, inputs_1: tuple[int, ...] | int,
                 inputs_2: tuple[int, ...] | int, outputs: int, 
                 optimizer: Optimizer = StandardGD()):
        """
        Initializes a `Bilinear` dense layer using the given parameters.

        Parameters
        ----------
        funct : Activation
            An activation function class which supports both forward and backward non-linear 
            transformations.
        inputs_1 : int | tuple[int, ...]
            An integer or tuple of integers matching the shape of the first set of expected input arrays.
        inputs_2 : int | tuple[int, ...]
            An integer or tuple of integers matching the shape of the second set of expected input arrays.
        outputs : int
            An integer representing the expected length of the final dimension of the output. 
        optimizer : Optimizer, default: StandardGD()
            An optimizer class which processes given loss gradients and adjusts them to match a desired 
            gradient descent path.
        """
        first_ins = inputs_1 if isinstance(inputs_1, tuple) else (inputs_1,)
        second_ins = inputs_2 if isinstance(inputs_2, tuple) else (inputs_2,)

        assert first_ins[:-1] == second_ins[:-1], \
        f"Input shape must be equal except for last dimension: {first_ins} - {second_ins}"

        self.layer_weight = np.random.uniform(-0.5, 0.5, (outputs, first_ins[-1], second_ins[-1]))
        self.bias_weight = np.zeros((outputs,))

        in_size = (first_ins, second_ins)
        out_size = (*first_ins[:-1], outputs)
        super().__init__(in_size, out_size)
        
        self.funct = funct
        self.opt = optimizer

    
    def __call__(self, data_1: np.ndarray, data_2: np.ndarray, training: bool = False) -> np.ndarray:
        """Calls the class forward function and provides the given parameters."""
        return self.forward(data_1, data_2, training)
    

    def forward(self, data_1: np.ndarray, data_2: np.ndarray, training: bool = False) -> np.ndarray:
        """
        Performs a forward propagation pass through this layer given the current weights and biases.
        
        Parameters
        ----------
        data_1 : ndarray
            The first data set that the forward pass will be performed on. Must match the input size 
            of this layer.
        data_2 : ndarray
            The second data set that the forward pass will be performed on. Must match the second input 
            size of this layer.
        training : bool
            Specify whether the layer is currently training or not to save the necessary information
            required for the backward pass.
        
        Returns
        -------
        ndarray
            The forward propagated array with the shape equal to this layer's output shape.
        """
        new_val = self.funct(np.einsum("...i,oij,...j->...o", 
                                       data_1, 
                                       self.layer_weight, 
                                       data_2) + self.bias_weight)
        if training:
            self.__last_in = (data_1, data_2)
            self.__last_out = new_val
        return new_val
    
    
    def backward(self, cost_err: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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
        tuple[ndarray, ndarray]
            Two new learning gradient for any layers that provided data to this instance. Both will have the
            same shapes as this layer's input shapes. The first gradient corresponds to the first input,
            and the second gradient corresponds to the second input.
        """
        new_err = cost_err * self.funct(self.__last_out, True)
        new_grad = self.opt(new_err)
        layer_grad = np.einsum("...o,...i,...j->...oij", 
                               new_grad, 
                               self.__last_in[0], 
                               self.__last_in[1])
        
        first_ret_grad = np.einsum("...o,oij->...i", new_err, self.layer_weight)
        second_ret_grad = np.einsum("...o,oij->...j", new_err, self.layer_weight)

        self.layer_weight += layer_grad if len(layer_grad.shape) == 3 else \
                             layer_grad.reshape(-1, *layer_grad.shape[-3:]).sum(axis=0)
        self.bias_weight += new_grad.reshape(-1, new_grad.shape[-1]).sum(axis=0)
        return first_ret_grad, second_ret_grad
    

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


    def deepcopy(self) -> 'Bilinear':
        """Creates a new deepcopy of this layer with the exact same weights (if applicable) and parameters."""
        new_neuron = Bilinear(self.funct, self.in_size[0], self.in_size[1], self.out_size[-1], self.opt.deepcopy())
        new_neuron.layer_weight = self.layer_weight.copy()
        new_neuron.bias_weight = self.bias_weight.copy()
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
        write_ret_str = f"Bilinear\u00A0{repr(self.funct)}\n" + \
                        "".join([" ".join(list(map(str, row.tolist()))) + "\n" for row in 
                                          self.layer_weight.reshape(-1, self.layer_weight.shape[-1])]) + \
                        "BIAS\n" + \
                        " ".join(list(map(str, self.bias_weight.flatten().tolist()))) + "\n" \
                        "SIZES\u00A0" + " ".join(list(map(str, self.in_size[0]))) + "\u00A0" + \
                                        " ".join(list(map(str, self.in_size[1]))) + "\u00A0" + \
                                        f"{self.out_size[-1]}\n\u00A0" 
        if not filename:
            return write_ret_str
        if filename.find(".cspn") == -1:
            filename += ".cspn"
        with open(filename, "w+") as file:
            file.write(write_ret_str)
            file.close()


    @staticmethod
    def from_save(context: str, file_load: bool = False) -> 'Bilinear':
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
        Bilinear
            A new `Bilinear` layer containing all of the information encoded in the string or file provided.
        """
        def parse_and_return(handled_str: str):
            data_arr = handled_str.splitlines()
            prop_info = data_arr[0].split("\u00A0")
            size_info = data_arr[-2].split("\u00A0")
            in_size_1 = tuple(map(int, size_info[1].split()))
            in_size_2 = tuple(map(int, size_info[2].split()))
            out_size = int(size_info[3])
            new_neuron = Bilinear(parse_act_info(prop_info[1]), in_size_1, in_size_2, out_size)

            weights = []
            biases = []
            bias_bool = False
            for line in data_arr[1:]:
                if line.find("BIAS") != -1:
                    bias_bool = True
                    continue
                if line.find("SIZES") != -1:
                    new_neuron.layer_weight = np.array(weights, dtype=float).reshape(new_neuron.layer_weight.shape)
                    new_neuron.bias_weight = np.array(biases, dtype=float).reshape(-1, 1)
                    break
                if not bias_bool:
                    weights.append(list(map(float, line.split())))
                    continue
                biases.append(list(map(float, line.split())))
            return new_neuron

        if file_load:
            full_parse_str = None
            with open(context, "r") as file:
                full_parse_str = file.read()
                file.close()
            return parse_and_return(full_parse_str)
        return parse_and_return(context)
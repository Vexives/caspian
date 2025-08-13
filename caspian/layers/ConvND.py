from ..cudalib import np
from . import Layer
from ..optimizers import Optimizer, StandardGD, parse_opt_info
from ..activations import Activation, parse_act_info
from ..utilities import dilate_array, all_positive, all_ints, confirm_shape, check_types, \
                        InvalidDataException, UnsafeMemoryAccessException

class ConvND(Layer):
    """
    An any-dimensional convolutional layer which performs a downwards convolution transform on the data provided.

    Supports data of any size or shape containing at least 2 dimensions (one for channels, at least one for convolving).
    The expected size must be specified in the `input_size` argument upon initialization.
    All other parameters which are dependent on size, such as `kernel_size`, `padding`, and `strides`, must
    be either integers or tuples with one less dimension than `input_size`.

    
    Memory Safety
    -------------
    This layer is not memory safe if modified. Be extremely careful when modifying any sort of
    variable of this layer, as it may cause memory dangers if done incorrectly.


    Attributes
    ---------
    in_size : tuple[int, ...]
        A tuple containing the expected input size `(C, *)`, where `C` is the number of channels, 
        and * represents the final dimensions of the input (at least 1). Will have a total of `N` dimensions.
    out_size : tuple[int, ...]
        A tuple containing the expected output size `(F, *)`, where `F` represents the new 
        number of channels, and `*` represents the new output dimension size. Will have `N` dimensions.
    funct : Activation
        The given activation function which takes specific data from each partition of the input.
    pad_details : tuple[tuple[int, int], ...]
        A tuple of tuples, defining the padding across each of the convolved dimensions given.
    padding_all : tuple[int, ...]
        The padding input value in tuple form that was provided at initialization. If the padding was given
        as an integer, then the tuple will be of size `N-1`
    strides_all : tuple[int, ...]
        A tuple of integers representing the strides for each convolved dimension. Will have a size of `N-1`.
    kernel_size : tuple[int, ...]
        A tuple of integers representing the kernel size for each convolved dimension. Will have a size of `N-1`.
    kernel_weights : ndarray
        A set of trainable kernel weights which are applied to each partition extracted from the 
        given input. Has the shape `(F, C, k*)`, with `k*` being the `N-1` shape kernel dimensions listed
        in `kernel_size`.
    input_length : int
        An integer representing the total number of dimensions (including channels) that is expected from
        the input data. 
    bias_weights : ndarray | None
        A set of trainable bias weights which are applied to the final result of the convolution.
        Matches the expected output shape, and can be disabled if specified.
    use_bias : bool
        A boolean which determines if the bias weights are initialized, used, and trained.
    opt : Optimizer
        The provided optimizer which modifies the learning gradient before updating weights.


    Examples
    --------
    >>> layer1 = ConvND(ReLU(), 2, 3, (5, 9, 10, 8, 8), strides=1)
    >>> in_arr = np.random.uniform(0.0, 1.0, (4, 5, 9, 10, 8, 8))
    >>> out_arr = layer1(in_arr)
    >>> print(out_arr.shape)
    (4, 2, 7, 8, 6, 6)
    """
    @check_types(("layers", lambda x: x > 0, "Argument \"layers\" must be greater than 0."),
                 
                 ("kernel_size", all_positive, "Argument \"kernel_size\" must be greater than 0."),
                 ("kernel_size", all_ints, "Argument \"kernel_size\" must contain all integers."),
                 
                 ("strides", all_positive, "Argument \"strides\" must be greater than 0."),
                 ("strides", all_ints, "Argument \"strides\" must contain all integers."),
                 
                 ("padding", lambda x: all_positive(x, True), "Argument \"padding\" must be greater than or equal to 0."),
                 ("padding", all_ints, "Argument \"padding\" must contain all integers."),
                 
                 ("input_size", all_positive, "Argument \"input_size\" must contain all positive values above 0."),
                 ("input_size", lambda x: len(x) >= 2, "Argument \"input_size\" must have at least one channel and one convolution dimension."))
    def __init__(self, funct: Activation, layers: int, 
                 kernel_size: tuple[int, ...] | int,
                 input_size: tuple[int, ...], 
                 strides: tuple[int, ...] | int = 1, 
                 padding: tuple[int, ...] | int = 0, 
                 biases: bool = True, optimizer: Optimizer = StandardGD()):
        """
        Initializes a `ConvND` layer using given parameters.

        Parameters
        ----------
        funct : Activation
            An activation function class which supports both forward and backward non-linear
            transformations.
        layers : int
            The number of resulting channels that the output array will have. Will represent the new
            `C` dimension size for the output.
        kernel_size : tuple[int, ...] | int
            An integer or tuple of integers representing the size of the sliding window to extract 
            partitions of the input data.
        input_size : tuple[int, ...]
            A tuple of integers matching the shape of the expected input arrays.
        strides : tuple[int, ...] | int, default: 1
            An integer or tuple of integers that determines how many data points are skipped for 
            every iteration of the sliding window. Must be greater than or equal to 1.
        padding : tuple[int, ...] | int, default: 0
            An integer or tuple of integers that determines how many empty data points are put on 
            the edges of the final dimensions as padding layers.
        biases : bool, default: True
            Determines whether the bias values for this layer are initialized, used, and trained.
        opt : Optimizer, default: StandardGD()
            An optimizer class which processes given loss gradients and adjusts them to match a desired 
            gradient descent path.

        Raises
        ------
        InvalidDataException
            If any of the data provided is not an integer, tuple of integers, or less than one (with the exception 
            of padding, which can be 0), or if the given function is not of type `Activation`. 
            Also applies to the expected input shape, which must be a tuple of integers.
            Padding, strides, and kernel size must ALL be either an integer or a tuple with N-1 dimensions, where
            input size has N > 2 dimensions. 
        """ 
        # Extra dimensionality checks before processing
        _in_len = len(input_size)-1
        if type(kernel_size) != int and len(kernel_size) != _in_len:
            raise InvalidDataException("Argument \"kernel_size\" must have one less dimension than expected input.")
        if type(strides) != int and len(strides) != _in_len:
            raise InvalidDataException("Argument \"strides\" must have one less dimension than expected input.")
        if type(padding) != int and len(padding) != _in_len:
            raise InvalidDataException("Argument \"strides\" must have one less dimension than expected input.")            

        # Stride, Kernel, and Padding Initialization
        self.padding_all = (padding,) * _in_len if type(padding) == int else padding
        self.pad_details = ((0,0), (0,0)) + tuple(((p+1)//2, p//2) for p in self.padding_all)

        self.strides_all = (strides,) * _in_len if type(strides) == int else strides
        self.kernel_size = (kernel_size,) * _in_len if type(kernel_size) == int else kernel_size

        # Other settings
        self.funct = funct
        self.opt = optimizer
        self.use_bias = biases
        self.input_length = len(input_size)

        #In/Out Sizes
        in_size = input_size
        out_size = (layers,) + \
                   tuple((in_size[i+1] - self.kernel_size[i] + self.padding_all[i]) // self.strides_all[i] + 1
                         for i in range(_in_len))
        super().__init__(in_size, out_size)

        #Strides and Window Shapes
        self.__window_shape = (layers, 
                               self.in_size[0], 
                               *self.kernel_size, 
                               *self.out_size[1:]) #(F, C, kernel_Dims..., out_Shape[1:]...)
        self.__grad_shape = (self.in_size[0], 
                             layers, 
                             *self.kernel_size, 
                             *tuple(map(sum, zip(self.in_size[1:], self.padding_all)))) #(C, F, kernel_Dims..., in_Dims[1:]... + padding)
        self.__dx_shape = (self.out_size[0],) + \
                          tuple(self.out_size[i+1] + (self.out_size[i+1]-1) * (self.strides_all[i]-1) 
                                for i in range(_in_len)) #(F, out_Dims[1:]... + dilate)
        
        self.kernel_weights = np.random.uniform(-0.5, 0.5, (layers, self.in_size[0], *self.kernel_size))
        self.bias_weights = np.zeros(self.out_size) if biases is True else None


    def forward(self, data: np.ndarray, training: bool = False) -> np.ndarray:
        """
        Performs a forward propagation pass through this layer given the current weights and biases.
        
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
            The forward propagated array with the shape equal to this layer's output shape.

        Raises
        ------
        UnsafeMemoryAccessException
            If the shape of the given array will lead to any un-safe memory calls during the pass.
        """
        if not confirm_shape(data.shape, self.in_size, self.input_length):
            raise UnsafeMemoryAccessException(f"Input data shape does not match expected shape. - {data.shape}, {self.in_size}")
        new_data = np.expand_dims(data, axis=0) if len(data.shape) < self.input_length+1 else data    #Enforce batches.
        new_data = np.pad(new_data, self.pad_details, mode="constant")
        
        #Sliding view strides and window shape
        data_strides = (new_data.strides[0],
                        0, 
                        *new_data.strides[1:], 
                        *tuple(map(lambda x,y: x*y, self.strides_all, new_data.strides[2:])))
        data_win_shape = (new_data.shape[0],) + self.__window_shape

        #Sliding windows and convolution multiplication
        sliding_view = np.lib.stride_tricks.as_strided(new_data, 
                                                       shape=data_win_shape, 
                                                       strides=data_strides)

        view_match = (None, Ellipsis) + ((None,) * (self.input_length-1))
        conv_val = (sliding_view * self.kernel_weights[view_match]).sum(axis=tuple(range(2, self.input_length+2)))

        last = self.funct(conv_val + self.bias_weights if self.use_bias is True else conv_val)
        if training:
            self.__last_in = new_data
            self.__last_out = last

        if len(data.shape) < self.input_length+1:
            last = last.squeeze(axis=0)
        return last
    

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

        Raises
        ------
        UnsafeMemoryAccessException
            If the shape of the given array will lead to any un-safe memory calls during the pass.
        """
        if not confirm_shape(cost_err.shape, self.out_size, self.input_length):
            raise UnsafeMemoryAccessException(f"Gradient data shape does not match expected shape. - {cost_err.shape}, {self.out_size}")
        new_err = np.expand_dims(cost_err, axis=0) if len(cost_err.shape) < self.input_length+1 else cost_err   #Enforce batches.
        new_err = self.funct(self.__last_out, new_err)
        opt_grad = self.opt.process_grad(new_err)

        main_strides = (self.__last_in.strides[0],
                        0, 
                        *self.__last_in.strides[1:],
                        *tuple(map(lambda x,y: x*y, self.strides_all, self.__last_in.strides[2:])))
        main_win_shape = (new_err.shape[0],) + self.__window_shape

        #Input gradient
        flipped_weights = np.flip(self.kernel_weights, axis=tuple(range(2, self.input_length+1)))
        flipped_weights = np.swapaxes(flipped_weights, 0, 1)
        dx_ins = dilate_array(new_err, (new_err.shape[0],) + self.__dx_shape, self.strides_all)

        p_x_pre = (self.__last_in.shape[i+2] + self.kernel_size[i] - (dx_ins.shape[i+2] + 1)
                   for i in range(self.input_length-1))
        p_x = ((0, 0), (0, 0)) + tuple((p // 2, (p + 1) // 2) for p in p_x_pre)
        pad_grad = np.pad(dx_ins, p_x, mode="constant")
        
        #Layer weight gradient strides and window shape
        grad_strides = (pad_grad.strides[0],
                        0, 
                        *pad_grad.strides[1:], 
                        *pad_grad.strides[2:])
        grad_win_shape = (new_err.shape[0],) + self.__grad_shape
        
        grad_window = np.lib.stride_tricks.as_strided(pad_grad, 
                                                      grad_win_shape, 
                                                      grad_strides)
        view_match = (None, Ellipsis) + ((None,) * (self.input_length-1))
        clip_vals = tuple(slice(x, (-y or None)) for (x, y) in self.pad_details)

        ret_grad = (grad_window * flipped_weights[view_match]) \
                    .sum(axis=tuple(range(2, self.input_length+2)))
        ret_grad = ret_grad[clip_vals]

        #Updating weights
        main_window = np.lib.stride_tricks.as_strided(self.__last_in, 
                                                      main_win_shape, 
                                                      main_strides)
        opt_grad_match = np.expand_dims(opt_grad, axis=tuple(range(2, self.input_length+2)))
        self.kernel_weights += (main_window * opt_grad_match) \
                               .sum(axis=(0,) + tuple(range(-1, -self.input_length, -1)))
                    
        if self.use_bias is True:
            self.bias_weights += opt_grad.sum(axis=0)

        if len(cost_err.shape) < self.input_length+1:
            ret_grad = ret_grad.squeeze(axis=0) 
        return ret_grad
    

    def step(self) -> None:
        """Adds one step to this layer's optimizer and scheduler."""
        self.opt.step()
        self.funct.step()


    def clear_grad(self) -> None:
        """Clears the optimizer gradient history and deletes any data required by the backward pass."""
        self.__last_in = None
        self.__last_out = None
        self.opt.reset_grad()
        self.funct.reset_grad()


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


    def deepcopy(self) -> 'ConvND':
        """Creates a new deepcopy of this layer with the exact same weights and parameters."""
        new_neuron = ConvND(self.funct, self.kernel_weights.shape[0], 
                            self.kernel_size, 
                            self.in_size, 
                            self.strides_all, 
                            self.padding_all, 
                            self.use_bias, self.opt.deepcopy())
        new_neuron.kernel_weights = self.kernel_weights.copy()
        new_neuron.bias_weights = self.bias_weights.copy()
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
        write_ret_str = f"ConvND\u00A0{repr(self.funct)}\u00A0{self.kernel_weights.shape[0]}\u00A0" + \
                        " ".join(self.kernel_size) + "\u00A0" + \
                        " ".join(self.strides_all) + "\u00A0" +  \
                        " ".join(self.padding_all) + "\u00A0" +  \
                        f"\u00A0{int(self.use_bias)}\u00A0{repr(self.opt)}\n" + \
                        "BIAS " + " ".join(list(map(str, self.out_size))) + "\n" + \
                         " ".join(list(map(str, self.bias_weights.flatten().tolist()))) + "\n"
        write_ret_str += "KERNEL " + " ".join(list(map(str, self.kernel_weights.shape))) + "\n" + \
                         " ".join(list(map(str, self.kernel_weights.flatten().tolist()))) + "\n"
        write_ret_str += f"INPUTS " + " ".join(list(map(str, self.in_size))) + "\n\u00A0"

        if not filename:
            return write_ret_str
        if filename.find(".cspn") == -1:
            filename += ".cspn"
        with open(filename, "w+") as file:
            file.write(write_ret_str)
            file.close()


    @staticmethod
    def from_save(context: str, file_load: bool = False) -> 'ConvND':
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
        ConvND
            A new `ConvND` layer containing all of the information encoded in the string or file provided.
        """
        def parse_and_return(handled_str: str):
            data_arr = handled_str.splitlines()
            prop_info = data_arr[0].split("\u00A0")[1:]
            input_info = data_arr[-2].strip().split()[1:]

            bias_size = tuple(map(int, data_arr[1].split()[1:]))
            biases = np.array(list(map(float, data_arr[2].strip().split()))).reshape(bias_size)
            kernel_size = tuple(map(int, data_arr[3].split()[1:]))
            kernels = np.array(list(map(float, data_arr[4].strip().split()))).reshape(kernel_size)

            act = parse_act_info(prop_info[0])                                  # Activation
            opt = parse_opt_info(prop_info[-1])                                 # Optimizer

            new_neuron = ConvND(act,
                                int(prop_info[1]),                              # Layers
                                tuple(map(int, prop_info[2].split())),          # Kernel size
                                tuple(map(int, input_info)),                    # Input size
                                tuple(map(int, prop_info[3].split())),          # Strides
                                tuple(map(int, prop_info[4].split())),          # Padding
                                bool(prop_info[5]),                             # Use-bias
                                opt)
            new_neuron.bias_weights = biases
            new_neuron.kernel_weights = kernels
            return new_neuron

        if file_load:
            full_parse_str = None
            with open(context, "r") as file:
                full_parse_str = file.read()
                file.close()
            return parse_and_return(full_parse_str)
        return parse_and_return(context)
    

    @staticmethod
    @check_types(("strides", all_positive, "Argument \"strides\" must be greater than 0."),
                 ("strides", all_ints, "Argument \"strides\" must contain all integers."),                  

                 ("padding", lambda x: all_positive(x, True), "Argument \"padding\" must be greater than or equal to 0."),
                 ("padding", all_ints, "Argument \"padding\" must contain all integers."),

                 ("input_size", all_positive, "Argument \"input_size\" must contain all positive values above 0."),
                 ("input_size", lambda x: len(x) >= 2, "Argument \"input_size\" must have at least one channel and one convolution dimension."))
    def from_kernel(funct: Activation, input_size: tuple[int, ...], 
                    kernel: np.ndarray, 
                    strides: tuple[int, ...] | int = 1, 
                    padding: tuple[int, ...] | int = 0, 
                    bias: np.ndarray = None, optimizer: Optimizer = StandardGD()) -> 'ConvND':
        """
        Creates a `ConvND` layer from a pre-constructed set of weights and biases.
        
        Notes
        -----
        The kernel shape for this layer should be as follows:

        `(F, C, k*)`, where `F` is the number of output filters/channels,
        `C` is the number of input channels, and `k*` represents the kernel sizes. If the size
        of the input has `N` dimensions, then the kernel should have `N+1` dimensions, with `F` and `C`
        being the first two.

        If the sizes of the strides, padding, or kernel shape do not match the input, an 
        InvalidDataException is raised.


        Parameters
        ----------
        funct : Activation
            An activation function class which supports both forward and backward non-linear
            transformations.
        input_size : tuple[int, ...]
            A tuple of integers matching the shape of the expected input arrays.
        kernel : ndarray
            An array which will be the new layer's set of kernel weights. Based on the sizes of the kernel,
            will set the number of batches and filters.
        strides : tuple[int, ...] | int, default: 1
            An integer or tuple of integers that determines how many data points are skipped for 
            every iteration of the sliding window. Must be greater than or equal to 1.
        padding : tuple[int, ...] | int, default: 0
            An integer or tuple of integers that determines how many empty data points are put on 
            the edges of the final dimensions as padding layers.
        bias : ndarray, default: None
            An array which will be the new layer's set of bias weights. If set to None, the layer will not
            use any biases.
        opt : Optimizer, default: StandardGD()
            An optimizer class which processes given loss gradients and adjusts them to match a desired 
            gradient descent path.

        Returns
        -------
        ConvND
            A new `ConvND` layer containing all of the information given and interpreted from the input kernel.
        
        Raises
        ------
        InvalidDataException
            If the input channel or batch sizes are not equal between the kernel and the inputs, or if the
            kernel is not the correct shape length.
        """
        # Extra dimensionality checks before processing
        _in_len = len(input_size)-1
        if type(strides) != int and len(strides) != _in_len:
            raise InvalidDataException("Argument \"strides\" must have one less dimension than expected input.")
        if type(padding) != int and len(padding) != _in_len:
            raise InvalidDataException("Argument \"padding\" must have one less dimension than expected input.") 

        if input_size[0] != kernel.shape[1]: 
            raise InvalidDataException("Kernel channel dimension must be equal to the input channels.")
        if len(input_size) != len(kernel.shape)-1:
            raise InvalidDataException("Kernel must have one more dimension than the input size.")

        conv_layer = ConvND(funct, 
                            kernel.shape[0], 
                            tuple(kernel.shape[2:]), 
                            input_size, 
                            strides, 
                            padding,
                            True if bias is not None else False, 
                            optimizer)
        
        if bias is not None and bias.shape != conv_layer.out_size:
            raise InvalidDataException("Bias weights must have the same shape as the expected output shape.")
        conv_layer.kernel_weights = kernel
        conv_layer.bias_weights = bias
        return conv_layer
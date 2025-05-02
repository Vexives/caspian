from ..cudalib import np
from . import Conv1D
from ..optimizers import Optimizer, StandardGD, parse_opt_info
from ..activations import Activation, parse_act_info
from ..utilities import dilate_array, all_ints, all_positive, confirm_shape, check_types, \
                        InvalidDataException, UnsafeMemoryAccessException

class Conv1DTranspose(Conv1D):
    """
    A 1D transposed convolutional layer which performs a upward convolution transform on the 
    data provided. Outputs will generally have a larger size on the last dimension than the input.

    Only supports data with 2 or 3 (when batch is included) dimensions as input. The exact shape
    and/or batch size must be specifically stated when initializing the layer.

    
    Memory Safety
    -------------
    This layer is not memory safe if modified. Be extremely careful when modifying any sort of
    variable of this layer, as it may cause memory dangers if done incorrectly.    


    Attributes
    ---------
    in_size : tuple[int, int]
        A tuple containing the expected input size `(C, W)`, where `C` is the number of channels, 
        and `W` is the final dimension of the input.
    out_size : tuple[int, int]
        A tuple containing the expected output size `(F, Ow)`, where `F` represents the new number 
        of channels, and `Ow` represents the final convolved dimension of the output.
    funct : Activation
        The given activation function which takes specific data from each partition of the input.
    strides : int
        The number of data points that the kernel will move over at each step of pooling.
    kernel_size : int
        The size of each partition that will be taken from the original input array.
    padding_all : int
        The total number of data points to be added to the input array as padding.
    pad_left, pad_right : int
        The number of data points to be added to the left and right sides of the data, respectively.
        Corresponds to each half of `padding_all`, with `pad_left` being the first to increment.
    out_padding_all : int 
        The total number of data points to be added to the output array as padding before being
        returned.
    out_pad_left, out_pad_right : int
        The number of data points to be added to the left and right sides of the output, respectively.
        Corresponds to each half of `out_padding_all`, with `out_pad_left` being the first to increment.
    kernel_weights : ndarray
        A set of trainable kernel weights which are applied to each partition extracted from the 
        given input. Has the shape `(F, C, kW)`, with `kW` being the kernel width.
    bias_weights : ndarray | None
        A set of trainable bias weights which are applied to the final result of the convolution.
        Matches the expected output shape, and can be disabled if specified.
    use_bias : bool
        A boolean which determines if the bias weights are initialized, used, and trained.
    opt : Optimizer
        The provided optimizer which modifies the learning gradient before updating weights.


    Examples
    --------
    >>> layer1 = Conv1DTranspose(ReLU(), 2, 3, (5, 9), strides=1)
    >>> in_arr = np.random.uniform(0.0, 1.0, (5, 9))
    >>> out_arr = layer1(in_arr)
    >>> print(out_arr.shape)
    (1, 3, 11)
    """
    @check_types([
                  ("layers", lambda x: x > 0, "Argument \"layers\" must be greater than zero."),
                  ("padding", lambda x: x >= 0, "Argument \"padding\" must be greater than or equal to zero."),
                  ("out_padding", lambda x: x >= 0, "Argument \"out_padding\" must be greater than or equal to zero."),
                  ("strides", lambda x: x > 0, "Argument \"strides\" must be greater than or equal to one."),
                  ("kernel_size", lambda x: x > 0, "Argument \"kernel_size\" must be greater than or equal to one."),
                  ("input_size", all_positive, "Argument \"input_size\" must contain all positive values above 0."),
                  ("input_size", lambda x: len(x) == 2, "Argument \"input_size\" must have a length of 2.")
                ])
    def __init__(self, funct: Activation, layers: int, kernel_size: int, 
                 input_size: tuple[int, int], strides: int = 1, 
                 padding: int = 0, out_padding: int = 0,
                 biases: bool = True, optimizer: Optimizer = StandardGD()):
        """
        Initializes a `Conv1DTranspose` layer using given parameters.

        Parameters
        ----------
        funct : Activation
            An activation function class which supports both forward and backward non-linear
            transformations.
        layers : int
            The number of resulting channels that the output array will have. Will represent the new
            `C` dimension size for the output.
        kernel_size : int
            An integer representing the size of the sliding window to extract partitions of the input data.
        input_size : tuple[int, int] | tuple[int, int, int]
            A tuple of integers matching the shape of the expected input arrays. If a third dimension is added,
            the first dimension is used as the batch size.
        strides : int, default: 1
            An integer that determines how many data points are skipped for every iteration of the 
            sliding window. Must be greater than or equal to 1.
        padding : int, default: 0
            An integer that determines how many empty data points are removed before going forward and added on the
            backwards pass. Represents the padding that would be added in a standard `Conv1D` layer.
        out_padding : int, default: 0
            An integer that represents the number of data points to be added to the output array as padding
            before returning.
        batch_size : int, default: 1
            An integer representing the batch size of the expected input data. Must be greater than or equal
            to 1.
        biases : bool, default: True
            Determines whether the bias values for this layer are initialized, used, and trained.
        opt : Optimizer, default: StandardGD()
            An optimizer class which processes given loss gradients and adjusts them to match a desired 
            gradient descent path.

        Raises
        ------
        InvalidDataException
            If any of the data provided is not an integer or less than one (with the exception of padding, 
            which can be 0), or if the given function is not of type `Activation`. 
            Also applies to the expected input shape, which must be a tuple of integers.
        """
        #Padding Initialization
        if not all_ints(padding):
            raise InvalidDataException("Padding must be all integers.")
        if not all_positive(padding, True):
            raise InvalidDataException("Padding must be greater than or equal to zero.")
        self.padding_all = padding
        self.pad_left, self.pad_right = ((padding+1)//2, padding//2)

        if not all_ints(out_padding):
            raise InvalidDataException("Output padding must be all integers.")
        if not all_positive(out_padding, True):
            raise InvalidDataException("Output padding must be greater than or equal to zero.")
        self.out_padding_all = out_padding
        self.out_pad_left, self.out_pad_right = ((out_padding+1)//2, out_padding//2)

        #Other params
        if not isinstance(funct, Activation):
            raise InvalidDataException("Function must be a descendant of Caspian \'Activation\' class.")
        self.funct = funct
        self.opt = optimizer

        if not all_ints(strides):
            raise InvalidDataException("Strides must be all integers.")
        if not all_ints(kernel_size):
            raise InvalidDataException("Kernel size must be all integers.")
        if not all_positive(strides): 
            raise InvalidDataException("Strides must be greater than or equal to one.")
        if not all_positive(kernel_size): 
            raise InvalidDataException("Kernel size must be greater than or equal to one.")
        self.strides = strides        
        self.kernel_size = kernel_size
        self.use_bias = biases

        #In/Out Sizes
        if not all_ints(input_size):
            raise InvalidDataException("Input size must be all integers.")
        in_size = input_size
        out_size = (layers, 
                         max(((self.in_size[1] - 1) * strides) + out_padding + kernel_size - padding, 0))
        super().__init__(in_size, out_size)

        #Window Shapes
        self.__window_shape = (layers, 
                             self.in_size[0], 
                             kernel_size, 
                             self.out_size[1] + padding - out_padding) #(F, C, K, out_S + pad)
        self.__grad_shape = (self.in_size[0], 
                           layers, 
                           kernel_size, 
                           self.in_size[1]) #(C, F, K, in_S)
        self.__dx_shape = (self.in_size[0],
                         self.in_size[1] + (self.in_size[1]-1) * (strides-1)) #(C, in_S + dilate)

        self.kernel_weights = np.random.uniform(-0.5, 0.5, (layers, self.in_size[0], kernel_size))
        self.bias_weights = np.zeros((self.out_size[0], 
                                      self.out_size[1] - out_padding)) if biases is True else None


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
        if not confirm_shape(data.shape, self.in_size, 2):
            raise UnsafeMemoryAccessException(f"Input data shape does not match expected shape. - {data.shape}, {self.in_size}")
        new_data = np.expand_dims(data, axis=0) if len(data.shape) < 3 else data    #Enforce batches.

        #Initial dilation of array
        dil_data = dilate_array(new_data, (new_data.shape[0],) + self.__dx_shape, (self.strides,))

        #Padding, shape, and strides calculation
        p_x = self.out_size[1] - self.out_padding_all + self.padding_all + self.kernel_size - (dil_data.shape[2] + 1)
        pad_data = np.pad(dil_data, ((0, 0), (0, 0),
                                     (p_x // 2, (p_x + 1) // 2)), mode="constant")
        new_strides = (pad_data.strides[0],
                       0, 
                       pad_data.strides[1], 
                       pad_data.strides[2], 
                       pad_data.strides[2])
        data_win_shape = (new_data.shape[0],) + self.__window_shape

        #Stride windows and summation
        sliding_view = np.lib.stride_tricks.as_strided(pad_data, 
                                                       shape=data_win_shape, 
                                                       strides=new_strides)
        conv_val = np.einsum("nfckx,fck->nfx", sliding_view, self.kernel_weights)
        conv_val = conv_val[:, :, self.pad_left:(-self.pad_right or None)]

        #Activation & output padding
        last = self.funct(conv_val + self.bias_weights if self.use_bias is True else conv_val)
        last = np.pad(last, ((0, 0), (0, 0),
                             (self.out_pad_left, self.out_pad_right)), mode="constant")

        if training:
            self.__last_in = dil_data
            self.__last_out = last

        if len(data.shape) < 3:
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
        if not confirm_shape(cost_err.shape, self.out_size, 2):
            raise UnsafeMemoryAccessException(f"Input data shape does not match expected shape. - {cost_err.shape}, {self.in_size}")
        new_err = np.expand_dims(cost_err, axis=0) if len(cost_err.shape) < 3 else cost_err   #Enforce batches.

        #Optimized & standard gradient preparation
        new_err = new_err * self.funct(self.__last_out, True)         # Gradient for backward pass
        opt_grad = self.opt.process_grad(new_err)                   # Gradient for updating weights

        new_err = new_err[:, :, self.out_pad_left:(-self.out_pad_right or None)]
        opt_grad = opt_grad[:, :, self.out_pad_left:(-self.out_pad_right or None)]
        opt_grad_pad = np.pad(opt_grad, ((0, 0), (0, 0), 
                                         (self.pad_left, self.pad_right)), mode="constant")

        #Input gradient preparation
        flipped_weights = np.flip(self.kernel_weights, axis=2)
        err_pad = np.pad(new_err, ((0, 0), (0, 0),
                                   (self.pad_left, self.pad_right)), mode="constant")
        err_strides = (err_pad.strides[0],
                       0, 
                       err_pad.strides[1], 
                       err_pad.strides[2], 
                       self.strides * err_pad.strides[2])
        grad_win_shape = (new_err.shape[0],) + self.__grad_shape

        err_view = np.lib.stride_tricks.as_strided(err_pad, 
                                                   shape=grad_win_shape, 
                                                   strides=err_strides)
        ret_grad = np.einsum("ncfkx,fck->ncx", err_view, flipped_weights)

        #Weights gradient
        p_x = self.out_size[1] + self.padding_all + self.kernel_size - (self.__last_in.shape[2] + 1)
        opt_pad = np.pad(self.__last_in, ((0, 0), (0, 0),
                                        (p_x // 2, (p_x + 1) // 2)), mode="constant")
        opt_strides = (opt_pad.strides[0],
                       0, 
                       opt_pad.strides[1], 
                       opt_pad.strides[2], 
                       opt_pad.strides[2])
        opt_win_shape = (new_err.shape[0],) + self.__window_shape

        opt_view = np.lib.stride_tricks.as_strided(opt_pad, 
                                                   shape=opt_win_shape, 
                                                   strides=opt_strides)
        self.kernel_weights += np.einsum("nfckx,nfx->fck", opt_view, opt_grad_pad)

        if self.use_bias is True:
            self.bias_weights += opt_grad.sum(axis=0)

        if len(cost_err.shape) < 3:
            ret_grad = ret_grad.squeeze(axis=0) 
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
    

    def deepcopy(self) -> 'Conv1DTranspose':
        """Creates a new deepcopy of this layer with the exact same weights and parameters."""
        new_neuron = Conv1DTranspose(self.funct, self.kernel_weights.shape[0], self.kernel_size, self.in_size, 
                                     self.strides, self.padding_all, self.out_padding_all, self.use_bias, self.opt.deepcopy())
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
        write_ret_str = f"Conv1DTranspose\u00A0{repr(self.funct)}\u00A0{self.kernel_weights.shape[0]}" + \
                        f"\u00A0{self.kernel_size}\u00A0{self.strides}\u00A0{self.padding_all}\u00A0{self.out_padding_all}" + \
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
    def from_save(context: str, file_load: bool = False) -> 'Conv1DTranspose':
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
        Conv1DTranspose
            A new `Conv1DTranspose` layer containing all of the information encoded in the string or file provided.
        """
        def parse_and_return(handled_str: str):
            data_arr = handled_str.splitlines()
            prop_info = data_arr[0].split("\u00A0")
            input_info = data_arr[-2].strip().split()[1:]

            bias_size = tuple(map(int, data_arr[1].split()[1:]))
            biases = np.array(list(map(float, data_arr[2].strip().split()))).reshape(bias_size)
            kernel_size = tuple(map(int, data_arr[3].split()[1:]))
            kernels = np.array(list(map(float, data_arr[4].strip().split()))).reshape(kernel_size)

            act = parse_act_info(prop_info[1])
            opt = parse_opt_info(prop_info[-1])

            new_neuron = Conv1DTranspose(act, 
                                         int(prop_info[2]), 
                                         int(prop_info[3]), 
                                         tuple(map(int, input_info)), 
                                         int(prop_info[4]), 
                                         int(prop_info[5]), 
                                         int(prop_info[6]), 
                                         bool(prop_info[7]),
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
    @check_types([
                  ("padding", lambda x: x >= 0, "Argument \"padding\" must be greater than or equal to zero."),
                  ("out_padding", lambda x: x >= 0, "Argument \"out_padding\" must be greater than or equal to zero."),
                  ("strides", lambda x: x > 0, "Argument \"strides\" must be greater than or equal to one."),
                  ("input_size", all_positive, "Argument \"input_size\" must contain all positive values above 0."),
                  ("input_size", lambda x: len(x) == 2, "Argument \"input_size\" must have a length of 2."),
                  ("kernel", lambda x: len(x.shape) == 3, "Argument \"kernel\" must have dimension shape of 3.")
                ])
    def from_kernel(funct: Activation, input_size: tuple[int, int], 
                    kernel: np.ndarray, strides: int = 1, padding: int = 0, out_padding: int = 0, 
                    bias: np.ndarray = None, optimizer: Optimizer = StandardGD()) -> 'Conv1D':
        """
        Creates a `Conv1DTranspose` layer from a pre-constructed set of weights and biases.
        
        Notes
        -----
        The kernel shape for this layer should be as follows:

        `(F, C, kW)`, where `F` is the number of output filters/channels,
        `C` is the number of input channels, and `kW` is the kernel size.

        If the sizes do not match the input, an InvalidDataException is raised.


        Parameters
        ----------
        funct : Activation
            An activation function class which supports both forward and backward non-linear
            transformations.
        input_size : tuple[int, int]
            A tuple of integers matching the shape of the expected input arrays.
        kernel : ndarray
            An array which will be the new layer's set of kernel weights. Based on the sizes of the kernel,
            will set the number of filters.
        strides : int, default: 1
            An integer that determines how many data points are skipped for every iteration of the 
            sliding window. Must be greater than or equal to 1.
        padding : int, default: 0
            An integer that determines how many empty data points are removed from the edges of the final 
            input dimension as padding layers.
        out_padding : int, default: 0
            An integer that represents the number of data points to be added to the output array as padding
            before returning.
        bias : ndarray, default: None
            An array which will be the new layer's set of bias weights. If set to None, the layer will not
            use any biases.
        opt : Optimizer, default: StandardGD()
            An optimizer class which processes given loss gradients and adjusts them to match a desired 
            gradient descent path.

        Returns
        -------
        Conv1DTranspose
            A new `Conv1DTranspose` layer containing all of the information given and interpreted from the input kernel.

        Raises
        ------
        InvalidDataException
            If the input channel or batch sizes are not equal between the kernel and the inputs, or if the
            kernel is not the correct shape length.
        """
        if len(kernel.shape) != 3:
            raise InvalidDataException(f"Kernel shape must be of length 3 instead of length {len(kernel.shape)}.")
        if input_size[0] != kernel.shape[1]: 
            raise InvalidDataException("Kernel channel dimension must be equal to the input channels.")

        conv_layer = Conv1DTranspose(funct, kernel.shape[0], kernel.shape[-1], 
                                     input_size, strides, padding, out_padding, 
                                     True if bias is not None else False, optimizer)
        
        if bias is not None and not isinstance(bias, np.ndarray):
            raise InvalidDataException("Bias weights must be of type ndarray.")
        if bias is not None and bias.shape != conv_layer.out_size:
            raise InvalidDataException("Bias weights must have the same shape as the expected output shape.")
        conv_layer.kernel_weights = kernel
        conv_layer.bias_weights = bias
        return conv_layer
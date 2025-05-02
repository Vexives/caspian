from ..cudalib import np
from . import Layer
from ..optimizers import Optimizer, StandardGD, parse_opt_info
from ..activations import Activation, parse_act_info
from ..utilities import dilate_array, all_ints, all_positive, confirm_shape, check_types, \
                        InvalidDataException, UnsafeMemoryAccessException

class Conv3D(Layer):
    """
    A 3D convolutional layer which performs a downwards convolution transform on the data provided.

    Only supports data with 4 or 5 (when batch is included) dimensions as input. The exact shape
    and/or batch size must be specifically stated when initializing the layer.

    
    Memory Safety
    -------------
    This layer is not memory safe if modified. Be extremely careful when modifying any sort of
    variable of this layer, as it may cause memory dangers if done incorrectly.


    Attributes
    ---------
    in_size : tuple[int, int, int, int]
        A tuple containing the expected input size `(C, D, H, W)`, where `C` is the number of channels, 
        and `D`, `H`, `W` are the final dimensions of the input.
    out_size : tuple[int, int, int, int]
        A tuple containing the expected output size `(F, Od, Oh, Ow)`, where `F` represents the new 
        number of channels, and `Od`, `Oh`, `Ow` represents the final convolved dimensions of the 
        output.
    funct : Activation
        The given activation function which takes specific data from each partition of the input.
    stride_d, stride_h, stride_w : int
        The number of data points that the kernel will move over at each step of the convolution. 
        Represents depth, height, and width strides respectively.
    kernel_depth, kernel_height, kernel_width : int
        The size of each partition that will be taken from the original input array. Represents the 
        depth, height, and width of the partition, respectively.
    padding_all : tuple[int, int, int] | int
        The padding input value that was provided upon initialization. Used for utilities like deep
        copies.
    pad_depth, pad_height, pad_width : int
        The total number of data points to be added to the input array as padding on the depth, height, 
        and width dimensions, respectively.
    pad_left, pad_right : int
        The number of data points to be added to the left and right sides of the data, respectively.
        Corresponds to each half of `pad_width`, with `pad_left` being the first to increment.
    pad_top, pad_bottom : int
        The number of data points to be added to the top and bottom sides of the data, respectively.
        Corresponds to each half of `pad_height`, with `pad_top` being the first to increment.
    pad_front, pad_back : int
        The number of data points to be added to the front and back sides of the data, respectively.
        Corresponds to each half of `pad_depth`, with `pad_front` being the first to increment.
    kernel_weights : ndarray
        A set of trainable kernel weights which are applied to each partition extracted from the 
        given input. Has the shape `(F, C, kD, kH, kW)`, with `kD`, `kH`, `kW` being the kernel
        depth, height, and width, respectively.
    bias_weights : ndarray | None
        A set of trainable bias weights which are applied to the final result of the convolution.
        Matches the expected output shape, and can be disabled if specified.
    use_bias : bool
        A boolean which determines if the bias weights are initialized, used, and trained.
    opt : Optimizer
        The provided optimizer which modifies the learning gradient before updating weights.


    Examples
    --------
    >>> layer1 = Conv3D(ReLU(), 2, 3, (5, 9, 10, 8), strides=1)
    >>> in_arr = np.random.uniform(0.0, 1.0, (5, 9, 10, 8))
    >>> out_arr = layer1(in_arr)
    >>> print(out_arr.shape)
    (1, 2, 7, 8, 6)
    """
    @check_types(("layers", lambda x: x > 0, "Argument \"layers\" must be greater than zero."),

                 ("kernel_size", all_positive, "Argument \"kernel_size\" must be greater than 0."),
                 ("kernel_size", all_ints, "Argument \"kernel_size\" must contain all integers."),
                 ("kernel_size", lambda x: isinstance(x, int) or len(x) == 3, "Argument \"kernel_size\" must have a length of 3."),

                 ("strides", all_positive, "Argument \"strides\" must be greater than 0."),
                 ("strides", all_ints, "Argument \"strides\" must contain all integers."),                  
                 ("strides", lambda x: isinstance(x, int) or len(x) == 3, "Argument \"strides\" must have a length of 3."),

                 ("padding", lambda x: all_positive(x, True), "Argument \"padding\" must be greater than or equal to 0."),
                 ("padding", all_ints, "Argument \"padding\" must contain all integers."),
                 ("padding", lambda x: isinstance(x, int) or len(x) == 3, "Argument \"padding\" must have a length of 3."),

                 ("input_size", all_positive, "Argument \"input_size\" must contain all positive values above 0."),
                 ("input_size", lambda x: len(x) == 4, "Argument \"input_size\" must have a length of 4."))
    def __init__(self, funct: Activation, layers: int, kernel_size: tuple[int, int, int] | int, 
                 input_size: tuple[int, int, int, int], 
                 strides: tuple[int, int, int] | int = 1, padding: tuple[int, int, int] | int = 0, 
                 biases: bool = True, optimizer: Optimizer = StandardGD()) -> None:
        """
        Initializes a `Conv3D` layer using given parameters.

        Parameters
        ----------
        funct : Activation
            An activation function class which supports both forward and backward non-linear
            transformations.
        layers : int
            The number of resulting channels that the output array will have. Will represent the new
            `C` dimension size for the output.
        kernel_size : tuple[int, int, int] | int
            An integer or tuple of integers representing the size of the sliding window to extract 
            partitions of the input data.
        input_size : tuple[int, int, int, int]
            A tuple of integers matching the shape of the expected input arrays.
        strides : tuple[int, int, int] | int, default: 1
            An integer or tuple of integers that determines how many data points are skipped for 
            every iteration of the sliding window. Must be greater than or equal to 1.
        padding : tuple[int, int, int] | int, default: 0
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
        """ 
        #Padding Initialization
        self.pad_depth, self.pad_height, self.pad_width = padding if isinstance(padding, tuple) else (padding, padding, padding)
        self.pad_front, self.pad_back = ((self.pad_depth+1)//2, self.pad_depth//2)
        self.pad_top, self.pad_bottom = ((self.pad_height+1)//2, self.pad_height//2)
        self.pad_left, self.pad_right = ((self.pad_width+1)//2, self.pad_width//2)

        #Other settings
        self.funct = funct
        self.opt = optimizer
        self.stride_d, self.stride_h, self.stride_w = strides if isinstance(strides, tuple) else (strides, strides, strides)
        self.kernel_depth, self.kernel_height, self.kernel_width = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size, kernel_size)
        self.use_bias = biases

        #In/Out Sizes
        in_size = input_size
        out_size = (layers,
                    (in_size[1] - self.kernel_depth + self.pad_depth) // self.stride_d + 1,
                    (in_size[2] - self.kernel_height + self.pad_height) // self.stride_h + 1, 
                    (in_size[3] - self.kernel_width + self.pad_width) // self.stride_w + 1)
        super().__init__(in_size, out_size)

        #Strides and Window Shapes
        self.__window_shape = (layers, 
                             self.in_size[0], 
                             self.kernel_depth,
                             self.kernel_height, 
                             self.kernel_width, 
                             self.out_size[1], 
                             self.out_size[2],
                             self.out_size[3]) #(F, C, D, H, W, out_D, out_H, out_W)
        self.__grad_shape = (self.in_size[0], 
                           layers, 
                           self.kernel_depth,
                           self.kernel_height, 
                           self.kernel_width, 
                           self.in_size[1] + self.pad_depth, 
                           self.in_size[2] + self.pad_height,
                           self.in_size[3] + self.pad_width) #(C, F, D, H, in_D + pad, in_H + pad, in_W + pad)
        self.__dx_shape = (self.out_size[0],
                         self.out_size[1] + (self.out_size[1]-1) * (self.stride_d-1),
                         self.out_size[2] + (self.out_size[2]-1) * (self.stride_h-1),
                         self.out_size[3] + (self.out_size[3]-1) * (self.stride_w-1)) #(F, out_D + dilate, out_H + dilate, out_W + dilate)

        self.kernel_weights = np.random.uniform(-0.5, 0.5, (layers, 
                                                            self.in_size[0], 
                                                            self.kernel_depth, 
                                                            self.kernel_height, 
                                                            self.kernel_width))
        self.bias_weights = np.zeros(self.out_size) if biases is True else None
    

    def forward(self, data: np.ndarray, training: bool = False) -> np.ndarray:
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

        Raises
        ------
        UnsafeMemoryAccessException
            If the shape of the given array will lead to any un-safe memory calls during the pass.
        """
        if not confirm_shape(data.shape, self.in_size, 4):
            raise UnsafeMemoryAccessException(f"Input data shape does not match expected shape. - {data.shape}, {self.in_size}")
        new_data = np.expand_dims(data, axis=0) if len(data.shape) < 5 else data    #Enforce batches.
        new_data = np.pad(new_data, ((0, 0), (0, 0),
                                     (self.pad_front, self.pad_back),
                                     (self.pad_top, self.pad_bottom), 
                                     (self.pad_left, self.pad_right)), mode="constant")
        
        #Sliding view strides and window shape
        data_strides = (new_data.strides[0],
                        0, 
                        new_data.strides[1], 
                        new_data.strides[2], 
                        new_data.strides[3],
                        new_data.strides[4], 
                        self.stride_d * new_data.strides[2], 
                        self.stride_h * new_data.strides[3],
                        self.stride_w * new_data.strides[4])
        data_win_shape = (new_data.shape[0],) + self.__window_shape

        #Sliding windows and convolution multiplication
        sliding_view = np.lib.stride_tricks.as_strided(new_data, 
                                                       shape=data_win_shape, 
                                                       strides=data_strides)
        conv_val = np.einsum("nfcdhwxyz,fcdhw->nfxyz", sliding_view, self.kernel_weights)

        last = self.funct(conv_val + self.bias_weights if self.use_bias else conv_val)
        if training:
            self.__last_in = new_data
            self.__last_out = last

        if len(data.shape) < 5:
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
        if not confirm_shape(cost_err.shape, self.out_size, 4):
            raise UnsafeMemoryAccessException(f"Input data shape does not match expected shape. - {cost_err.shape}, {self.in_size}")
        new_err = np.expand_dims(cost_err, axis=0) if len(cost_err.shape) < 5 else cost_err   #Enforce batches.
        new_err = new_err * self.funct(self.__last_out, True)
        opt_grad = self.opt.process_grad(new_err)

        main_strides = (self.__last_in.strides[0],
                        0, 
                        self.__last_in.strides[1], 
                        self.__last_in.strides[2], 
                        self.__last_in.strides[3],
                        self.__last_in.strides[4], 
                        self.stride_d * self.__last_in.strides[2], 
                        self.stride_h * self.__last_in.strides[3],
                        self.stride_w * self.__last_in.strides[4])
        main_win_shape = (new_err.shape[0],) + self.__window_shape

        #Input gradient
        flipped_weights = np.flip(self.kernel_weights, axis=(2, 3, 4))
        dx_ins = dilate_array(new_err, (new_err.shape[0],) + self.__dx_shape, 
                              (self.stride_d, self.stride_h, self.stride_w))

        p_d = self.__last_in.shape[2] + self.kernel_depth - (dx_ins.shape[2] + 1)
        p_h = self.__last_in.shape[3] + self.kernel_height - (dx_ins.shape[3] + 1)
        p_w = self.__last_in.shape[4] + self.kernel_width - (dx_ins.shape[4] + 1)
        pad_grad = np.pad(dx_ins, ((0, 0), (0, 0),
                                   (p_d // 2, (p_d + 1) // 2), 
                                   (p_h // 2, (p_h + 1) // 2), 
                                   (p_w // 2, (p_w + 1) // 2)), mode="constant")
        
        #Layer weight gradient strides and window shape
        grad_strides = (pad_grad.strides[0],
                        0, 
                        pad_grad.strides[1], 
                        pad_grad.strides[2], 
                        pad_grad.strides[3],
                        pad_grad.strides[4], 
                        pad_grad.strides[2], 
                        pad_grad.strides[3],
                        pad_grad.strides[4])
        grad_win_shape = (new_err.shape[0],) + self.__grad_shape
        
        grad_window = np.lib.stride_tricks.as_strided(pad_grad, 
                                                      grad_win_shape, 
                                                      grad_strides)
        ret_grad = np.einsum("ncfdhwxyz,fcdhw->ncxyz", grad_window, flipped_weights)
        ret_grad = ret_grad[:, :, self.pad_front:(-self.pad_back or None), 
                                  self.pad_top:(-self.pad_bottom or None), 
                                  self.pad_left:(-self.pad_right or None)]

        #Updating weights
        main_window = np.lib.stride_tricks.as_strided(self.__last_in, 
                                                      main_win_shape, 
                                                      main_strides)
        self.kernel_weights += np.einsum("nfcdhwxyz,nfxyz->fcdhw", main_window, opt_grad)
                    
        if self.use_bias:
            self.bias_weights += opt_grad.sum(axis=0)

        if len(cost_err.shape) < 5:
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
    

    def deepcopy(self) -> 'Conv3D':
        """Creates a new deepcopy of this layer with the exact same weights and parameters."""
        new_neuron = Conv3D(self.funct, self.kernel_weights.shape[0], 
                            (self.kernel_depth, self.kernel_height, self.kernel_width), 
                            self.in_size, 
                            (self.stride_d, self.stride_h, self.stride_w), 
                            (self.pad_depth, self.pad_height, self.pad_width), 
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
        write_ret_str = f"Conv2D\u00A0{repr(self.funct)}\u00A0{self.kernel_weights.shape[0]}" + \
                        f"\u00A0{self.kernel_depth}\u00A0{self.kernel_height}\u00A0{self.kernel_width}" + \
                        f"\u00A0{self.stride_d}\u00A0{self.stride_h}\u00A0{self.stride_w}" + \
                        f"\u00A0{self.pad_depth}\u0A00{self.pad_height}\u00A0{self.pad_width}" + \
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
    def from_save(context: str, file_load: bool = False) -> 'Conv3D':
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
        Conv3D
            A new `Conv3D` layer containing all of the information encoded in the string or file provided.
        """
        def parse_and_return(handled_str: str):
            data_arr = handled_str.splitlines()
            prop_info = data_arr[0].split("\u00A0")
            input_info = data_arr[-2].strip().split()[1:]

            bias_size = tuple(map(int, data_arr[1].split()[1:]))
            biases = np.array(list(map(float, data_arr[2].strip().split()))).reshape(bias_size)
            kernel_size = tuple(map(int, data_arr[3].split()[1:]))
            kernels = np.array(list(map(float, data_arr[4].strip().split()))).reshape(kernel_size)

            act = parse_act_info(prop_info[1])                                  #Activation
            opt = parse_opt_info(prop_info[-1])                                 #Optimizer

            new_neuron = Conv3D(act,
                                int(prop_info[2]),                                                 #Layers
                                tuple(int(prop_info[3]), int(prop_info[4]), int(prop_info[5])),    #Kernel size
                                tuple(map(int, input_info)),                                       #Input size
                                tuple(int(prop_info[6]), int(prop_info[7]), int(prop_info[8])),    #Strides
                                tuple(int(prop_info[9]), int(prop_info[10]), int(prop_info[11])),  #Padding
                                bool(prop_info[12]),                                               #Use-bias
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
                  ("strides", all_positive, "Argument \"strides\" must be greater than 0."),
                  ("strides", all_ints, "Argument \"strides\" must contain all integers."),                  
                  ("strides", lambda x: isinstance(x, int) or len(x) == 3, "Argument \"strides\" must have a length of 3."),

                  ("padding", lambda x: all_positive(x, True), "Argument \"padding\" must be greater than or equal to 0."),
                  ("padding", all_ints, "Argument \"padding\" must contain all integers."),
                  ("padding", lambda x: isinstance(x, int) or len(x) == 3, "Argument \"padding\" must have a length of 3."),

                  ("input_size", all_positive, "Argument \"input_size\" must contain all positive values above 0."),
                  ("input_size", lambda x: len(x) == 4, "Argument \"input_size\" must have a length of 4."),

                  ("kernel", lambda x: len(x.shape) == 5, "Argument \"kernel\" must have dimension shape of 5.")
                ])
    def from_kernel(funct: Activation, input_size: tuple[int, int, int, int], 
                    kernel: np.ndarray, strides: tuple[int, int, int] | int = 1, padding: tuple[int, int, int] | int = 0, 
                    bias: np.ndarray = None, optimizer: Optimizer = StandardGD()) -> 'Conv3D':
        """
        Creates a `Conv3D` layer from a pre-constructed set of weights and biases.
        
        Notes
        -----
        The kernel shape for this layer should be as follows:

        `(F, C, kD, kH, kW)`, where `F` is the number of output filters/channels,
        `C` is the number of input channels, and `kD`, `kH`, `kW` are the kernel sizes.

        If the sizes do not match the input, an InvalidDataException is raised.


        Parameters
        ----------
        funct : Activation
            An activation function class which supports both forward and backward non-linear
            transformations.
        input_size : tuple[int, int, int, int]
            A tuple of integers matching the shape of the expected input arrays.
        kernel : ndarray
            An array which will be the new layer's set of kernel weights. Based on the sizes of the kernel,
            will set the number of batches and filters.
        strides : tuple[int, int, int] | int, default: 1
            An integer or tuple of integers that determines how many data points are skipped for every 
            iteration of the sliding window. Must be greater than or equal to 1.
        padding : tuple[int, int, int] | int, default: 0
            An integer or tuple of integers that determines how many empty data points are put on the 
            edges of the final dimension as padding layers.
        bias : ndarray, default: None
            An array which will be the new layer's set of bias weights. If set to None, the layer will not
            use any biases.
        opt : Optimizer, default: StandardGD()
            An optimizer class which processes given loss gradients and adjusts them to match a desired 
            gradient descent path.

        Returns
        -------
        Conv3D
            A new `Conv3D` layer containing all of the information given and interpreted from the input kernel.
        
        Raises
        ------
        InvalidDataException
            If the input channel or batch sizes are not equal between the kernel and the inputs, or if the
            kernel is not the correct shape length.
        """
        if input_size[0] != kernel.shape[1]: 
            raise InvalidDataException("Kernel channel dimension must be equal to the input channels.")

        conv_layer = Conv3D(funct, kernel.shape[0], tuple(kernel.shape[-3:]), 
                                  input_size, strides, padding,
                                  True if bias is not None else False, optimizer)
        
        if bias is not None and bias.shape != conv_layer.out_size:
            raise InvalidDataException("Bias weights must have the same shape as the expected output shape.")
        conv_layer.kernel_weights = kernel
        conv_layer.bias_weights = bias
        return conv_layer
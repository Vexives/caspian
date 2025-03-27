from cudalib import np
from . import Layer
from optimizers import Optimizer, StandardGD
from activations import Activation, parse_act_info
from utilities import dilate_array

class Conv2D(Layer):
    """
    A 2D convolutional layer which performs a downwards convolution transform on the data provided.

    Only supports data with 3 or 4 (when batch is included) dimensions as input. The exact shape
    and/or batch size must be specifically stated when initializing the layer.


    Attributes
    ---------
    in_size : tuple[int, int, int, int]
        A tuple containing the expected input size `(N, C, H, W)`, where `N` is the number of batches, 
        `C` is the number of channels, and `H`, `W` are the final dimensions of the input.
    out_size : tuple[int, int, int, int]
        A tuple containing the expected output size `(N, F, Oh, Ow)`, where `N` is the same 
        as the input, with `F` representing the new number of channels, and `Oh`, `Ow` representing the 
        final convolved dimensions of the output.
    funct : Activation
        The given activation function which takes specific data from each partition of the input.
    stride_h, stride_w : int
        The number of data points that the kernel will move over at each step of the convolution. 
        Represents height and width strides respectively.
    kernel_height, kernel_width : int
        The size of each partition that will be taken from the original input array. Represents the 
        height and width of the partition, respectively.
    padding_all : tuple[int, int] | int
        The padding input value that was provided upon initialization. Used for utilities like deep
        copies.
    pad_height, pad_width : int
        The total number of data points to be added to the input array as padding on the height and
        width dimensions, respectively.
    pad_left, pad_right : int
        The number of data points to be added to the left and right sides of the data, respectively.
        Corresponds to each half of `pad_width`, with `pad_left` being the first to increment.
    pad_top, pad_bottom : int
        The number of data points to be added to the top and bottom sides of the data, respectively.
        Corresponds to each half of `pad_height`, with `pad_top` being the first to increment.
    window_shape : tuple[int, int, int, int, int, int, int]
        The shape that the data will take whenever a strided view is created for the forward pass.
    grad_shape : tuple[int, int, int, int, int, int, int]
        The shape that the data will take whenever a strided view is created for the backward pass. 
        Some dimensions from the `window_shape` variable are transposed.
    kernel_weights : ndarray
        A set of trainable kernel weights which are applied to each partition extracted from the 
        given input. Has the shape `(F, C, kH, kW)`, with `kH`, `kW` being the kernel
        height and width respectively.
    bias_weights : ndarray | None
        A set of trainable bias weights which are applied to the final result of the convolution.
        Matches the expected output shape, and can be disabled if specified.
    use_bias : bool
        A boolean which determines if the bias weights are initialized, used, and trained.
    opt : Optimizer
        The provided optimizer which modifies the learning gradient before updating weights.
    last_in : ndarray | None
        A saved value for the last input, only stored if `training` is set to `True`
        during a forward pass.
    last_out : ndarray | None
        A saved value for the last output, only stored if `training` is set to `True`
        during a forward pass.


    Examples
    --------
    >>> layer1 = Conv2D(ReLU(), 2, 3, (5, 9, 10), strides=1)
    >>> in_arr = np.random.uniform(0.0, 1.0, (5, 9, 10))
    >>> out_arr = layer1(in_arr)
    >>> print(out_arr.shape)
    (1, 2, 7, 8)
    """
    def __init__(self, funct: Activation, layers: int, kernel_size: tuple[int, int] | int, 
                 input_size: tuple[int, int, int], 
                 strides: tuple[int, int] | int = 1, padding: tuple[int, int] | int = 0, 
                 biases: bool = True, optimizer: Optimizer = StandardGD()) -> None:
        """
        Initializes a `Conv2D` layer using given parameters.

        Parameters
        ----------
        funct : Activation
            An activation function class which supports both forward and backward non-linear
            transformations.
        layers : int
            The number of resulting channels that the output array will have. Will represent the new
            `C` dimension size for the output.
        kernel_size : tuple[int, int] | int
            An integer or tuple of integers representing the size of the sliding window to extract 
            partitions of the input data.
        input_size : tuple[int, int, int] | tuple[int, int, int, int]
            A tuple of integers matching the shape of the expected input arrays. If a fourth dimension is added,
            the first dimension is used as the batch size.
        strides : tuple[int, int] | int, default: 1
            An integer or tuple of integers that determines how many data points are skipped for 
            every iteration of the sliding window. Must be greater than or equal to 1.
        padding : tuple[int, int] | int, default: 0
            An integer or tuple of integers that determines how many empty data points are put on 
            the edges of the final dimensions as padding layers.
        biases : bool, default: True
            Determines whether the bias values for this layer are initialized, used, and trained.
        opt : Optimizer, default: StandardGD()
            An optimizer class which processes given loss gradients and adjusts them to match a desired 
            gradient descent path.
        """        
        #Stride, Kernel, and Padding Initialization
        self.stride_h, self.stride_w = strides if isinstance(strides, tuple) else (strides, strides)

        self.kernel_height, self.kernel_width = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)

        self.pad_height, self.pad_width = padding if isinstance(padding, tuple) else (padding, padding)
        self.pad_top, self.pad_bottom = ((self.pad_height+1)//2, self.pad_height//2)
        self.pad_left, self.pad_right = ((self.pad_width+1)//2, self.pad_width//2)
        self.padding_all = padding

        #In/Out Sizes
        in_size = input_size
        out_size = (layers,
                    (in_size[1] - self.kernel_height + self.pad_height) // self.stride_h + 1, 
                    (in_size[2] - self.kernel_width + self.pad_width) // self.stride_w + 1)
        super().__init__(in_size, out_size)

        #Strides and Window Shapes
        self.window_shape = (layers, 
                             self.in_size[0], 
                             self.kernel_height, 
                             self.kernel_width, 
                             self.out_size[1], 
                             self.out_size[2]) #(F, C, H, W, out_H, out_W)
        self.grad_shape = (self.in_size[0], 
                           layers, 
                           self.kernel_height, 
                           self.kernel_width, 
                           self.in_size[1] + self.pad_height, 
                           self.in_size[2] + self.pad_width) #(C, F, H, W, in_H + pad, in_W + pad)
        self.dx_shape = (self.out_size[0],
                         self.out_size[1] + (self.out_size[1]-1) * (self.stride_h-1),
                         self.out_size[2] + (self.out_size[2]-1) * (self.stride_w-1)) #(F, out_H + dilate, out_W + dilate)

        #Other settings
        self.funct = funct
        self.opt = optimizer
        self.kernel_weights = np.random.uniform(-0.5, 0.5, (layers, self.in_size[0], self.kernel_height, self.kernel_width))
        self.bias_weights = np.zeros(self.out_size) if biases else None
        self.use_bias = biases
    

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
        """
        new_data = np.expand_dims(data, axis=0) if len(data.shape) < 4 else data    #Enforce batches.
        new_data = np.pad(new_data, ((0,0), (0,0),
                                     (self.pad_top, self.pad_bottom), 
                                     (self.pad_left, self.pad_right)), mode="constant")
        
        #Sliding view strides and window shape
        data_strides = (new_data.strides[0],
                        0, 
                        new_data.strides[1], 
                        new_data.strides[2], 
                        new_data.strides[3], 
                        self.stride_h * new_data.strides[2], 
                        self.stride_w * new_data.strides[3])
        data_win_shape = (new_data.shape[0],) + self.window_shape

        #Sliding windows and convolution multiplication
        sliding_view = np.lib.stride_tricks.as_strided(new_data, 
                                                       shape=data_win_shape, 
                                                       strides=data_strides)
        conv_val = np.einsum("nfchwxy,fchw->nfxy", sliding_view, self.kernel_weights)

        last = self.funct(conv_val + self.bias_weights if self.use_bias else conv_val)
        if training:
            self.last_in = new_data
            self.last_out = last

        if len(data.shape) < 4:
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
        """
        new_err = np.expand_dims(cost_err, axis=0) if len(cost_err.shape) < 4 else cost_err   #Enforce batches.
        new_err = new_err * self.funct(self.last_out, True)
        opt_grad = self.opt.process_grad(new_err)

        main_strides = (self.last_in.strides[0],
                        0, 
                        self.last_in.strides[1], 
                        self.last_in.strides[2], 
                        self.last_in.strides[3], 
                        self.stride_h * self.last_in.strides[2], 
                        self.stride_w * self.last_in.strides[3])
        main_win_shape = (new_err.shape[0],) + self.window_shape

        #Input gradient
        flipped_weights = np.flip(self.kernel_weights, axis=(2, 3))
        dx_ins = dilate_array(new_err, (new_err.shape[0],) + self.dx_shape, (self.stride_h, self.stride_w))

        p_h = self.last_in.shape[2] + self.kernel_height - (dx_ins.shape[2] + 1)
        p_w = self.last_in.shape[3] + self.kernel_width - (dx_ins.shape[3] + 1)
        pad_grad = np.pad(dx_ins, ((0, 0), (0, 0),
                                   (p_h // 2, (p_h + 1) // 2), 
                                   (p_w // 2, (p_w + 1) // 2)), mode="constant")
        
        #Layer weight gradient strides and window shape
        grad_strides = (pad_grad.strides[0],
                        0, 
                        pad_grad.strides[1], 
                        pad_grad.strides[2], 
                        pad_grad.strides[3], 
                        pad_grad.strides[2], 
                        pad_grad.strides[3])
        grad_win_shape = (new_err.shape[0],) + self.grad_shape
        
        grad_window = np.lib.stride_tricks.as_strided(pad_grad, 
                                                      grad_win_shape, 
                                                      grad_strides)
        ret_grad = np.einsum("ncfhwxy,fchw->ncxy", grad_window, flipped_weights)
        ret_grad = ret_grad[:, :, self.pad_top:(-self.pad_bottom or None), 
                                  self.pad_left:(-self.pad_right or None)]

        #Updating weights
        main_window = np.lib.stride_tricks.as_strided(self.last_in, 
                                                      main_win_shape, 
                                                      main_strides)
        self.kernel_weights += np.einsum("nfchwxy,nfxy->fchw", main_window, opt_grad)
                    
        if self.use_bias:
            self.bias_weights += opt_grad.sum(axis=0)

        if len(cost_err.shape) < 4:
            ret_grad = ret_grad.squeeze(axis=0) 
        return ret_grad
    

    def step(self) -> None:
        """Adds one step to this layer's optimizer and scheduler."""
        self.opt.step()


    def clear_grad(self) -> None:
        """Clears the optimizer gradient history and deletes any data required by the backward pass."""
        self.last_in = None
        self.last_out = None
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
    

    def deepcopy(self) -> 'Conv2D':
        """Creates a new deepcopy of this layer with the exact same weights and parameters."""
        new_neuron = Conv2D(self.funct, self.kernel_weights.shape[0], 
                            (self.kernel_height, self.kernel_width), self.in_size, 
                            (self.stride_h, self.stride_w), self.padding_all, 
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
                        f"\u00A0{self.kernel_height}\u00A0{self.kernel_width}" + \
                        f"\u00A0{self.stride_h}\u00A0{self.stride_w}" + \
                        f"\u00A0{self.pad_height}\u00A0{self.pad_width}" + \
                        f"\u00A0{int(self.use_bias)}\n" + \
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
    def from_save(context: str, file_load: bool = False) -> 'Conv2D':
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
        Conv2D
            A new `Conv2D` layer containing all of the information encoded in the string or file provided.
        """
        def parse_and_return(handled_str: str):
            data_arr = handled_str.splitlines()
            prop_info = data_arr[0].split("\u00A0")
            input_info = data_arr[-2].strip().split()[1:]

            bias_size = tuple(map(int, data_arr[1].split()[1:]))
            biases = np.array(list(map(int, data_arr[2].strip().split()))).reshape(bias_size)
            kernel_size = tuple(map(int, data_arr[3].split()[1:]))
            kernels = np.array(list(map(int, data_arr[4].strip().split()))).reshape(kernel_size)

            new_neuron = Conv2D(parse_act_info(prop_info[1]),                 #Activation 
                                int(prop_info[2]),                              #Layers
                                (int(prop_info[3]), int(prop_info[4])),         #Kernel size
                                tuple(map(int, input_info)),                    #Input size
                                (int(prop_info[5]), int(prop_info[6])),         #Strides
                                (int(prop_info[7]), int(prop_info[8])),         #Padding
                                bool(prop_info[9]))                             #Use-bias
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
    def from_kernel(funct: Activation, input_size: tuple[int, int, int], 
                    kernel: np.ndarray, strides: tuple[int, int] | int = 1, padding: tuple[int, int] | int = 0, 
                    bias: np.ndarray = None, optimizer: Optimizer = StandardGD()) -> 'Conv2D':
        """
        Creates a `Conv2D` layer from a pre-constructed set of weights and biases.
        
        Notes
        -----
        The kernel shape for this layer should be as follows:

        `(F, C, kH, kW)`, where `F` is the number of output filters/channels,
        `C` is the number of input channels, and `kH`, `kW` are the kernel sizes.

        If the sizes do not match the input, an AssertionError is raised.


        Parameters
        ----------
        funct : Activation
            An activation function class which supports both forward and backward non-linear
            transformations.
        input_size : tuple[int, int, int]
            A tuple of integers matching the shape of the expected input arrays.
        kernel : ndarray
            An array which will be the new layer's set of kernel weights. Based on the sizes of the kernel,
            will set the number of filters.
        strides : tuple[int, int] | int, default: 1
            An integer or tuple of integers that determines how many data points are skipped for 
            every iteration of the sliding window. Must be greater than or equal to 1.
        padding : tuple[int, int] | int, default: 0
            An integer or tuple of integers that determines how many empty data points are put on 
            the edges of the final dimension as padding layers.
        bias : ndarray, default: None
            An array which will be the new layer's set of bias weights. If set to None, the layer will not
            use any biases.
        opt : Optimizer, default: StandardGD()
            An optimizer class which processes given loss gradients and adjusts them to match a desired 
            gradient descent path.


        Raises
        ------
        AssertionError
            If the input channel or batch sizes are not equal between the kernel and the inputs, or if the
            kernel is not the correct shape length.


        Returns
        -------
        Conv2D
            A new `Conv2D` layer containing all of the information given and interpreted from the input kernel.
        """
        assert len(kernel.shape) == 4, f"Kernel shape must be of length 4 instead of length {len(kernel.shape)}."
        assert input_size[0] == kernel.shape[1], "Kernel channel dimension must be equal to the input channels."

        conv_layer = Conv2D(funct, kernel.shape[0], tuple(kernel.shape[-2:]), 
                                  input_size, strides, padding,
                                  True if bias else False, optimizer)
        conv_layer.kernel_weights = kernel
        conv_layer.bias_weights = bias
        return conv_layer
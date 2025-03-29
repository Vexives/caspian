from caspian.cudalib import np
from . import Layer
from caspian.pooling import PoolFunc, parse_pool_info

class Pooling3D(Layer):
    """
    A 2D pooling layer which performs a downsampling transformation on the data provided.

    Only supports data with 3 or 4 (when batch is included) dimensions as input. The exact shape
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
        A tuple containing the expected output size `(C, Od, Oh, Ow)`, where `C` is the same 
        as the input, with `Od`, `Oh`, `Ow` representing the final pooled dimensions of the output.
    funct : PoolFunc
        The given pooling function which takes specific data from each partition of the input.
    stride_d, stride_h, stride_w : int
        The number of data points that the kernel will move over at each step of pooling. Represents
        depth, height, and width strides respectively.
    kernel_depth, kernel_height, kernel_width : int
        The size of each partition that will be taken from the original input array. Represents the 
        depth, height, and width of the partition, respectively.
    pad_depth, pad_height, pad_width : int
        The total number of data points to be added to the input array as padding on the depth, height, and
        width dimensions, respectively.
    pad_left, pad_right : int
        The number of data points to be added to the left and right sides of the data, respectively.
        Corresponds to each half of `pad_width`, with `pad_left` being the first to increment.
    pad_top, pad_bottom : int
        The number of data points to be added to the top and bottom sides of the data, respectively.
        Corresponds to each half of `pad_height`, with `pad_top` being the first to increment.
    pad_front, pad_back : int
        The number of data points to be added to the front and back sides of the data, respectively.
        Corresponds to each half of `pad_depth`, with `pad_front` being the first to increment.
    opt : Optimizer
        The provided optimizer which modifies the learning gradient before updating weights.


    Examples
    --------
    >>> layer1 = Pooling3D(Maximum(), 3, (5, 9, 12, 6), 3)
    >>> in_arr = np.random.uniform(0.0, 1.0, (5, 9, 12, 6))
    >>> out_arr = layer1(in_arr)
    >>> print(out_arr.shape)
    (5, 3, 4, 2)
    """
    def __init__(self, pool_funct: PoolFunc, kernel_size: tuple[int, int, int] | int, 
                 input_size: tuple[int, int, int, int], 
                 strides: tuple[int, int, int] | int = 1, padding: tuple[int, int, int] | int = 0) -> None:
        """
        Initializes a `Pooling3D` layer using given parameters.

        Parameters
        ----------
        pool_funct : PoolFunc
            A pooling function class which supports both forward and backward pooling 
            transformations.
        kernel_size : tuple[int, int, int] | int
            An integer or tuple of two integers representing the size of the sliding window to 
            extract partitions of the input data.
        input_size : tuple[int, int, int, int]
            A tuple of integers matching the shape of the expected input arrays. If a fifth dimension is added,
            the first dimension is used as the batch size.
        strides : tuple[int, int, int] | int, default: 1
            An integer that determines how many data points are skipped for every iteration of the 
            sliding window. Must be greater than or equal to 1.
        padding : tuple[int, int, int] | int, default: 0
            An integer that determines how many empty data points are put on the edges of the final dimensions
            as padding layers before pooling.
        """
        #Pooling function
        self.funct = pool_funct

        #Strides and Kernel size initialization
        self.stride_d, self.stride_h, self.stride_w = strides if isinstance(strides, tuple) else (strides, strides, strides)

        self.kernel_depth, self.kernel_height, self.kernel_width = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size, kernel_size)

        #Padding size initialization
        self.pad_depth, self.pad_height, self.pad_width = padding if isinstance(padding, tuple) else (padding, padding, padding)
        self.pad_front, self.pad_back = ((self.pad_depth+1)//2, self.pad_depth//2)
        self.pad_top, self.pad_bottom = ((self.pad_height+1)//2, self.pad_height//2)
        self.pad_left, self.pad_right = ((self.pad_width+1)//2, self.pad_width//2)

        #Out-shape and sliding window shape initialization
        in_size = input_size
        out_size = (in_size[0],
                    (in_size[1] - self.kernel_depth + self.pad_depth) // self.stride_d + 1,
                    (in_size[2] - self.kernel_height + self.pad_height) // self.stride_h + 1, 
                    (in_size[3] - self.kernel_width + self.pad_width) // self.stride_w + 1)
        super().__init__(in_size, out_size)
        self.__window_shape = (*self.out_size, 
                             self.kernel_depth,
                             self.kernel_height, 
                             self.kernel_width)


    def forward(self, data: np.ndarray, training: bool = False) -> np.ndarray:
        """
        Performs a forward propagation pass through this layer given the current initialization parameters.
        
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
        new_data = np.expand_dims(data, axis=0) if len(data.shape) < 5 else data    #Enforce batches.
        data_padded = np.pad(new_data, pad_width=((0, 0), (0, 0),
                                                  (self.pad_front, self.pad_back),
                                                  (self.pad_top, self.pad_bottom), 
                                                  (self.pad_left, self.pad_right)), mode='constant')
        strides = (data_padded.strides[0],
                   data_padded.strides[1],
                   self.stride_d * data_padded.strides[2], 
                   self.stride_h * data_padded.strides[3], 
                   self.stride_w * data_padded.strides[4], 
                   data_padded.strides[2], 
                   data_padded.strides[3],
                   data_padded.strides[4])
        data_win_shape = (new_data.shape[0],) + self.__window_shape

        #Split into windows, and apply the pooling function to each window.
        data_windows = np.lib.stride_tricks.as_strided(data_padded, 
                                                       shape=data_win_shape, 
                                                       strides=strides)
        pool_val = self.funct(data_windows.reshape((*data_windows.shape[:-3], -1)))

        if training:
            self.__last_in = data_padded
            self.__last_out = pool_val

        if len(data.shape) < 5:
            pool_val = pool_val.squeeze(axis=0)
        return pool_val
    

    def backward(self, cost_err: np.ndarray) -> np.ndarray:
        """
        Performs a backward propagation pass through this layer and returns a gradient fit for the
        previous layer.

        Parameters
        ----------
        cost_err : ndarray
            The learning gradient that will be processed and returned.

        Returns
        -------
        ndarray
            The new learning gradient for any layers that provided data to this instance. Will have the
            same shape as this layer's input shape.
        """
        new_err = np.expand_dims(cost_err, axis=0) if len(cost_err.shape) < 5 else cost_err   #Enforce batches.
        strides = (self.__last_in.strides[0], 
                   self.__last_in.strides[1],
                   self.stride_d * self.__last_in.strides[2],
                   self.stride_h * self.__last_in.strides[3], 
                   self.stride_w * self.__last_in.strides[4], 
                   self.__last_in.strides[2], 
                   self.__last_in.strides[3],
                   self.__last_in.strides[4])
        main_win_shape = (new_err.shape[0],) + self.__window_shape
        
        #Window frames for previous input / Mask creation
        main_windows = np.lib.stride_tricks.as_strided(self.__last_in,
                                                       main_win_shape, 
                                                       strides)
        mask = self.funct(main_windows.reshape((*main_windows.shape[:-3], -1)), backward=True) \
                         .reshape(main_windows.shape)

        #Use mask to distribute the gradient into the mask, reshaped into (channels, kernel height, kernel width, num of windows)
        pre_grad = np.einsum("ngdhw,ngdhwxyz->ngdhwxyz", new_err, mask)

        # Zero array of original size (channels, in height, in width)
        ret_grad = np.zeros_like(self.__last_in)
        ret_windows = np.lib.stride_tricks.as_strided(ret_grad, 
                                                      main_win_shape, 
                                                      strides)
        np.add.at(ret_windows, (slice(None)), pre_grad)
        
        #Final cleanup
        ret_grad = ret_grad[:, :,
                            self.pad_front:(-self.pad_back or None),
                            self.pad_top:(-self.pad_bottom or None), 
                            self.pad_left:(-self.pad_right or None)]
        if len(cost_err.shape) < 5:
            ret_grad = ret_grad.squeeze(axis=0)   
        return ret_grad


    def step(self) -> None:
        """Not applicable for this layer."""
        pass


    def clear_grad(self) -> None:
        """Clears any data required by the backward pass and sets the variables to `None`."""
        self.__last_in = None
        self.__last_out = None


    def set_optimizer(self, *_) -> None:
        """Not applicable for this layer."""
        pass   


    def deepcopy(self) -> 'Pooling3D':
        """Creates a new deepcopy of this layer with the exact same parameters."""
        new_neuron = Pooling3D(self.funct, 
                               (self.kernel_depth, self.kernel_height, self.kernel_width), self.in_size, 
                               (self.stride_d, self.stride_h, self.stride_w), 
                               (self.pad_depth, self.pad_height, self.pad_width))
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
        write_ret_str = f"Pooling3D\u00A0{repr(self.funct)}\u00A0" + " ".join(list(map(str, self.in_size))) + \
                        f"\nLENS\u00A0{self.kernel_depth}\u00A0{self.kernel_height}\u00A0{self.kernel_width}" + \
                        f"\u00A0{self.stride_d}\u00A0{self.stride_h}\u00A0{self.stride_w}" + \
                        f"\u00A0{self.pad_depth}\u00A0{self.pad_height}\u00A0{self.pad_width}\n\u00A0"
        if not filename:
            return write_ret_str
        if filename.find(".cspn") == -1:
            filename += ".cspn"
        with open(filename, "w+") as file:
            file.write(write_ret_str)
            file.close()


    @staticmethod
    def from_save(context: str, file_load: bool = False) -> 'Pooling3D':
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
        Pooling3D
            A new `Pooling3D` layer containing all of the information encoded in the string or file provided.
        """
        def parse_and_return(handled_str: str):
            data_arr = handled_str.splitlines()
            main_info = data_arr[0].split("\u00A0")
            funct = parse_pool_info(main_info[1])
            in_size = tuple(map(int, main_info[2].split()))

            sec_info = data_arr[1].split("\u00A0")[1:]
            k_sizes = (int(sec_info[0]), int(sec_info[1]), int(sec_info[2]))
            strides = (int(sec_info[3]), int(sec_info[4]), int(sec_info[5]))
            padding = (int(sec_info[6]), int(sec_info[7]), int(sec_info[8]))
            return Pooling3D(funct, k_sizes, in_size, strides, padding)

        if file_load:
            full_parse_str = None
            with open(context, "r") as file:
                full_parse_str = file.read()
                file.close()
            return parse_and_return(full_parse_str)
        return parse_and_return(context)
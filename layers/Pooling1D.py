from caspian.cudalib import np
from . import Layer
from caspian.pooling import PoolFunc, parse_pool_info

class Pooling1D(Layer):
    """
    A 1D pooling layer which performs a downsampling transformation on the data provided.

    Only supports data with 2 or 3 (when batch is included) dimensions as input. The exact shape
    and/or batch size must be specifically stated when initializing the layer.


    Attributes
    ---------
    in_size : tuple[int, int]
        A tuple containing the expected input size `(C, W)`, where `C` is the number of channels, 
        and `W` is the final dimension of the input.
    out_size : tuple[int, int]
        A tuple containing the expected output size `(C, Ow)`, where `C` is the same 
        as the input, with `Ow` representing the final pooled dimension of the output.
    funct : PoolFunc
        The given pooling function which takes specific data from each partition of the input.
    strides : int
        The number of data points that the kernel will move over at each step of pooling.
    kernel_size : int
        The size of each partition that will be taken from the original input array.
    padding_all : int
        The total number of data points to be added to the input array as padding.
    pad_left, pad_right : int
        The number of data points to be added to the left and right sides of the data, respectively.
        Corresponds to each half of `padding_all`, with `pad_left` being the first to increment.
    window_shape : tuple[int, int, int]
        The shape that the data will take whenever a strided view is created for forward and backward
        passes.
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
    >>> layer1 = Pooling1D(Maximum(), 3, (5, 9), 3)
    >>> in_arr = np.random.uniform(0.0, 1.0, (5, 9))
    >>> out_arr = layer1(in_arr)
    >>> print(out_arr.shape)
    (5, 3)
    """
    def __init__(self, pool_funct: PoolFunc, kernel_size: int, 
                 input_size: tuple[int, int], 
                 strides: int = 1, padding: int = 0) -> None:
        """
        Initializes a `Pooling1D` layer using given parameters.

        Parameters
        ----------
        pool_funct : PoolFunc
            A pooling function class which supports both forward and backward pooling 
            transformations.
        kernel_size : int
            An integer representing the size of the sliding window to extract partitions of the input data.
        input_size : tuple[int, int]
            A tuple of integers matching the shape of the expected input arrays. If a third dimension is added,
            the first dimension is used as the batch size.
        strides : int, default: 1
            An integer that determines how many data points are skipped for every iteration of the 
            sliding window. Must be greater than or equal to 1.
        padding : int, default: 0
            An integer that determines how many empty data points are put on the edges of the final dimension
            as padding layers before pooling.
        """
        #Pooling Function
        self.funct = pool_funct

        #Strides and Kernel size initialization
        self.strides = strides
        self.kernel_size = kernel_size

        #Padding Initialization
        self.padding_all = padding
        self.pad_left, self.pad_right = ((padding+1)//2, padding//2)
        
        #Out-Shape and Sliding Window Shape Initialization
        in_size = input_size
        out_size = (in_size[0],
                    (in_size[1] - self.kernel_size + padding) // self.strides + 1)
        super().__init__(in_size, out_size)
        self.window_shape = (*self.out_size, self.kernel_size)


    def forward(self, data: np.ndarray, training: bool = False) -> np.ndarray:
        """
        Performs a forward propagation pass through this layer given the current initialization parameters.
        
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
        new_data = np.expand_dims(data, axis=0) if len(data.shape) < 3 else data    #Enforce batches.
        data_padded = np.pad(new_data, ((0, 0), (0, 0), 
                                        (self.pad_left, self.pad_right)), mode="constant")
        strides = (data_padded.strides[0], 
                   data_padded.strides[1],
                   self.strides * data_padded.strides[2], 
                   data_padded.strides[2])
        data_win_shape = (new_data.shape[0],) + self.window_shape

        #Split data into windows, then apply function to each window.
        data_windows = np.lib.stride_tricks.as_strided(data_padded, 
                                                       shape=data_win_shape, 
                                                       strides=strides)
        pool_val = self.funct(data_windows)

        if training:
            self.last_in = data_padded
            self.last_out = pool_val

        if len(data.shape) < 3:
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
        new_err = np.expand_dims(cost_err, axis=0) if len(cost_err.shape) < 3 else cost_err   #Enforce batches.
        main_strides = (self.last_in.strides[0], 
                        self.last_in.strides[1],
                        self.strides * self.last_in.strides[2], 
                        self.last_in.strides[2])
        main_win_shape = (new_err.shape[0],) + self.window_shape
        
        #Window frames for previous input / Mask creation
        main_windows = np.lib.stride_tricks.as_strided(self.last_in, 
                                                       main_win_shape, 
                                                       main_strides)
        mask = self.funct(main_windows, backward=True)

        #Use mask to distribute the gradient into input size
        pre_grad = np.einsum("ngw,ngwx->ngwx", new_err, mask)

        #Zero array of original size
        ret_grad = np.zeros_like(self.last_in)
        ret_windows = np.lib.stride_tricks.as_strided(ret_grad, 
                                                      main_win_shape, 
                                                      main_strides)
        np.add.at(ret_windows, (slice(None)), pre_grad)
        
        #Final cleanup
        ret_grad = ret_grad[:, :, self.pad_left:(-self.pad_right or None)]
        if len(cost_err.shape) < 3:
            ret_grad = ret_grad.squeeze(axis=0)    
        return ret_grad


    def step(self) -> None:
        """Not applicable for this layer."""
        pass


    def clear_grad(self) -> None:
        """Clears any data required by the backward pass and sets the variables to `None`."""
        self.last_in = None
        self.last_out = None


    def set_optimizer(self, *_) -> None:
        """Not applicable for this layer."""
        pass    


    def deepcopy(self) -> 'Pooling1D':
        """Creates a new deepcopy of this layer with the exact same parameters."""
        new_neuron = Pooling1D(self.funct, self.kernel_size, self.in_size, 
                               self.strides, self.padding_all)
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
        write_ret_str = f"Pooling1D\u00A0{repr(self.funct)}\u00A0" + " ".join(list(map(str, self.in_size))) + \
                        f"\nLENS\u00A0{self.kernel_size}\u00A0{self.strides}\u00A0{self.padding_all}\n\u00A0"
        if not filename:
            return write_ret_str
        if filename.find(".cspn") == -1:
            filename += ".cspn"
        with open(filename, "w+") as file:
            file.write(write_ret_str)
            file.close()
    

    @staticmethod
    def from_save(context: str, file_load: bool = False) -> 'Pooling1D':
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
        Pooling1D
            A new `Pooling1D` layer containing all of the information encoded in the string or file provided.
        """
        def parse_and_return(handled_str: str):
            data_arr = handled_str.splitlines()
            main_info = data_arr[0].split("\u00A0")
            funct = parse_pool_info(main_info[1])
            in_size = tuple(map(int, main_info[2].split()))

            sec_info = data_arr[1].split("\u00A0")
            k_len, strides, padding = tuple(map(int, sec_info[1:]))
            return Pooling1D(funct, k_len, in_size, strides, padding)

        if file_load:
            full_parse_str = None
            with open(context, "r") as file:
                full_parse_str = file.read()
                file.close()
            return parse_and_return(full_parse_str)
        return parse_and_return(context)
from ..cudalib import np
from . import Layer
from ..pooling import PoolFunc, parse_pool_info
from ..utilities import all_positive, all_ints, confirm_shape, check_types, \
                        UnsafeMemoryAccessException, InvalidDataException

class PoolingND(Layer):
    """
    An any-dimensional pooling layer which performs a downsampling transformation on the data provided.

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
        A tuple containing the expected output size `(C, *)`, where `C` is the same number of channels
        as the input, and `*` represents the new output dimension size. Will have `N` dimensions.
    funct : PoolFunc
        The given pooling function which takes specific data from each partition of the input.
    pad_details : tuple[tuple[int, int], ...]
        A tuple of tuples, defining the padding across each of the pooled dimensions given.
    padding_all : tuple[int, ...]
        The padding input value in tuple form that was provided at initialization. If the padding was given
        as an integer, then the tuple will be of size `N-1`
    strides_all : tuple[int, ...]
        A tuple of integers representing the strides for each pooled dimension. Will have a size of `N-1`.
    kernel_size : tuple[int, ...]
        A tuple of integers representing the kernel size for each pooled dimension. Will have a size of `N-1`.
    input_length : int
        An integer representing the total number of dimensions (including channels) that is expected from
        the input data. 


    Examples
    --------
    >>> layer1 = PoolingND(Maximum(), 3, (5, 9, 12, 6, 9, 12), 3)
    >>> in_arr = np.random.uniform(0.0, 1.0, (5, 9, 12, 6, 9, 12))
    >>> out_arr = layer1(in_arr)
    >>> print(out_arr.shape)
    (5, 3, 4, 2, 3, 4)
    """
    @check_types(("kernel_size", all_positive, "Argument \"kernel_size\" must be greater than 0."),
                 ("kernel_size", all_ints, "Argument \"kernel_size\" must contain all integers."),

                 ("strides", all_positive, "Argument \"strides\" must be greater than 0."),
                 ("strides", all_ints, "Argument \"strides\" must contain all integers."),                  

                 ("padding", lambda x: all_positive(x, True), "Argument \"padding\" must be greater than or equal to 0."),
                 ("padding", all_ints, "Argument \"padding\" must contain all integers."),

                 ("input_size", all_positive, "Argument \"input_size\" must contain all positive values above 0."),
                 ("input_size", lambda x: len(x) >= 2, "Argument \"input_size\" must have a length of at least 2."))
    def __init__(self, pool_funct: PoolFunc, 
                 kernel_size: tuple[int, ...] | int, 
                 input_size: tuple[int, ...], 
                 strides: tuple[int, ...] | int = 1, 
                 padding: tuple[int, ...] | int = 0) -> None:
        """
        Initializes a `PoolingND` layer using given parameters.

        Parameters
        ----------
        pool_funct : PoolFunc
            A pooling function class which supports both forward and backward pooling 
            transformations.
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

        Raises
        ------
        InvalidDataException
            If any of the data provided is not an integer or tuple of integers, or less than one 
            (with the exception of padding, which can be 0). Expected input size must be a tuple of integers, and
            the pooling function must be of type `PoolFunc`. Also applies to the expected input shape, which must 
            be a tuple of integers. Padding, strides, and kernel size must ALL be either an integer or a tuple with 
            N-1 dimensions, where input size has N > 2 dimensions. 
        """
        # Extra dimensionality checks before processing
        _in_len = len(input_size)-1
        if type(kernel_size) != int and len(kernel_size) != _in_len:
            raise InvalidDataException("Argument \"kernel_size\" must have one less dimension than expected input.")
        if type(strides) != int and len(strides) != _in_len:
            raise InvalidDataException("Argument \"strides\" must have one less dimension than expected input.")
        if type(padding) != int and len(padding) != _in_len:
            raise InvalidDataException("Argument \"strides\" must have one less dimension than expected input.")        

        #Pooling function
        self.funct = pool_funct

        #Strides and Kernel size initialization
        self.input_length = len(input_size)
        self.strides_all = strides if isinstance(strides, tuple) else (strides,) * _in_len
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size,) * _in_len

        #Padding size initialization
        self.padding_all = (padding,) * _in_len if type(padding) == int else padding
        self.pad_details = ((0,0), (0,0)) + tuple(((p+1)//2, p//2) for p in self.padding_all)

        #Out-shape and sliding window shape initialization
        in_size = input_size
        out_size = (in_size[0],) + \
                    tuple((in_size[i+1] - self.kernel_size[i] + self.padding_all[i]) // self.strides_all[i] + 1 \
                         for i in range(_in_len))
        super().__init__(in_size, out_size)
        self.__window_shape = (*self.out_size, 
                               *self.kernel_size)
        

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

        Raises
        ------
        UnsafeMemoryAccessException
            If the shape of the given array will lead to any un-safe memory calls during the pass.
        """
        if not confirm_shape(data.shape, self.in_size, self.input_length):
            raise UnsafeMemoryAccessException(f"Input data shape does not match expected shape. - {data.shape}, {self.in_size}")
        new_data = np.expand_dims(data, axis=0) if len(data.shape) < self.input_length+1 else data    #Enforce batches.
        data_padded = np.pad(new_data, pad_width=self.pad_details, mode='constant')

        strides = (data_padded.strides[0],
                   data_padded.strides[1],
                   *tuple(map(lambda x,y: x*y, self.strides_all, data_padded.strides[2:])), 
                   *data_padded.strides[2:])
        data_win_shape = (new_data.shape[0],) + self.__window_shape

        #Split into windows, and apply the pooling function to each window.
        data_windows = np.lib.stride_tricks.as_strided(data_padded, 
                                                       shape=data_win_shape, 
                                                       strides=strides)
        pool_val = self.funct(data_windows.reshape((*data_windows.shape[:-self.input_length+1], -1)))

        if training:
            self.__last_in = data_padded

        if len(data.shape) < self.input_length+1:
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

        Raises
        ------
        UnsafeMemoryAccessException
            If the shape of the given array will lead to any un-safe memory calls during the pass.
        """
        if not confirm_shape(cost_err.shape, self.out_size, self.input_length):
            raise UnsafeMemoryAccessException(f"Input data shape does not match expected shape. - {cost_err.shape}, {self.in_size}")
        new_err = np.expand_dims(cost_err, axis=0) if len(cost_err.shape) < self.input_length+1 else cost_err   #Enforce batches.
        strides = (self.__last_in.strides[0], 
                   self.__last_in.strides[1],
                   *tuple(map(lambda x,y: x*y, self.strides_all, self.__last_in.strides[2:])),
                   *self.__last_in.strides[2:])
        main_win_shape = (new_err.shape[0],) + self.__window_shape
        
        #Window frames for previous input / Mask creation
        main_windows = np.lib.stride_tricks.as_strided(self.__last_in,
                                                       main_win_shape, 
                                                       strides)
        mask = self.funct(main_windows.reshape((*main_windows.shape[:-self.input_length+1], -1)), backward=True) \
                         .reshape(main_windows.shape)

        #Use mask to distribute the gradient into the mask, reshaped into (channels, kernel height, kernel width, num of windows)
        grad_match = (Ellipsis,) + (None,) * (self.input_length-1)
        pre_grad = (mask * new_err[grad_match])

        # Zero array of original size (channels, in height, in width)
        ret_grad = np.zeros_like(self.__last_in)
        ret_windows = np.lib.stride_tricks.as_strided(ret_grad, 
                                                      main_win_shape, 
                                                      strides)
        np.add.at(ret_windows, (slice(None)), pre_grad)
        
        #Final cleanup
        clip_vals = tuple(slice(x, (-y or None)) for (x, y) in self.pad_details)
        ret_grad = ret_grad[clip_vals]
        if len(cost_err.shape) < self.input_length+1:
            ret_grad = ret_grad.squeeze(axis=0)   
        return ret_grad


    def step(self) -> None:
        """Not applicable for this layer."""
        pass


    def clear_grad(self) -> None:
        """Clears any data required by the backward pass and sets the variables to `None`."""
        self.__last_in = None


    def set_optimizer(self, *_) -> None:
        """Not applicable for this layer."""
        pass   


    def deepcopy(self) -> 'PoolingND':
        """Creates a new deepcopy of this layer with the exact same parameters."""
        new_neuron = PoolingND(self.funct, 
                               self.kernel_size, 
                               self.in_size, 
                               self.strides_all, 
                               self.padding_all)
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
        write_ret_str = f"PoolingND\u00A0{repr(self.funct)}\u00A0" + " ".join(list(map(str, self.in_size))) + \
                        f"\nLENS\u00A0" + " ".join(list(map(str, self.kernel_size))) + \
                        f"\u00A0" + " ".join(list(map(str, self.strides_all))) + \
                        f"\u00A0" + " ".join(list(map(str, self.padding_all))) + "\n\u00A0"
        if not filename:
            return write_ret_str
        if filename.find(".cspn") == -1:
            filename += ".cspn"
        with open(filename, "w+") as file:
            file.write(write_ret_str)
            file.close()


    @staticmethod
    def from_save(context: str, file_load: bool = False) -> 'PoolingND':
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
        PoolingND
            A new `PoolingND` layer containing all of the information encoded in the string or file provided.
        """
        def parse_and_return(handled_str: str):
            data_arr = handled_str.splitlines()
            main_info = data_arr[0].split("\u00A0")
            funct = parse_pool_info(main_info[1])
            in_size = tuple(map(int, main_info[2].split()))

            sec_info = data_arr[1].split("\u00A0")[1:]
            k_sizes = tuple(map(int, sec_info[0].strip().split()))
            strides = tuple(map(int, sec_info[1].strip().split()))
            padding = tuple(map(int, sec_info[2].strip().split()))
            return PoolingND(funct, k_sizes, in_size, strides, padding)

        if file_load:
            full_parse_str = None
            with open(context, "r") as file:
                full_parse_str = file.read()
                file.close()
            return parse_and_return(full_parse_str)
        return parse_and_return(context)
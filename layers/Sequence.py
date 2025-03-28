from caspian.cudalib import np
from . import Layer
from caspian.optimizers import Optimizer, StandardGD

class Sequence(Layer):
    """
    A sequence layer which takes in multiple layers in a list as the primary argument and runs
    the forward and backward passes through the entire sequence before returning the result.

    All layers must have corresponding input and output shapes to function properly. The order in
    which the layers are added are kept, and determines the path which the data will take.
       
    
    Notes
    -----
    Using a `Sequence` layer inside of another `Sequence` layer may cause issues with saving and loading
    via the `save_to_file()` and `from_save()` methods. It is possible to have a `Sequence` inside of
    another `Sequence`, but not all functions and behaviors may work as expected.
    
    Some other layers may also not be used inside of a `Sequence`, specifically layers which expect
    either multiple inputs or multiple gradients to work properly. An example of this would be the
    `Recurrent` cell layer, or the `Bilinear` layer.
        
    
    Attributes
    ---------
    layers : list[Layer]
        A list of layers that (in order) will process the data and return the result. The forward pass
        will go from the first layer to the last, the backward pass will perform the opposite.
    num_layers : int
        An integer representing the number of layers currently in the `Sequence`.
    opt : Optimizer
        The provided optimizer which modifies the learning gradient before updating weights.
    trainable : bool
        A boolean which determines if the entire `Sequence` is able to be trained using a backward pass.
        Is set to `False` by default unless a forward pass is performed in training mode. 


    Examples
    --------
    >>> l1 = Dense(Linear(), 10, 5)
    >>> l2 = Dense(ReLU(), 5, 20)
    >>> seq = Sequence([l1, l2])
    >>> in_arr = np.ones((10,))
    >>> out_arr = seq(in_arr)
    >>> print(out_arr.shape)
    (20,)
    """
    def __init__(self, layers: list[Layer] | Layer, optimizer: Optimizer = StandardGD()):
        """
        Initializes a `Sequence` layer using given parameters.

        Parameters
        ----------
        layers : list[Layer] | Layer
            Either a single layer or a list of layers containing at least 1 `Layer` class.
        opt : Optimizer, default: StandardGD()
            An optimizer class which processes given loss gradients and adjusts them to match a desired 
            gradient descent path.

        Raises
        ------
        AssertionError
            If either the length of the `layers` list is 0 or if any of the layers contained inside of
            the list are disjointed and cannot be seamlessly processed sequentially.
        """
        # Ensure that there is at least one layer in the system at all times.
        if isinstance(layers, list):
            assert len(layers) >= 1, "List of layers must have at least one initial layer."

        # Ensure that all layers can feed into each other seamlessly.
        self.layers = layers if isinstance(layers, list) else [layers]
        for first, second in zip(self.layers[:-1], self.layers[1:]):
            assert self.__verify_shapes(first, second), \
            f"Layer input and output shapes must not be disjoint. {first.out_size} - {second.in_size}"

        in_size = self.layers[0].in_size
        out_size = None
        for layer in reversed(self.layers):
            if layer.out_size is not None:
                self.out_size = layer.out_size
                break
        super().__init__(in_size, out_size)

        self.num_layers = len(self.layers)
        self.opt = optimizer
        self.trainable = False


    def __add__(self, new_layer: Layer) -> 'Sequence':
        """
        Appends a new layer to this `Sequence` layer - will be the new final layer in the list.

        Parameters
        ----------
        new_layer : Layer
            The new layer to be added, must be a valid layer and cannot be disjoint with the current
            last layer.

        Raises
        ------
        AssertionError
            If the new layer provided does not take the same general input shape as the current last
            layer in this `Sequence`, then an error is raised.
        """
        assert self.__verify_shapes(self.layers[-1], new_layer), \
               "Layer input and output shapes must not be disjoint."
        self.layers.append(new_layer)
        self.num_layers += 1
        self.out_size = new_layer.out_size if new_layer.out_size is not None else self.out_size
        return self
    

    def __verify_shapes(self, layer_one: Layer, layer_two: Layer) -> bool:
        """Private method for verifying that two layers are not disjoint."""
        if layer_one.out_size is None or layer_two.in_size is None:
            return True
        min_size = min(len(layer_one.out_size), len(layer_two.in_size))
        return layer_one.out_size[-min_size:] == layer_two.in_size[-min_size:]
    

    def forward(self, data: np.ndarray, training: bool = False) -> np.ndarray:
        """
        Performs a forward propagation pass through this `Sequence` given the current layers.
        
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
            The forward propagated array with the shape equal to this `Sequence`'s output shape.
        """
        for layer in self.layers:
            data = layer.forward(data, training)
        self.trainable = True
        return data


    def backward(self, cost_err: np.ndarray) -> np.ndarray:
        """
        Performs a backward propagation pass through this `Sequence` and updates the weights of the sublayers
        according to the provided learning gradient.

        Parameters
        ----------
        cost_err : ndarray
            The learning gradient that will be processed and returned. Any information gained by the layers' weights
            will be taken from this gradient and will fit to this data.

        Returns
        -------
        ndarray
            The new learning gradient for any layers that provided data to this instance. Will have the
            same shape as this `Sequence`'s input shape.
        """
        for layer in reversed(self.layers):
            cost_err = layer.backward(cost_err)
        return cost_err
    

    def step(self) -> None:
        """Adds one step to each layer in the `Sequence`."""
        for layer in self.layers:
            layer.step()


    def clear_grad(self) -> None:
        """Clears the optimizer gradient history and deletes any data required by the backward pass."""
        self.trainable = False


    def set_optimizer(self, opt: Optimizer = StandardGD()) -> None:
        """
        Sets the optimizer of each sublayer to the one given. Will revert to a standard gradient descent
        optimizer if none is provided.
           
        Parameters
        ----------
        opt : Optimizer, default: StandardGD()
            The new optimizer for this `Sequence` to keep.
        """
        for layer in self.layers:
            layer.set_optimizer(opt.deepcopy())


    def deepcopy(self) -> 'Sequence':
        """Creates a new deepcopy of this `Sequence` with deep copies of each sublayer contained."""
        new_seq = Sequence([layer.deepcopy() for layer in self.layers], self.opt.deepcopy())
        return new_seq


    def save_to_file(self, filename: str = None) -> None | str:
        """
        Encodes the current `Sequence` information into a string, and saves it to a file if the
        path is specified.

        Parameters
        ----------
        filename : str, default: None
            The file for the `Sequence`'s information to be stored to. If this is not provided and
            is instead of type `None`, the encoded string will just be returned.

        Returns
        -------
        str | None
            If no file is specified, a string containing all information about this model is returned.
        """
        write_ret_str = "SEQUENCE\n\u00AD" # Custom sequence break character
        for layer in self.layers:
            write_ret_str += layer.save_to_file() + "\n\u00AD"
        write_ret_str += "\n\u00A0"

        if not filename:
            return write_ret_str
        if filename.find(".cspn") == -1:
            filename += ".cspn"
        with open(filename, "w+") as file:
            file.write(write_ret_str)
            file.close()


    @staticmethod
    def from_save(context: str, file_load: bool = False) -> 'Sequence':
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
        Sequence
            A new `Sequence` layer containing all of the information encoded in the string or file provided.
        """
        def parse_and_return(handled_str: str):
            all_layers = handled_str.split("\n\u00AD")
            processed_layers = [getattr(__import__("layers"), layer.split("\u00A0")[0].strip()).from_save(layer) 
                                for layer in all_layers[1:-1]]
            new_seq = Sequence(processed_layers)
            return new_seq

        if file_load:
            full_parse_str = None
            with open(context, "r") as file:
                full_parse_str = file.read()
                file.close()
            return parse_and_return(full_parse_str)
        return parse_and_return(context)
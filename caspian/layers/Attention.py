from ..cudalib import np
from . import Layer, Linear
from ..activations import Softmax
from ..optimizers import Optimizer, StandardGD, parse_opt_info
from ..utilities import check_types

class Attention(Layer):
    """
    A singular Attention layer which performs a complex attention-mechanism transformation of the given
    data in groups of three, as `query`, `key`, and `value` respectively.

    ``Attention(q, k, v) = Softmax(QK^T / sqrt(d_e))V``

    Supports any given shape and dimensionality as an input except for the final dimension, which is
    provided at initialization as `embed_size`.

        
    Attributes
    ---------
    embed_size : int
        The expected size of the final dimensions for all given data arrays.
    use_mask : bool
        A boolean which determines whether a zeroing mask is applied to the attention weights before
        it is matrix multiplied by the value array.
    use_bias : bool
        A boolean which determines whether the internal `Linear` layers are initialized with a bias weight.


    Examples
    --------
    >>> layer1 = Attention(8, True)
    >>> in_arr = np.random.randn(12, 8)
    >>> out_arr = layer1(in_arr, in_arr, in_arr)
    >>> print(out_arr.shape)
    (12, 8)
    """
    @check_types(("embed_size", lambda x: x > 0, "Argument \"embed_size\" must be greater than 0."))
    def __init__(self, embed_size: int, use_mask: bool = False, 
                 biases: bool = False, optimizer: Optimizer = StandardGD()):
        """
        Initializes an `Attention` layer using given parameters.

        Parameters
        ----------
        embed_size : int
            An integer representing the expected last dimension size for each of the input data arrays.
        use_mask : bool
            A boolean which determines whether a zeroing mask is applied to the attention weights before
            it is matrix multiplied by the value array.
        biases : bool
            A boolean which determines whether the internal `Linear` layers are initialized with a bias weight.
        optimizer : Optimizer, default: StandardGD()
            An optimizer class which processes given loss gradients and adjusts them to match a desired 
            gradient descent path. A copy is given to each of the internal `Linear` layers.

        Raises
        ------
        InvalidDataException
            If the embed size is not an integer, is less than 1, or the parameters given are not the expected type.
        """
        super().__init__((embed_size,), (embed_size,))

        self.d_embed = embed_size
        self.use_mask = use_mask
        self.use_bias = biases

        self.__q_layer = Linear(embed_size, embed_size, biases, optimizer.deepcopy())
        self.__k_layer = Linear(embed_size, embed_size, biases, optimizer.deepcopy())
        self.__v_layer = Linear(embed_size, embed_size, biases, optimizer.deepcopy())
        self.__softmax = Softmax()
    

    def __call__(self, q_data: np.ndarray, k_data: np.ndarray, 
                       v_data: np.ndarray, training: bool = False):
        """Calls the class forward function and provides the given parameters."""
        return self.forward(q_data, k_data, v_data, training)


    def forward(self, q_data: np.ndarray, k_data: np.ndarray, 
                      v_data: np.ndarray, training: bool = False) -> np.ndarray:
        """
        Performs a forward propagation pass through this layer given the current `Linear` layers.
        
        Parameters
        ----------
        q_data : ndarray
            The first data array that is expected for the forward pass. Represents the query matrix and
            the last dimension must match this layer's embedding size.
        k_data : ndarray
            The second data array that is expected for the forward pass. Represents the key matrix and
            the last dimension must match this layer's embedding size.
        v_data : ndarray
            The third data array that is expected for the forward pass. Represents the value matrix and
            the last dimension must match this layer's embedding size.
        training : bool
            Specify whether the layer is currently training or not to save the necessary information
            required for the backward pass.
        
        Returns
        -------
        ndarray
            The forward propagated array with the shape equal to this layer's output shape.
        """
        # Linear-layer processing
        q_vals = self.__q_layer(q_data, training)
        k_vals = self.__k_layer(k_data, training)
        v_vals = self.__v_layer(v_data, training)

        # Initial attention scores and value mat-mul
        qk_set = q_vals @ np.moveaxis(k_vals, -1, -2)
        att_score = qk_set / np.sqrt(self.d_embed)
        if self.use_mask:                                               # Pre-softmax mask
            self.__mask = np.triu(np.ones_like(att_score) * -np.inf, 1)
            att_score += self.__mask
        att_score = self.__softmax(att_score)

        if training:
            self.__last_ins = (q_data, k_data, v_data)
            self.__last_score = att_score
        return att_score @ v_vals
    

    def backward(self, cost_err: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Performs a backward propagation pass through this layer and updates the internal linear layers 
        according to the provided learning gradient.

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
        last_q, last_k, last_v = self.__last_ins

        # Re-apply mask if applicable, and calculate the main score gradient
        new_err = cost_err @ np.moveaxis(last_v, -1, -2)
        soft_grad = self.__softmax(self.__last_score, new_err)
        if self.use_mask is True:
            soft_grad = np.tril(soft_grad)
        soft_grad /= np.sqrt(self.d_embed)

        # Pre-linear gradients
        v_grad = np.moveaxis(self.__last_score, -1, -2) @ cost_err
        k_grad = np.moveaxis(np.moveaxis(last_q, -1, -2) @ soft_grad, -1, -2)
        q_grad = soft_grad @ last_k

        # Post-linear gradients
        q_grad = self.__q_layer.backward(q_grad)
        k_grad = self.__k_layer.backward(k_grad)
        v_grad = self.__v_layer.backward(v_grad)
        return q_grad, k_grad, v_grad


    def step(self) -> None:
        """Adds one step to this layer's optimizer and scheduler."""
        self.__q_layer.step()
        self.__k_layer.step()
        self.__v_layer.step()


    def clear_grad(self) -> None:
        """Clears the optimizer gradient history and deletes any data required by the backward pass."""
        self.__last_ins = None
        self.__last_score = None
        self.__q_layer.opt.reset_grad()
        self.__k_layer.opt.reset_grad()
        self.__v_layer.opt.reset_grad()


    def set_optimizer(self, opt: Optimizer = StandardGD()) -> None:
        """
        Sets the optimizer of this layer to a new one. Will revert to a standard gradient descent
        optimizer if none is provided.
           
        Parameters
        ----------
        opt : Optimizer, default: StandardGD()
            The new optimizer for this layer to keep.
        """
        self.__q_layer.set_optimizer(opt.deepcopy())
        self.__k_layer.set_optimizer(opt.deepcopy())
        self.__v_layer.set_optimizer(opt.deepcopy())


    def deepcopy(self) -> 'Attention':
        """Creates a new deepcopy of this layer with the exact same weights (if applicable) and parameters."""
        new_neuron = Attention(self.d_embed, self.use_mask, self.use_bias)
        new_neuron.__q_layer = self.__q_layer.deepcopy()
        new_neuron.__k_layer = self.__k_layer.deepcopy()
        new_neuron.__v_layer = self.__v_layer.deepcopy()
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
        write_ret_str = f"Attention\u00A0{self.d_embed}\u00A0{self.use_mask}\u00A0{self.use_bias}\u00A0{repr(self.__q_layer.opt)}" + \
                        "\nQ_WEIGHTS\u00A0" + " ".join(list(map(str, self.__q_layer.layer_weight.flatten().tolist()))) + \
                        "\nK_WEIGHTS\u00A0" + " ".join(list(map(str, self.__k_layer.layer_weight.flatten().tolist()))) + \
                        "\nV_WEIGHTS\u00A0" + " ".join(list(map(str, self.__v_layer.layer_weight.flatten().tolist()))) + \
                        "\nQ_BIASES\u00A0" + (" ".join(list(map(str, self.__q_layer.bias_weight.flatten().tolist()))) \
                                                 if self.use_bias is True else "None") + \
                        "\nK_BIASES\u00A0" + (" ".join(list(map(str, self.__k_layer.bias_weight.flatten().tolist()))) \
                                                 if self.use_bias is True else "None") + \
                        "\nV_BIASES\u00A0" + (" ".join(list(map(str, self.__v_layer.bias_weight.flatten().tolist()))) \
                                                 if self.use_bias is True else "None") + \
                        "\n\u00A0" 
        if not filename:
            return write_ret_str
        if filename.find(".cspn") == -1:
            filename += ".cspn"
        with open(filename, "w+") as file:
            file.write(write_ret_str)
            file.close()


    @staticmethod
    def from_save(context: str, file_load: bool = False) -> 'Attention':
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
        Attention
            A new `Attention` layer containing all of the information encoded in the string or file provided.
        """
        def parse_and_return(handled_str: str):
            data_arr = handled_str.splitlines()
            prop_info = data_arr[0].split("\u00A0")[1:]

            q_weights = data_arr[1].split("\u00A0")[1]
            k_weights = data_arr[2].split("\u00A0")[1]
            v_weights = data_arr[3].split("\u00A0")[1]

            q_bias = data_arr[4].split("\u00A0")[1]
            k_bias = data_arr[5].split("\u00A0")[1]
            v_bias = data_arr[6].split("\u00A0")[1]


            
            new_neuron = Attention(d_e := int(prop_info[0]), prop_info[1] == "True", prop_info[2] == "True", parse_opt_info(prop_info[3]))
            new_neuron.__q_layer.layer_weight = np.array(list(map(float, q_weights.split()))).reshape((d_e, d_e))
            new_neuron.__k_layer.layer_weight = np.array(list(map(float, k_weights.split()))).reshape((d_e, d_e))
            new_neuron.__v_layer.layer_weight = np.array(list(map(float, v_weights.split()))).reshape((d_e, d_e))

            new_neuron.__q_layer.bias_weight = np.array(list(map(float, q_bias.split()))).reshape((d_e, 1)) \
                                             if q_bias != "None" else None
            new_neuron.__k_layer.bias_weight = np.array(list(map(float, k_bias.split()))).reshape((d_e, 1)) \
                                             if k_bias != "None" else None
            new_neuron.__v_layer.bias_weight = np.array(list(map(float, v_bias.split()))).reshape((d_e, 1)) \
                                             if v_bias != "None" else None
            return new_neuron

        if file_load:
            full_parse_str = None
            with open(context, "r") as file:
                full_parse_str = file.read()
                file.close()
            return parse_and_return(full_parse_str)
        return parse_and_return(context)
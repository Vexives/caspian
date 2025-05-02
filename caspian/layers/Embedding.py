from ..cudalib import np
from . import Layer
from ..optimizers import Optimizer, StandardGD, parse_opt_info
from ..utilities import check_types

class Embedding(Layer):
    """
    A token embedding layer which performs a linear transformation of the input data.

    Creates an embedding table of shape `(V, E)` where `V` is the vocab length and `E` is the
    embedding size.

        
    Attributes
    ---------
    v_len : int
        The vocabulary length of this layer. Designates how many word or token embeddings
        are registered as weights in the table.
    e_len : int
        The number of embedded values for each token registered in the embedding table.
    embed_table : ndarray
        The `(V, E)` shape array of weights that are matrix multiplied by the input
        to grant the desired output shape. Trainable parameters that are modified
        after each backward pass.
    opt : Optimizer
        The provided optimizer which modifies the learning gradient before updating weights.


    Examples
    --------
    >>> layer1 = Embedding(10, 5)
    >>> in_arr = np.random.randint(0, 10, (12, 10))
    >>> out_arr = layer1(in_arr)
    >>> print(out_arr.shape)
    (12, 5)
    """
    @check_types([
                  ("vocab_len", lambda x: x > 0, "Argument \"vocab_len\" must be greater than 0."),
                  ("embed_size", lambda x: x > 0, "Argument \"embed_size\" must be greater than 0.")
                  ])
    def __init__(self, vocab_len: int, embed_size: int,
                 optimizer: Optimizer = StandardGD()):
        """
        Initializes an `Embedding` layer using given parameters.

        Parameters
        ----------
        v_len : int
            The vocabulary length of this layer. Designates how many word or token embeddings
            are registered as weights in the table.
        e_len : int
            The number of embedded values for each token registered in the embedding table.
        optimizer : Optimizer, default: StandardGD()
            An optimizer class which processes given loss gradients and adjusts them to match a desired 
            gradient descent path.
        """
        super().__init__(None, None)
        self.v_len = vocab_len
        self.e_len = embed_size
        self.embed_table = np.random.uniform(-0.5, 0.5, (vocab_len, embed_size))
        self.opt = optimizer
    

    def forward(self, data: np.ndarray, training: bool = False) -> np.ndarray:
        """
        Performs a forward propagation pass through this layer given the current table weights.
        
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
            The forward propagated array with added embedding information.
        """
        if training:
            self.__last_in = data
        return data @ self.embed_table
    

    def backward(self, cost_err: np.ndarray) -> np.ndarray:
        """
        Performs a backward propagation pass through this layer and updates the table weights according 
        to the provided learning gradient.

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
        ret_grad = cost_err @ self.embed_table.T
        self.embed_table += self.__last_in.T @ self.opt.process_grad(cost_err)
        return ret_grad
    

    def step(self) -> None:
        """Adds one step to this layer's optimizer and scheduler."""
        self.opt.step()
    

    def clear_grad(self) -> None:
        """Clears the optimizer gradient history and deletes any data required by the backward pass."""
        self.__last_in = None
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
    

    def deepcopy(self) -> 'Embedding':
        """Creates a new deepcopy of this layer with the exact same weights and parameters."""
        new_neuron = Embedding(self.v_len, self.e_len, self.opt.deepcopy())
        new_neuron.embed_table = self.embed_table.copy()
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
        write_ret_str = f"Embedding\u00A0{self.v_len}\u00A0{self.e_len}\u00A0{repr(self.opt)}\n" + \
                        " ".join(list(map(str, self.embed_table.flatten().tolist()))) + "\n\u00A0" 
        if not filename:
            return write_ret_str
        if filename.find(".cspn") == -1:
            filename += ".cspn"
        with open(filename, "w+") as file:
            file.write(write_ret_str)
            file.close()


    @staticmethod
    def from_save(context: str, file_load: bool = False) -> 'Embedding':
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
        Embedding
            A new `Embedding` layer containing all of the information encoded in the string or file provided.
        """
        def parse_and_return(handled_str: str):
            data_arr = handled_str.splitlines()
            prop_info = data_arr[0].split("\u00A0")
            weight_info = data_arr[-2].strip().split()

            v_len, e_len = int(prop_info[1]), int(prop_info[2])
            table = np.array(list(map(float, weight_info))).reshape((v_len, e_len))
            opt = parse_opt_info(prop_info[-1])

            new_neuron = Embedding(v_len, e_len, opt)
            new_neuron.embed_table = table
            return new_neuron

        if file_load:
            full_parse_str = None
            with open(context, "r") as file:
                full_parse_str = file.read()
                file.close()
            return parse_and_return(full_parse_str)
        return parse_and_return(context)
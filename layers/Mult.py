from caspian.cudalib import np
from . import Layer

class Mult(Layer):
    '''
    A simple operation layer used to gain the element-wise multiplication of any number of arrays.

    On a backward pass, will return the appropriate gradients respectful to each of the inputs from
    the previous forward pass.

    Notes
    -----
    This layer can NOT be put into a standard Sequence layer, as it requires more than one input
    to function properly. A custom model which incorporates this layer must be created for it to
    function in a Sequence.

    This layer can also not be saved or loaded from a file, as it does not take any parameters.
    '''
    def __init__(self):
        """
        Initializes a `Mult` layer without parameters.
        """
        super().__init__(None, None)


    def __call__(self, data: tuple[np.ndarray, ...], training: bool = False) -> np.ndarray:
        """Calls the class forward function and provides the given parameters."""
        return self.forward(data, training)


    def forward(self, data: tuple[np.ndarray, ...], training: bool = False) -> np.ndarray:
        """
        Performs a forward propagation pass through this layer with the given input data.
        
        Parameters
        ----------
        data : tuple[ndarray]
            The data arrays that the forward pass will be performed on. Each array must have the
            same size as the others, as it is an element-wise operation.
        training : bool
            Specify whether the layer is currently training or not to save the necessary information
            required for the backward pass.
        
        Returns
        -------
        ndarray
            The forward propagated array with the shape equal to this layer's output shape.
        """
        full_arr = np.array(data)
        if training:
            self.last_ins = full_arr
        return np.prod(full_arr, axis=0)
    

    def backward(self, cost_err: np.ndarray) -> tuple[np.ndarray, ...]:
        """
        Returns the provided gradient, as there is no change for an addition operation.

        Parameters
        ----------
        cost_err : ndarray
            The learning gradient that will be processed and returned.

        Returns
        -------
        ndarray
            The given learning gradient.
        """
        return tuple(np.split(self.last_ins * cost_err, self.last_ins.shape[0]))
    

    def deepcopy(self):
        """Creates a new deepcopy of this layer."""
        return Mult()
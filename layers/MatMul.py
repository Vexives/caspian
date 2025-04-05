from caspian.cudalib import np
from . import Layer

class MatMul(Layer):
    '''
    A simple operation layer used to gain the matrix multiplication result of two arrays. Takes any 
    shape for both arrays, but the shapes must be compatible for the multiplication.

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
        Initializes a `MatMul` layer with given axes parameter.
        """
        super().__init__(None, None)


    def __call__(self, data: tuple[np.ndarray, np.ndarray], training: bool = False) -> np.ndarray:
        """Calls the class forward function and provides the given parameters."""
        return self.forward(data, training)


    def forward(self, data: tuple[np.ndarray, np.ndarray], training: bool = False) -> np.ndarray:
        """
        Performs a forward propagation pass through this layer with the given input data.
        
        Parameters
        ----------
        data : tuple[ndarray, ...]
            The data arrays that the forward pass will be performed on. Each array must have the
            same size as the others, as it is an element-wise operation.
        training : bool
            Specify whether the layer is currently training or not to save the necessary information
            required for the backward pass.
        
        Returns
        -------
        ndarray
            The forward propagated array with the shape equal to this layer's output shape.

        Raises
        ------
        AssertionError
            If the size of the data tuple is not equal to 2.
        """
        assert len(data) == 2, f"Only two arrays are permitted for this layer. - Size: {len(data)}"
        if training:
            self.last_ins = data
        return data[0] @ data[1]
    

    def backward(self, cost_err: np.ndarray) -> tuple[np.ndarray, ...]:
        """
        Returns the provided gradient, as there is no change for an addition operation.

        Parameters
        ----------
        cost_err : ndarray
            The learning gradient that will be processed and returned.

        Returns
        -------
        tuple[ndarray, ...]
            The given learning gradient.
        """
        return (cost_err @ np.moveaxis(self.last_ins[1], -1, -2), 
                np.moveaxis(self.last_ins[0], -1, -2) @ cost_err)
    

    def deepcopy(self):
        """Creates a new deepcopy of this layer."""
        return MatMul()
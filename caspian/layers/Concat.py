from ..cudalib import np
from . import Layer
from ..utilities import check_types, InvalidDataException

class Concat(Layer):
    '''
    A simple operation layer used to concatenate any number of arrays across a certain axis.
    The given arrays must be the same shape EXCEPT for the axis of concatenation.

    On a backward pass, will return the split gradients respectful to each of the inputs from
    the previous forward pass.

    Notes
    -----
    This layer can NOT be put into a standard Sequence layer, as it requires more than one input
    to function properly. A custom model which incorporates this layer must be created for it to
    function in a Sequence.

    This layer can also not be saved or loaded from a file.

    Attributes
    ----------
    axis : int
        The axis at which each given array will be concatenated.
    '''
    @check_types()
    def __init__(self, axis: int = 0):
        """
        Initializes a `Concat` layer with given axis parameter.

        Parameters
        ----------
        axis : int
            The axis at which each given array will be concatenated.
        """
        self.axis = axis
        super().__init__(None, None)


    def __call__(self, data: tuple[np.ndarray, ...], training: bool = False) -> np.ndarray:
        """Calls the class forward function and provides the given parameters."""
        return self.forward(data, training)


    def forward(self, data: tuple[np.ndarray, ...], training: bool = False) -> np.ndarray:
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
        InvalidDataException
            If the data provided is not in tuple format or has a size of less than 2.
        """
        if not isinstance(data, tuple) or len(data) < 2:
            raise InvalidDataException("Must have more than one array and in tuple form.")
        if training:
            self.__last_ins = data
        return np.concatenate(data, axis=self.axis)
    

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
        indexes = np.cumsum(list(map(lambda x: x.shape[self.axis], self.__last_ins)))
        return tuple(np.split(cost_err, indexes, axis=self.axis))
    

    def deepcopy(self):
        """Creates a new deepcopy of this layer."""
        return Concat(self.axis)
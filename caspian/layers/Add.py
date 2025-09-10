from ..cudalib import np
from . import Layer
from ..utilities import InvalidDataException

class Add(Layer):
    '''
    A simple operation layer used to add arrays any number of arrays together.

    On a backward pass, will return the given gradient with no other modifications.

    Notes
    -----
    This layer can NOT be put into a standard Sequence layer, as it requires more than one input
    to function properly. A custom model which incorporates this layer must be created for it to
    function in a Sequence.

    This layer can also not be saved or loaded from a file, as it does not take any parameters.
    '''
    def __init__(self):
        """
        Initializes an `Add` layer without parameters.
        """
        super().__init__(None, None)


    def __call__(self, data: tuple[np.ndarray, ...], *_) -> np.ndarray:
        """Calls the class forward function and provides the given parameters."""
        return self.forward(data)


    def forward(self, data: tuple[np.ndarray, ...], *_):
        """
        Performs a forward propagation pass through this layer with the given input data.
        
        Parameters
        ----------
        data : tuple[ndarray, ...]
            The data arrays that the forward pass will be performed on. Each array must have the
            same size as the others, as it is an element-wise operation.
        
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
        return sum(data)
    

    def deepcopy(self):
        """Creates a new deepcopy of this layer."""
        return Add()